import numpy as np
import cv2
import math
import mido
from skimage import feature
from skimage.transform import resize
from sklearn.cluster import AffinityPropagation
from itertools import cycle

NUM_SAMPLES_TO_STORE = 5
kernel = np.ones((10, 10), np.uint8)
cap = cv2.VideoCapture(0)
port = mido.open_output()
historical_positions = []

while True:
    ret, img = cap.read()

    # convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # mask of green (36,0,0) ~ (70, 255,255)
    mask = cv2.inRange(hsv, (150, 100, 100), (255, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # slice the green
    imask = mask > 0

    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # clustering
    # we resize image to make it process faster
    mask_small = resize(
        mask, (mask.shape[0] // 4, mask.shape[1] // 4), mode="constant", anti_aliasing=False)
    preview = resize(
        green, (green.shape[0] // 4, green.shape[1] // 4), mode="constant", anti_aliasing=False)
    blobs = feature.blob_dog(mask_small, threshold=.5,
                             min_sigma=0.5, max_sigma=20)

    # clustering
    af = AffinityPropagation(
        preference=-50).fit([[x, y] for y, x, sigma in blobs])
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)

    colors = cycle([(255, 0, 0), (0, 255, 0), (0, 0, 255)])
    kolors = [c for _, c in zip(range(n_clusters_), colors)]

    # output blobs
    for (label, (y, x, sigma)) in zip(labels, blobs):
        cv2.circle(preview, (int(x), int(y)), 4, kolors[label], -1)

    # store last NUM_SAMPLES_TO_STORE blob locations
    historical_positions.append(
        [[x, y] for y, x, sigma in blobs]
    )
    historical_positions = historical_positions[-NUM_SAMPLES_TO_STORE:]

    # show
    cv2.imshow("frame", preview)

    # send MIDI data
    distances = []
    blobs_to_measure = blobs.tolist()
    while len(blobs_to_measure) > 1:
        blob_a = blobs_to_measure.pop()
        for blob_b in blobs:
            distances.append(math.sqrt(pow(blob_a[0] - blob_a[0], 2) +
                                       pow(blob_a[1] - blob_b[1], 2)))

    max_distance = math.sqrt(
        pow(mask_small.shape[0], 2) + pow(mask_small.shape[1], 2))
    average_distance = sum(distances) / \
        len(distances) if len(distances) else max_distance

    centroids = []
    for positions in historical_positions:
        x_sum = 0
        y_sum = 0

        for x, y in positions:
            x_sum += x
            y_sum += y

        centroids.append([x_sum, y_sum])

    mean_centroid = [
        sum([x for x, y in centroids]) / len(centroids),
        sum([y for x, y in centroids]) / len(centroids),
    ]

    centroid_errors = [abs(x - mean_centroid[0]) +
                       abs(y - mean_centroid[1]) for x, y in centroids]

    sum_centroid_errors = sum(centroid_errors)

    msg = mido.Message('control_change', channel=0, control=1, value=int(
        average_distance / max_distance * 127))
    port.send(msg)

    msg = mido.Message('control_change', channel=0,
                       control=2, value=len(blobs))
    port.send(msg)

    msg = mido.Message('control_change', channel=0,
                       control=3, value=min(127, int(sum_centroid_errors / 1500 * 127)))
    port.send(msg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
