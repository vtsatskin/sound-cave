import numpy as np
import cv2
import math
import mido
from skimage import feature
from skimage.transform import resize
from sklearn.cluster import AffinityPropagation
from itertools import cycle
from typing import NamedTuple, Union, Dict, Tuple, List

CHANNEL: int = 0

ChannelControl = NamedTuple("ChannelControl", [
    ('channel', int),
    ('control', int),
])

ControlChange = NamedTuple("ControlChange", [
    ('channel', int),
    ('control', int),
    ('value', int)
])

MetricName = Union[
    "blob_number",
    "blob_average_distance",
    "blob_movement",
    "cluster_number",
    "cluster_average_distance",
    "cluster_distance"
]

Metric = NamedTuple('Metric', [
    ('value', float),
    ('max_value', float),
    ('min_value', float),
])


# MetricName -> (channel, control)
MetricToControlChange: Dict[MetricName, Tuple[ChannelControl]] = {
    "blob_number": (CHANNEL, 1),
    "blob_average_distance": (CHANNEL, 0),
    "blob_movement": (CHANNEL, 2),
    "cluster_number": (CHANNEL, 4),
    "cluster_average_distance": (CHANNEL, 3),
    "cluster_distance": (CHANNEL, 5)
}

SCALE_FACTOR = 4
NUM_SAMPLES_TO_STORE = 5
kernel = np.ones((10, 10), np.uint8)
cap = cv2.VideoCapture(1)
port = mido.open_output()
historical_positions = []

prev_ccs: Dict[ChannelControl, int] = {}

cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.namedWindow("mask", cv2.WND_PROP_FULLSCREEN)

h = 117
s = 111
v = 163

h2 = 255
s2 = 255
v2 = 255


while True:
    ret, img = cap.read()

    # convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # mask of green (36,0,0) ~ (70, 255,255)
    mask = cv2.inRange(hsv, (h, s, v), (h2, s2, v2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # slice the green
    imask = mask > 0

    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # clustering
    # we resize image to make it process faster
    mask_small = resize(
        mask, (mask.shape[0] // SCALE_FACTOR, mask.shape[1] // SCALE_FACTOR), mode="constant", anti_aliasing=False)
    preview = resize(
        green, (green.shape[0] // SCALE_FACTOR, green.shape[1] // SCALE_FACTOR), mode="constant", anti_aliasing=False)
    blobs = feature.blob_dog(mask_small, threshold=.5,
                             min_sigma=0.5, max_sigma=20)

    cv2.imshow("mask", preview)

    # clustering
    labels = []
    n_clusters_ = 0

    X = np.array([[x, y] for y, x, sigma in blobs])
    if len(X) > 0:
        af = AffinityPropagation(preference=-50).fit(X)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        n_clusters_ = len(cluster_centers_indices)

    colors = cycle([(255, 0, 0), (0, 255, 0), (0, 0, 255)])
    kolors = [c for _, c in zip(range(n_clusters_), colors)]

    # draw connections between blobs
    blobs_to_connect = blobs.tolist()
    while len(blobs_to_connect) > 1:
        blob_a = blobs_to_connect.pop()
        for blob_b in blobs:
            cv2.line(img, (int(blob_a[1]) * SCALE_FACTOR, int(blob_a[0]) * SCALE_FACTOR),
                     (int(blob_b[1]) * SCALE_FACTOR, int(blob_b[0]) * SCALE_FACTOR), (0, 0, 0), 1, cv2.LINE_AA)

    # output blobs
    for (label, (y, x, sigma)) in zip(labels, blobs):
        cv2.circle(img, (int(x) * SCALE_FACTOR, int(y) * SCALE_FACTOR),
                   4, (0, 255, 0), -1)

    # store last NUM_SAMPLES_TO_STORE blob locations
    historical_positions.append(
        [[x, y] for y, x, sigma in blobs]
    )
    historical_positions = historical_positions[-NUM_SAMPLES_TO_STORE:]

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

    centroids = [[0, 0]]
    for positions in historical_positions:
        x_sum = 0
        y_sum = 0

        for x, y in positions:
            x_sum += x
            y_sum += y

        if len(positions) > 0:
            centroids.append([x_sum / len(positions), y_sum / len(positions)])

    mean_centroid = [0, 0]
    if len(centroids) > 0:
        mean_centroid = [
            sum([x for x, y in centroids]) / len(centroids),
            sum([y for x, y in centroids]) / len(centroids),
        ]

    cv2.circle(
        img, (int(centroids[-1][0]) * SCALE_FACTOR, int(centroids[-1][1]) * SCALE_FACTOR), 4, (0, 0, 255), -1)

    centroid_errors = [abs(x - mean_centroid[0]) +
                       abs(y - mean_centroid[1]) for x, y in centroids]

    sum_centroid_errors = sum(centroid_errors)

    metrics: Dict[MetricName, Metric] = {
        'blob_number': Metric(
            value=len(blobs),
            max_value=127,
            min_value=0
        ),
        'blob_average_distance': Metric(
            value=average_distance,
            max_value=max_distance,
            min_value=0
        ),
        'blob_movement': Metric(
            value=sum_centroid_errors,
            max_value=1500,
            min_value=0
        )
    }

    cv2.rectangle(img, (0, 0), (500, 150), (0, 0, 0), -1)
    cv2.putText(img, 'Blobs: {}'.format(len(blobs)), (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, 'Blob Average Distance: {0:.2f}'.format(average_distance), (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, 'Blob Movement: {0:.2f}'.format(sum_centroid_errors), (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    ccs: List[ControlChange] = []
    for metric_name, metric in metrics.items():
        channel, control = MetricToControlChange[metric_name]

        cc = ControlChange(
            channel=channel,
            control=control,
            value=max(0, min(127, int((metric.value - metric.min_value) /
                                      metric.max_value * 127)))
        )
        ccs.append(cc)

    for cc in ccs:
        prev_cc = prev_ccs.get((cc.channel, cc.control))
        if prev_cc and prev_cc.value == cc.value:
            # skip ccs which haven't changed
            continue

        msg = mido.Message('control_change', channel=cc.channel,
                           control=cc.control, value=cc.value)
        port.send(msg)

    prev_ccs = dict([((cc.channel, cc.control), cc) for cc in ccs])

    cv2.putText(img, '{},{},{}'.format(h, s, v), (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(img, '{},{},{}'.format(h2, s2, v2), (10, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # show
    cv2.imshow("window", img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('a'):
        h = max(h - 1, 0)

    if key == ord('s'):
        s = max(s - 1, 0)

    if key == ord('d'):
        v = max(v - 1, 0)

    if key == ord('q'):
        h = min(h + 1, 255)

    if key == ord('w'):
        s = min(s + 1, 255)

    if key == ord('e'):
        v = min(v + 1, 255)

    if key == ord('f'):
        h2 = max(h2 - 1, 0)

    if key == ord('g'):
        s2 = max(s2 - 1, 0)

    if key == ord('h'):
        v2 = max(v2 - 1, 0)

    if key == ord('r'):
        h2 = min(h2 + 1, 255)

    if key == ord('t'):
        s2 = min(s2 + 1, 255)

    if key == ord('y'):
        v2 = min(v2 + 1, 255)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()
