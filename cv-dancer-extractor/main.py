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

NUM_SAMPLES_TO_STORE = 5
kernel = np.ones((10, 10), np.uint8)
cap = cv2.VideoCapture(1)
port = mido.open_output()
historical_positions = []

prev_ccs: Dict[ChannelControl, int] = {}

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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
