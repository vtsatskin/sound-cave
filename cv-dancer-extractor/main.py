import numpy as np
import cv2
import math
from skimage import feature
from skimage.transform import resize

im_width = 320
im_height = 240
kernel = np.ones((10, 10), np.uint8)
cap = cv2.VideoCapture(0)

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
    mask_small = resize(mask, (mask.shape[0] // 4, mask.shape[1] // 4))
    preview = resize(green, (green.shape[0] // 4, green.shape[1] // 4))
    blobs = feature.blob_dog(mask_small, threshold=.5,
                             min_sigma=0.5, max_sigma=20)

    # output blobs
    for y, x, sigma in blobs:
        cv2.circle(preview, (int(x), int(y)), 4, (0, 255, 0), -1)

    # show
    cv2.imshow("frame", preview)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
