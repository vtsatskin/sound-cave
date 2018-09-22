import numpy as np
import cv2

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

    # save
    cv2.imshow("frame", green)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # FF2D50

cap.release()
cv2.destroyAllWindows()
