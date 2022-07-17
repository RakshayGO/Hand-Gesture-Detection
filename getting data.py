import cv2
from math import ceil
import numpy as np
import time
from cvzone.HandTrackingModule import HandDetector
cap = cv2.VideoCapture(0) # 0 by default points to the webcam, but we'll change its address
address = 'https://192.168.1.34:8080/video'   # linking the phone camera
cap.open(address)
detector = HandDetector(maxHands=1)
imgsize = 300
offset = 20

folder = "DATA/paper"
counter = 0
while True:
    success, img = cap.read()
    # downsizing the image using new width and height
    down_width = 800
    down_height = 450
    down_points = (down_width, down_height)
    small_img = cv2.resize(img, down_points, interpolation=cv2.INTER_LINEAR)

    # img = cv2.flip(img, 1)  # to flip it, but then it messes up right and left
    hands, img = detector.findHands(small_img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox'] # gives us bounding box dimensions
        imgCrop = img[y - offset:y+h + offset, x - offset:x+w + offset]

        img_white = np.ones((imgsize, imgsize, 3), np.uint8)*255  # a constant size bg

        aspect_ratio = h/w

        # we need to center the image and resize it so that most of the white is occupied by the cropped image

        if aspect_ratio >= 1:
            k = imgsize/h
            wCal = ceil(w * k)
            img_resize = cv2.resize(imgCrop, (wCal, imgsize))  # since h>w, h = imgsize and w = proportional w
            img_new_shape = img_resize.shape
            wgap = ceil((imgsize - wCal) / 2)   # gap on either size of the smaller dimension
            img_white[:, wgap:wgap + wCal] = img_resize
            # white image is treated like a matrix,and cropped image is filled
        # same if w > h
        else:
            k = imgsize/w
            hCal = ceil(h * k)
            img_resize = cv2.resize(imgCrop, (imgsize, hCal))
            img_new_shape = img_resize.shape
            hgap = ceil((imgsize - hCal)/2)
            img_white[hgap: hgap + hCal, :] = img_resize

        cv2.imshow("image base:", img_white)

        cv2.imshow("image:", img)
        key = cv2.waitKey(1)  # if you increase the number, it captures lesser FPSs
        if key == ord('s'):
            cv2.imwrite(f'{folder}/sample_{time.time()}.jpg', img_white)
            counter += 1
            print(counter)
    else:
        cv2.imshow("image:", img)
        key = cv2.waitKey(1)  # if you increase the number, it captures lesser FPS
