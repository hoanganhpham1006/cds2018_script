import cv2
import numpy as np
from keras.models import load_model

model = load_model("/home/hieung1707/catkin_ws/src/video_stream_python/scripts/updated.h5")
# lowerBound = np.array([98, 109, 50])
# upperBound = np.array([112, 255, 255])
lowerBound = np.array([90, 105, 90])
upperBound = np.array([220, 255, 255])

kernelOpen = np.ones((5, 5))
kernelClose = np.ones((3, 3))
model.predict(np.zeros((1, 24, 24, 1)))

def detect(img):
    img_h, img_w, _ = img.shape
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, lowerBound, upperBound)
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)
    _, conts, _ = cv2.findContours(maskClose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(conts) != 0:
        x, y, w, h = cv2.boundingRect(conts[0])
        crop = img[max(0, y - 5):min(y + h + 5, img_h), max(0, x - 5): min(x + w + 5, img_w)]
        crop = cv2.resize(crop, (24, 24))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop = np.expand_dims(crop, 0)
        crop = np.expand_dims(crop, 3)
        pred = model.predict(crop)[0][0]
        s = w*h
        if pred >= 0.8:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return -1, s
        elif pred <= 0.2:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return 1, s
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            return 0, 1
    return 0, 0

