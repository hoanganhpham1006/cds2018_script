import cv2
import numpy as np
import time
from collections import deque
total_time = 0
last = time.time()
pre_flag = 0
pre_turn = False
class SegmentToSteer():
    def __init__(self, square=3, margin=30, roi=1/3, size=10):
        # self.roi_origin = roi
        # self.roi_turn = 0.3
        self.square = square
        self.margin = margin
        self.size = size
        self.memory = deque()
        self.roi = 1 - roi

    def get_point(self, img, flag):
        IMG_H, IMG_W = img.shape
        i = IMG_H - 2

        # i = int(IMG_H * self.roi)
        border = int((self.square - 1) / 2)
        i_l = border
        i_r = IMG_W - 1 - border
        turn_left = False
        turn_right = False
        while i > 0:
            check = img[i - border: i + border + 1, int(IMG_W / 2) - border : int(IMG_W / 2) + border + 1]
            white = np.sum(check) / 255
            if white != self.square ** 2:
                break
            i -= 1
        if i > 0 and i < int(IMG_H * self.roi):
            i = int(IMG_H * self.roi)
        while i_l <= IMG_W / 2:
            check = img[i - border: i + border + 1, i_l - border: i_l + border + 1]
            white = np.sum(check) / 255
            if white == self.square ** 2 and i_l <= self.margin:
                turn_left = True
                break
            elif white == self.square ** 2 and i_l > self.margin:
                break
            else:
                i_l += (border + 1)
        while i_r > IMG_W / 2:
            check = img[i - border: i + border + 1, i_r - border: i_r + border + 1]
            white = np.sum(check) / 255
            if white == self.square ** 2 and i_r >= IMG_W - self.margin:
                turn_right = True
                break
            elif white == self.square ** 2 and i_r < IMG_W - self.margin:
                break
            else:
                i_r -= (border + 1)
        if not turn_left and not turn_right:
            return i, int((i_l + i_r) / 2)
        
        elif turn_left and turn_right:
            if flag == 1:
                while img[i][i_r] == 255 and i >= 0:
                    i-=1
                return i, i_r
            elif flag == -1:
                while img[i][i_l] == 255 and i >= 0:
                    i-=1
                return i, i_l
            else:
                return i, int((i_l + i_r) / 2)
            return i, int((i_l + i_r) /2)
            
            # return i, i_r
        elif turn_left:
            while img[i][i_l] == 255 and i >= 0:
                i-=1
            return i, i_l
        else:
            while img[i][i_r] == 255 and i >= 0:
                i-=1
            return i, i_r

    def get_flag(self):
        arr = np.asarray(self.memory, np.int8)
        return np.bincount(arr).argmax() - 1
    
    def addFlag(self, flag):
        if len(self.memory) >= self.size:
            self.memory.popleft()
        self.memory.append(flag + 1)

    def get_steer(self, img, flag, s):
        global pre_flag
        global total_time
        global last
        IMG_H, IMG_W = img.shape
        interval = time.time() - last
        last = time.time()
        current_flag = 0
        speed = 65
        
        if total_time > 0 and total_time < 3:
            total_time += interval
            # if total_time > 0.25:
            current_flag = pre_flag
        else:
            total_time = 0
            self.addFlag(flag)
            current_flag = self.get_flag()
            if current_flag != 0 and current_flag != pre_flag:
                # self.roi = 1 - self.roi_turn
                pre_flag = current_flag
                total_time += interval
            else:
                # self.roi = 1 - self.roi_origin
                pre_flag = current_flag
        # self.forsee_road(img, current_flag)
        y, x = self.get_point(img, current_flag)
        cv2.line(img, (x, y), (int(IMG_W / 2), IMG_H - 1), (0, 0, 0), 1)
        steer = np.arctan((x - IMG_W/2 + 1) / (IMG_H - float(y))) * 57.32

        if current_flag != 0 or pre_flag != 0:
            speed = 40
        else:
            speed = speed*np.cos(abs(steer)*np.pi/180)

        if steer > 50:
            steer = 50
        elif steer < -50:
            steer = -50                

        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        bottomLeftCornerOfText = (30, 30)
        fontScale = 0.7
        fontColor = (255, 255, 255)
        lineType = 2

        cv2.putText(img, str(steer), bottomLeftCornerOfText, font, fontScale, fontColor, lineType, 0)
        return speed, steer, img

