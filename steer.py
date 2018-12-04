import cv2
import numpy as np
import time
total_time = 0
last = 0
flag = 0
class SegmentToSteer():
    def __init__(self, square=3, margin=30, roi=1/3):
        self.square = square
        self.margin = margin
        self.roi = 1-roi
    def get_point(self, img):
        global total_time
        global last
        global flag
        IMG_H, IMG_W = img.shape
        i = int(IMG_H * self.roi)
        border = int((self.square - 1) / 2)
        i_l = border
        i_r = IMG_W - 1 - border
        turn_left = False
        turn_right = False
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
        interval = time.time() - last
        last = time.time()
        if total_time > 0 and total_time < 3:
            total_time += interval
            if flag == 1:
                while img[i][i_r] == 255 and i >= 0:
                    i-=1
                return i, i_r
            elif flag == 2:
                while img[i][i_l] == 255 and i >= 0:
                    i-=1
                return i, i_l
            else:
                return i, int((i_l + i_r) /2)
        else:
            total_time = 0
            if flag == 3:
                flag = 0
        if not turn_left and not turn_right:
            return i, int((i_l + i_r) / 2)
        elif turn_left and turn_right:
            flag += 1
            total_time += interval
            return i, i_r
        elif turn_left:
            while img[i][i_l] == 255 and i >= 0:
                i-=1
            return i, i_l
        else:
            while img[i][i_r] == 255 and i >= 0:
                i-=1
            return i, i_r


    def get_point_2(self, img):
        IMG_H, IMG_W = img.shape
        mid_x = 0
        mid_y = 0
        count = 0
        for i in range(int(IMG_H * self.roi), IMG_H):
            count += 1
            left = 0
            while img[i][left] != 255 and left < IMG_W/2:
                left += 1
            right = IMG_W-1
            while img[i][right] != 255 and right >= IMG_W/2:
                right -= 1
            mid_x += int((left+right)/2)
            mid_y += i
        return int(mid_y/count), int(mid_x/count)

    def get_steer(self, img):
        IMG_H, IMG_W = img.shape
        img = np.asarray(img, np.uint8)
        # img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
        y, x = self.get_point(img)
        cv2.line(img, (x, y), (int(IMG_W / 2), IMG_H - 1), (0, 0, 0), 1)
        steer = np.arctan((x - IMG_W/2 + 1) / (IMG_H - float(y))) * 57.32

        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        bottomLeftCornerOfText = (30, 30)
        fontScale = 0.7
        fontColor = (255, 255, 255)
        lineType = 2

        cv2.putText(img, str(steer), bottomLeftCornerOfText, font, fontScale, fontColor, lineType, 0)
        return steer, img

