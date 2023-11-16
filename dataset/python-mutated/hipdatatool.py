import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io
from scipy import signal
import os
from random import randint
import shutil

class HipMarker:

    def __init__(self, path, file, extension):
        if False:
            while True:
                i = 10
        self.path = path
        self.file = file
        self.extension = extension
        self.route = self.path + self.file + self.extension
        self.pastVersions = []
        self.boxes = []
        self.pastBoxes = []

    def runImage(self):
        if False:
            print('Hello World!')
        cv2.namedWindow(self.route)
        cv2.moveWindow(self.route, 40, 30)
        self.image = cv2.imread(self.route)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.pastVersions.append(self.image.copy())
        self.pastBoxes.append(self.boxes.copy())
        cv2.setMouseCallback(self.route, self.selectBox)
        while True:
            self.display_half_size()
            key = cv2.waitKey(1) & 255
            if key == ord('u'):
                if len(self.pastVersions) > 1:
                    self.image = self.pastVersions[-2].copy()
                    self.boxes = self.pastBoxes[-2].copy()
                    self.pastVersions = self.pastVersions[:-1]
                    self.pastBoxes = self.pastBoxes[:-1]
                    self.display_half_size()
            if key == ord('s'):
                self.save_boxes('train/pos/', 'pos', self.boxes)
                self.save_boxes('train/neg/', 'neg', self.genNegData())
                shutil.move(self.route, self.path + 'marked/' + self.file + self.extension)
                self.close_windows()
                return 0
            if key == ord('d'):
                shutil.move(self.route, self.path + 'empty/' + self.file + self.extension)
                self.close_windows()
                return 0
            if key == ord('q'):
                self.close_windows()
                return 1
            if key == ord('x'):
                self.close_windows()
                return 2

    def save_boxes(self, extra_path, extra_name, boxes):
        if False:
            print('Hello World!')
        shot_num = 0
        for box in boxes:
            shot_num += 1
            pic = self.pastVersions[0][box[0][1]:box[1][1], box[0][0]:box[1][0]]
            filename = self.path + extra_path + self.file + '_' + extra_name + str(shot_num) + '.bmp'
            cv2.imwrite(filename, pic)

    def selectBox(self, event, x, y, flags, param):
        if False:
            for i in range(10):
                print('nop')
        x = x * 2
        y = y * 2
        if event == cv2.EVENT_LBUTTONDOWN:
            self.newPoint = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            old_x = self.newPoint[0][0]
            old_y = self.newPoint[0][1]
            delta_x = x - old_x
            new_y = old_y + delta_x
            if self.image.shape[0] - 1 < new_y:
                self.newPoint = []
                return
            if x > old_x:
                self.newPoint.append((x, new_y))
            else:
                self.newPoint = [(x, new_y), self.newPoint[0]]
            pic = self.pastVersions[0][self.newPoint[0][1]:self.newPoint[1][1], self.newPoint[0][0]:self.newPoint[1][0]]
            pic = cv2.resize(pic, (round(pic.shape[1] * 3), round(pic.shape[0] * 3)))
            cv2.imshow('selection', pic)
            cv2.rectangle(self.image, self.newPoint[0], self.newPoint[1], (0, 255, 0), 2)
            self.pastVersions.append(self.image.copy())
            self.boxes.append(self.newPoint)
            self.pastBoxes.append(self.boxes)
            self.newPoint = []
            self.display_half_size()

    def genNegData(self):
        if False:
            i = 10
            return i + 15
        neg_boxes = []
        for box in self.boxes:
            for t in range(0, 2):
                width = box[1][0] - box[0][0]
                dim = 8
                lower_x = min(box[0][1] - dim * width, 0)
                upper_x = max(box[1][1] + dim * width, self.image.shape[1])
                lower_y = min(box[0][0] - dim * width, 0)
                upper_y = max(box[1][0] + dim * width, self.image.shape[0])
                x = randint(lower_x, upper_x)
                y = randint(lower_y, upper_y)
                if x + width < self.image.shape[1] - 1 and y + width < self.image.shape[0] - 1 and (x >= 0) and (y >= 0):
                    new_neg_box = [(x, y), (x + width, y + width)]
                    neg_boxes.append(new_neg_box)
                else:
                    break
                for hip in self.boxes:
                    if self.too_close(hip, new_neg_box, width):
                        neg_boxes = neg_boxes[:-1]
                        break
        return neg_boxes

    @staticmethod
    def too_close(box, negBox, width):
        if False:
            i = 10
            return i + 15
        alpha = 0.5
        if abs(box[0][0] - negBox[0][0]) < width * alpha and abs(box[0][1] - negBox[0][1]) < width * alpha:
            return True
        return False

    def display_half_size(self):
        if False:
            for i in range(10):
                print('nop')
        i = cv2.resize(self.image, (round(self.image.shape[1] * 0.5), round(self.image.shape[0] * 0.5)))
        cv2.imshow(self.route, i)

    def close_windows(self):
        if False:
            while True:
                i = 10
        cv2.destroyAllWindows()