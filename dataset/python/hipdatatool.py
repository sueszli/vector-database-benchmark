import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io
from scipy import signal
import os
from random import randint
import shutil

# TODO extract all video frames
# for filename in os.listdir(path):
#     if (filename.endswith(".mp4")): #or .avi, .mpeg, whatever.
#         os.system("ffmpeg -i {0} -f image2 -vf fps=fps=1 output%d.png".format(filename))
#     else:
#         continue

class HipMarker:


    # colors = [(0,255, 0), (0,255, 0), (255,0, 0), (255,0, 0), (0,0, 255), (0,0, 255)]
    # color_index = 0

    def __init__(self, path, file, extension):
        self.path = path
        self.file = file
        self.extension = extension
        self.route = self.path + self.file + self.extension
        self.pastVersions = []
        self.boxes = []
        self.pastBoxes = []

    def runImage(self):
        cv2.namedWindow(self.route) 
        cv2.moveWindow(self.route, 40,30)

        self.image = cv2.imread(self.route)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        self.pastVersions.append(self.image.copy())
        self.pastBoxes.append(self.boxes.copy())

        cv2.setMouseCallback(self.route, self.selectBox)
        

        while True:
            self.display_half_size()
            key = cv2.waitKey(1) & 0xFF

            if key == ord("u"):
                # if there are boxes added, remove the most recent one
                if len(self.pastVersions) > 1:

                    #set the image to the version without the box
                    self.image = self.pastVersions[-2].copy()
                    self.boxes = self.pastBoxes[-2].copy()

                    #unstore the last version
                    self.pastVersions = self.pastVersions[:-1]
                    self.pastBoxes = self.pastBoxes[:-1]
                    # self.color_index -= 1

                    self.display_half_size()
            
            # save the selected image
            if key == ord("s"):
                self.save_boxes("train/pos/", "pos", self.boxes)
                self.save_boxes("train/neg/", "neg", self.genNegData())
                shutil.move(self.route, self.path + "marked/" + self.file + self.extension)
                
                self.close_windows()
                return 0

            # "delete" file by putting it in empty folder
            if key == ord("d"):
                shutil.move(self.route, self.path + "empty/" + self.file + self.extension)
                self.close_windows()
                return 0

            # allow quit
            if key == ord("q"):
                self.close_windows()
                return 1

            # skip 10 frames
            if key == ord("x"):
                self.close_windows()
                return 2


    def save_boxes(self, extra_path, extra_name, boxes):
        shot_num = 0
        for box in boxes:
            shot_num+=1
            # select the correct portion of the image
            pic = self.pastVersions[0][box[0][1]:box[1][1], box[0][0]:box[1][0]]
            # file should be in the negative training folder. Named as <mp4_name>_<frame_name>_<shot_nm>.bmp
            filename = self.path + extra_path + self.file + "_" + extra_name + str(shot_num) + ".bmp"
            cv2.imwrite(filename, pic)



    def selectBox(self, event, x, y, flags, param):
        x = x*2
        y = y*2
        if event == cv2.EVENT_LBUTTONDOWN:
            self.newPoint = [(x, y)]

        elif event == cv2.EVENT_LBUTTONUP:
            # create selection with a 1:1 aspect ratio based on the x distance
            old_x = self.newPoint[0][0]
            old_y = self.newPoint[0][1]
            delta_x = x-old_x
            new_y = old_y + delta_x
            if (self.image.shape[0]-1 < new_y): #make sure the y can be set to match the x
                self.newPoint = []
                return

            if (x > old_x):
                self.newPoint.append((x, new_y))
            else:
                self.newPoint = [(x, new_y), self.newPoint[0]]


            pic = self.pastVersions[0][self.newPoint[0][1]:self.newPoint[1][1], self.newPoint[0][0]:self.newPoint[1][0]]
            pic = cv2.resize(pic, (round(pic.shape[1]*3),round(pic.shape[0]*3)))
            cv2.imshow("selection", pic)


            # if (self.color_index < len(self.colors)):
            #     color = self.colors[self.color_index]
            # else:
            #     color = (0,0,0)
            # self.color_index += 1
            cv2.rectangle(self.image, self.newPoint[0], self.newPoint[1], (0,255, 0), 2)
            self.pastVersions.append(self.image.copy())
            self.boxes.append(self.newPoint)
            self.pastBoxes.append(self.boxes)
            self.newPoint = []
            self.display_half_size()


    def genNegData(self):
        neg_boxes = []
        for box in self.boxes:
            for t in range(0,2):
                width = box[1][0] - box[0][0]

                dim = 8

                lower_x = min(box[0][1] - dim * width, 0)
                upper_x = max(box[1][1] + dim*width, self.image.shape[1])
                lower_y = min(box[0][0] - dim * width, 0)
                upper_y = max(box[1][0] + dim*width, self.image.shape[0])

                x = randint(lower_x, upper_x)
                y = randint(lower_y, upper_y)

                #make sure that the box is fully within the image bounds
                if x+width < self.image.shape[1]-1 and y+width < self.image.shape[0]-1 and x >=0 and y >= 0:
                    new_neg_box = [(x,y),(x+width,y+width)]
                    neg_boxes.append(new_neg_box)
                else:
                    break

                # make sure that the data doesn't accidentally select a hip
                for hip in self.boxes:
                    if self.too_close(hip, new_neg_box, width):
                        neg_boxes = neg_boxes[:-1]
                        break
        return neg_boxes



    @staticmethod
    def too_close(box, negBox, width):
        # how many hip number widths away from actual hip numbers the box must be
        alpha = 0.5
        if (abs(box[0][0] - negBox[0][0]) < width*alpha and abs(box[0][1] - negBox[0][1]) < width*alpha):
            return True
        return False

    def display_half_size(self):
        i = cv2.resize(self.image, (round(self.image.shape[1]*0.5),round(self.image.shape[0]*0.5)))
        cv2.imshow(self.route, i)
        
    def close_windows(self):
        # cv2.destroyWindow(self.route)
        cv2.destroyAllWindows()



# path = "../data/finish-line/bmps/"
# file = "220"
# extension = ".bmp"
# ia = HipMarker(path, file, extension)