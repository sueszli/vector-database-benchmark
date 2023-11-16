'''
This script takes a set of ground truth data, detections and the viewpoint for cars as given in the KITTI object dataset
and segments the cars, resizes them to 128x128 and stores them along with their angles.
This is required for training the CNN that predicts viewpoints.
'''

import numpy as np
import math
import cv2
import os
import pandas
import csv
from os import listdir
from os.path import isfile, join
from scipy import misc
import matplotlib.pyplot as plt

# PATHS
DETECTIONS_PATH = '/home/ubuntu/csc420_data/object_label/training/label_2'

OUTPUT_PATH = '/home/ubuntu/csc420_data/segmented_cars'

IMAGES_PATH = '/home/ubuntu/csc420_data/data_object/training/image_2'

# Get filenames
detection_files = [DETECTIONS_PATH + os.sep + f for f in listdir(DETECTIONS_PATH) if isfile(join(DETECTIONS_PATH, f))]

image_files = [IMAGES_PATH + os.sep + f for f in listdir(IMAGES_PATH) if isfile(join(IMAGES_PATH, f))]

image_files.sort()
detection_files.sort()

all_cars = []
counter = 0
angles = []
# We are binnign the angles into 30degree segments, these are the boundaries of the bins.
bins = np.arange(-180,181,30)
for i, f in zip(image_files, detection_files):
    print(i,f)
    img = misc.imread(i)
    dataframe = pandas.read_csv(f, header=0, sep=' ')
    dataset = dataframe.values
    cars = []
    for d in dataset:
        print(d)
        # If the object is a car.
        if d[0] == 'Car':
            # Segment it.
            angle = float(d[14])
            bin = np.digitize(np.degrees(angle), bins)
            segmented = img[int(float(d[5])):int(float(d[7])), int(float(d[4])):int(float(d[6])), :]
            segmented = misc.imresize(segmented, (128,128), interp='bilinear', mode=None)
            # plt.imshow(segmented)
            # plt.show()
            # Dump the resized image into a folder structure arranged by bins.
            os.makedirs(OUTPUT_PATH + os.sep + str(bin) + os.sep, exist_ok=True)
            misc.imsave(OUTPUT_PATH + os.sep + str(bin) + os.sep + str(counter) + '.png', segmented)
            counter += 1
            angles.append((counter, d[14]))

# Ouput all angles to a csv.
with open(OUTPUT_PATH + os.sep + 'angles.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(angles)

