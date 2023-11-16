'''
The ground truth images for the road data contain information other than road.
This script creates binary masks for just the road part from the ground truth road provided in KITTI.
'''
import os
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np

# Set up directories
GROUND_TRUTH_DIR = '/home/ubuntu/csc420_data/data_road/training/gt_image_2'
OUTPUT_DIR = '/home/ubuntu/csc420_data/data_road/training/gt_image_road_mask/'

# The colour to segment out.
GROUND_COLOUR = np.array((255, 0, 255))


all_files = [join(GROUND_TRUTH_DIR, f) for f in listdir(GROUND_TRUTH_DIR) if isfile(join(GROUND_TRUTH_DIR, f))]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dump a binary mask for every image.
for i in all_files:
    img = cv2.imread(i)
    print("Processing: %s",i)
    mask = cv2.inRange(img, GROUND_COLOUR, GROUND_COLOUR)
    cv2.imwrite(OUTPUT_DIR + os.sep + os.path.basename(i),mask)
