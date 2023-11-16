"""
This script takes a set of DPM outputs for cars detected and runs them through a CNN that classifies the
detected cars by their viewpoint / angle and outputs that data to a CSV.
"""
import cv2
import os
import sys
import scipy.io as sio
from keras.models import load_model
import numpy as np
from scipy.misc import imsave, imread
import csv
DETECTIONS_DIR = '/Users/B-1P/Downloads/data_road/detections'
IMAGES_DIR = '/Users/B-1P/Downloads/data_road/testing/image_2'
OUTPUT_DIR = ''
image_files = [os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR) if os.path.isfile(os.path.join(IMAGES_DIR, f))]
detection_files = [os.path.join(DETECTIONS_DIR, f) for f in os.listdir(DETECTIONS_DIR) if os.path.isfile(os.path.join(DETECTIONS_DIR, f))]
image_files.sort()
detection_files.sort()
model = load_model('object_classifierfit2.h5')

def filter_detections(mat, threshold=-0.7):
    if False:
        while True:
            i = 10
    '\n\tFilter DPM detections by threshold and return them as a list of lists.\n\t'
    detections = []
    for m in mat['ds']:
        if m[5] > threshold:
            detections.append(m)
    return detections
for (i, d) in zip(image_files, detection_files):
    angles = []
    print(i, d)
    filtered = filter_detections(sio.loadmat(d))
    image = imread(i)
    for d in filtered:
        print(d)
        print(image.shape)
        segment = image[d[1]:d[3], d[0]:d[2], :]
        if segment.shape[0] == 0 or segment.shape[1] == 0:
            continue
        segment = cv2.resize(segment, (128, 128))
        segment = np.array([segment])
        segment = segment / 255.0
        preditction = model.predict(segment)
        print('Prediction', preditction)
        angle = np.argmax(preditction, axis=1) * 30 - 180
        print('Angle:', angle)
        x = d.tolist()
        x.append(angle[0])
        angles.append(x)
    with open(os.path.basename(i).split('.')[0] + '.csv', 'w') as f:
        w = csv.writer(f)
        w.writerows(angles)