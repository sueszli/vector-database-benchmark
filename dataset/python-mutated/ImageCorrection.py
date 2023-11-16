"""
This script performs all (TODO Deskewing) steps to improve Tesseract accuracy:
See --> https://github.com/tesseract-ocr/tesseract/wiki/ImproveQuality
"""
from __future__ import print_function
import numpy as np
import cv2 as cv
import json
import math
from pathlib import Path
import re
import os
import sys
import subprocess

def transparent_to_white(image):
    if False:
        print('Hello World!')
    if len(image.shape) <= 2:
        return image
    if image.shape[2] == 4:
        alpha_channel = image[:, :, 3]
        (_, mask) = cv.threshold(alpha_channel, 254, 255, cv.THRESH_BINARY)
        color = image[:, :, :3]
        image = cv.bitwise_not(cv.bitwise_not(color, mask=np.uint8(mask)))
    return image

def remove_shadow(image):
    if False:
        print('Hello World!')
    rgb_planes = cv.split(image)
    result_planes = []
    for plane in rgb_planes:
        dilated_img = cv.dilate(plane, np.ones((5, 5), np.uint8))
        bg_img = cv.medianBlur(dilated_img, 11)
        diff_img = 255 - cv.absdiff(plane, bg_img)
        result_planes.append(diff_img)
    return cv.merge(result_planes)

def detect_rotation(image):
    if False:
        return 10
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    image_y = np.zeros(img_gray.shape[0:2], np.uint8)
    image_y[:, :] = img_gray[:, :, 0]
    image_blurred = cv.GaussianBlur(image_y, (3, 3), 0)
    img_edges = cv.Canny(image_blurred, 250, 250, apertureSize=3)
    lines = cv.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=20)
    test_image = image.copy()
    angles = []
    for line in lines:
        for (x1, y1, x2, y2) in line:
            cv.line(test_image, (x1, y1), (x2, y2), (255, 255, 0), 5)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)
    median_angle = np.median(angles)
    return median_angle

def rotate_image(mat, angle):
    if False:
        i = 10
        return i + 15
    (height, width) = mat.shape[:2]
    image_center = (width / 2, height / 2)
    rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]
    rotated_mat = cv.warpAffine(mat, rotation_mat, (bound_w, bound_h), borderValue=(255, 255, 255))
    return rotated_mat

def get_rotation_data(original_image, rotated_image, angle, output_file):
    if False:
        return 10
    (oh, ow) = original_image.shape[:2]
    (rh, rw) = rotated_image.shape[:2]
    output_data = dict()
    origin = dict()
    origin['x'] = int(rw / 2)
    origin['y'] = int(rh / 2)
    output_data['origin'] = origin
    translation = dict()
    translation['x'] = int((ow - rw) / 2)
    translation['y'] = int((oh - rh) / 2)
    output_data['translation'] = translation
    output_data['degrees'] = int(angle)
    output_data['filename'] = output_file
    return json.dumps(output_data)

def is_face_down(image_path):
    if False:
        print('Hello World!')
    tesseract_output = subprocess.Popen(['tesseract', image_path, '-', '--psm', '0'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1, universal_newlines=True).stdout.read()
    try:
        tesseract_rotation = re.search('(?<=Rotate: )\\d+', tesseract_output).group(0)
        if tesseract_rotation != '0':
            return True
    except Exception as e:
        return False
    return False

def save_image(image, name):
    if False:
        while True:
            i = 10
    if cv.__version__.split('.')[0] == '3':
        cv.imwrite(name, image)
    else:
        cv.imwrite(name, image, [cv.IMWRITE_TIFF_XDPI, 300, cv.IMWRITE_TIFF_YDPI, 300])

def main():
    if False:
        for i in range(10):
            print('nop')
    try:
        src = sys.argv[1]
        original_image = cv.imread(src)
        no_transparent_image = transparent_to_white(original_image)
        rotated_image = no_transparent_image.copy()
        angle = detect_rotation(no_transparent_image)
        if angle != 0.0:
            rotated_image = rotate_image(no_transparent_image, angle)
        shadows_out = remove_shadow(rotated_image)
        shadows_out = cv.copyMakeBorder(shadows_out, 2, 2, 2, 2, cv.BORDER_CONSTANT, value=[1, 0, 0])
        output_file_name = src.split('.')[0] + '-corrected.' + '.'.join(src.split('.')[1:])
        save_image(shadows_out, output_file_name)
        if is_face_down(output_file_name):
            angle += 180
            shadows_out = rotate_image(shadows_out, 180)
            save_image(shadows_out, output_file_name)
        print(get_rotation_data(original_image, rotated_image, angle, output_file_name))
        sys.stdout.flush()
        sys.exit(0)
    except Exception as e:
        sys.stderr.write(str(e))
        sys.stderr.flush()
        sys.exit(1)
if __name__ == '__main__':
    main()