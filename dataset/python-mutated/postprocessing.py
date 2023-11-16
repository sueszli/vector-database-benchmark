import os
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import cv2
from utils import get_crop_pad_sequence, run_length_decoding
import settings

def resize_image(image, target_size):
    if False:
        while True:
            i = 10
    resized_image = cv2.resize(image, target_size)
    return resized_image

def crop_image(image, target_size):
    if False:
        print('Hello World!')
    (top_crop, right_crop, bottom_crop, left_crop) = get_crop_pad_sequence(image.shape[0] - target_size[0], image.shape[1] - target_size[1])
    cropped_image = image[top_crop:image.shape[0] - bottom_crop, left_crop:image.shape[1] - right_crop]
    return cropped_image

def binarize(image, threshold):
    if False:
        i = 10
        return i + 15
    image_binarized = (image > threshold).astype(np.uint8)
    return image_binarized

def save_pseudo_label_masks(submission_file):
    if False:
        print('Hello World!')
    df = pd.read_csv(submission_file, na_filter=False)
    print(df.head())
    img_dir = os.path.join(settings.TEST_DIR, 'masks')
    for (i, row) in enumerate(df.values):
        decoded_mask = run_length_decoding(row[1], (101, 101))
        filename = os.path.join(img_dir, '{}.png'.format(row[0]))
        rgb_mask = cv2.cvtColor(decoded_mask, cv2.COLOR_GRAY2RGB)
        print(filename)
        cv2.imwrite(filename, decoded_mask)
        if i % 100 == 0:
            print(i)
if __name__ == '__main__':
    save_pseudo_label_masks('V456_ensemble_1011.csv')