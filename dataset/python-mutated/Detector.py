import os
import time
import cv2
import numpy as np
from PIL import Image
import jetson.inference
import jetson.utils
NET = None

def tile_image(image, xPieces, yPieces):
    if False:
        while True:
            i = 10
    im = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    (imgwidth, imgheight) = im.size
    width = imgwidth // xPieces
    height = imgheight // yPieces
    crops = []
    for i in range(0, yPieces):
        for j in range(0, xPieces):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            crop = np.array(im.crop(box))
            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            crops.append(crop)
    return crops

def filter_detection(detection):
    if False:
        i = 10
        return i + 15
    return detection.ClassID == 1 and 100 < detection.Area < 10000

def load_net(net_name, threshold):
    if False:
        return 10
    global NET
    NET = jetson.inference.detectNet(net_name, threshold=threshold)

def run_inference(image, width_tiles, height_tiles):
    if False:
        i = 10
        return i + 15
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_np = np.asarray(image)
    tile_width = image_np.shape[1] // width_tiles
    tile_height = image_np.shape[0] // height_tiles
    channels = image_np.shape[2]
    all_detections = []
    for (tile_index, tile) in enumerate(tile_image(image_np, width_tiles, height_tiles)):
        image_cuda = jetson.utils.cudaFromNumpy(tile)
        detections = NET.Detect(image_cuda, overlay='none')
        all_detections.extend(detections)
        row = tile_index // width_tiles
        col = tile_index % width_tiles
        for detection in detections:
            detection.Left += col * tile_width
            detection.Top += row * tile_height
            detection.Right += col * tile_width
            detection.Bottom += row * tile_height
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    all_detections = list(filter(filter_detection, all_detections))
    for detection in all_detections:
        cv2.rectangle(image_np, (int(detection.Left), int(detection.Top)), (int(detection.Right), int(detection.Bottom)), color=(255, 255, 255), thickness=2)
    return (image_np, all_detections)