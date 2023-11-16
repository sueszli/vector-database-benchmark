import math
import operator
import tensorflow as tf
import FeatureDescriptor
from os import path
from os import listdir
import numpy as np
from six.moves import cPickle
from skimage.io import imread
import Utils
import cv2 as cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def extract_color_features(pathIn, feature_descriptor, seg_model):
    if False:
        for i in range(10):
            print('nop')
    data = dict()
    if path.exists(pathIn):
        if path.isdir(pathIn):
            print(pathIn)
            dirs = listdir(pathIn)
            for dir in dirs:
                dir_path = path.join(pathIn, dir)
                if path.isdir(dir_path):
                    print(dir_path)
                    imgs = listdir(dir_path)
                    for f in imgs:
                        image_path = path.join(pathIn, dir, f)
                        print(image_path)
                        image = imread(image_path)
                        mask = Utils.segment_dog(image, seg_model)
                        feature = feature_descriptor.histogram(image, mask)
                        data_feature = dict()
                        data_feature['feature'] = feature
                        data[image_path] = data_feature
    return data

def getQuery(image_path, feature_descriptor, seg_model):
    if False:
        while True:
            i = 10
    image = imread(image_path)
    mask = Utils.segment_dog(image, seg_model)
    cv2.imshow('img', image)
    cv2.imshow('mask', mask)
    query = feature_descriptor.histogram(image, mask)
    return query

def calc_distance(features, query):
    if False:
        print('Hello World!')
    return math.sqrt(sum([(x - y) ** 2 for (x, y) in zip(features, query)]))

def predict(image_path=None):
    if False:
        i = 10
        return i + 15
    seg_model = tf.keras.models.load_model('segementation_model.h5')
    feature_descriptor = FeatureDescriptor.FeatureDescriptor((8, 12, 3))
    with open('color_features.pickle', 'rb') as handle:
        data = cPickle.load(handle)
    if image_path == None:
        image_path = Utils.get_random()
    query = getQuery(image_path, feature_descriptor, seg_model)
    distances = dict()
    for id in data.keys():
        dist = calc_distance(data[id]['feature'], query)
        distances[id] = dist
    distances = sorted(distances.items(), key=operator.itemgetter(1))
    result_images = []
    for i in range(6):
        result_images.append(distances[i][0])
    return (image_path, result_images)

def get_train_features():
    if False:
        while True:
            i = 10
    train_src = 'dogImages/train'
    feature_descriptor = FeatureDescriptor.FeatureDescriptor((8, 12, 3))
    seg_model = tf.keras.models.load_model('segementation_model.h5')
    color_features = extract_color_features(train_src, feature_descriptor, seg_model)
    print('Creating pickle..')
    with open('color_features.pickle', 'wb') as handle:
        cPickle.dump(color_features, handle, protocol=cPickle.HIGHEST_PROTOCOL)
if __name__ == '__main__':
    get_train_features()
    predict()