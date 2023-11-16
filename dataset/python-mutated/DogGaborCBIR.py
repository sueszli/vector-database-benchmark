import math
import operator
from os import path
from os import listdir
import numpy as np
from six.moves import cPickle
from skimage.io import imread
import cv2
import tensorflow as tf
import scipy.spatial.distance as distance
import Utils
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_feature(image_path, seg_model):
    if False:
        i = 10
        return i + 15
    kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    kernel /= math.sqrt((kernel * kernel).sum())
    image_og = imread(image_path)
    mask = Utils.segment_dog(image_og, seg_model)
    result = cv2.bitwise_and(image_og, image_og, mask=np.int8(mask))
    cv2.imshow('og', image_og)
    cv2.imshow('seg', result)
    filtered_img = cv2.filter2D(result, cv2.CV_8UC3, kernel)
    cv2.imshow('feature', filtered_img)
    descriptor = cv2.resize(filtered_img, (256, 256), interpolation=cv2.INTER_CUBIC)
    feature = np.hstack(descriptor)
    return feature

def extract_texture_features(pathIn, seg_model):
    if False:
        while True:
            i = 10
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
                        feature = get_feature(image_path, seg_model)
                        data_feature = dict()
                        data_feature['feature'] = feature
                        data[image_path] = data_feature
    return data

def getQuery(image_path, seg_model):
    if False:
        i = 10
        return i + 15
    query = get_feature(image_path, seg_model)
    return query

def descriptorDistance(descriptor, inputed_descriptor):
    if False:
        print('Hello World!')
    return abs(distance.euclidean(descriptor.flatten(), inputed_descriptor.flatten()))

def predict(image_path=None):
    if False:
        return 10
    seg_model = tf.keras.models.load_model('segementation_model.h5')
    with open('texture_features.pickle', 'rb') as handle:
        data = cPickle.load(handle)
    if image_path == None:
        image_path = Utils.get_random()
    query = getQuery(image_path, seg_model)
    distances = dict()
    for id in data.keys():
        descriptor = data[id]['feature']
        distances[id] = descriptorDistance(query, descriptor)
    distances = sorted(distances.items(), key=operator.itemgetter(1))
    result_images = []
    for i in range(6):
        result_images.append(distances[i][0])
    return (image_path, result_images)

def get_train_features():
    if False:
        i = 10
        return i + 15
    train_src = 'dogImages/train'
    seg_model = tf.keras.models.load_model('segementation_model.h5')
    texture_features = extract_texture_features(train_src, seg_model)
    print('Creating pickle..')
    with open('texture_features.pickle', 'wb') as handle:
        cPickle.dump(texture_features, handle, protocol=cPickle.HIGHEST_PROTOCOL)
if __name__ == '__main__':
    get_train_features()
    predict()