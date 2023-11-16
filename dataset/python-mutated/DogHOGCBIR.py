import operator
from os import path
from os import listdir
import numpy as np
from six.moves import cPickle
import cv2
import tensorflow as tf
import scipy.spatial.distance as distance
import Utils
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_feature(hog, image_path, seg_model):
    if False:
        return 10
    image_og = cv2.imread(image_path)
    mask = Utils.segment_dog(image_og, seg_model)
    segmented_image = cv2.bitwise_and(image_og, image_og, mask=np.int8(mask))
    resized = cv2.resize(segmented_image, (64, 128), interpolation=cv2.INTER_AREA)
    h = hog.compute(resized)
    hogImage = h.T
    feature = np.hstack(hogImage)
    print(len(feature))
    return feature

def extract_hog_features(pathIn, seg_model):
    if False:
        for i in range(10):
            print('nop')
    data = dict()
    hog = cv2.HOGDescriptor()
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
                        data_feature = dict()
                        feature = get_feature(hog, image_path, seg_model)
                        data_feature['feature'] = feature
                        data[image_path] = data_feature
    return data

def getQuery(image_path, seg_model):
    if False:
        i = 10
        return i + 15
    hog = cv2.HOGDescriptor()
    query = get_feature(hog, image_path, seg_model)
    return query

def descriptorDistance(descriptor, inputed_descriptor):
    if False:
        while True:
            i = 10
    return abs(distance.euclidean(descriptor.flatten(), inputed_descriptor.flatten()))

def predict(image_path=None):
    if False:
        for i in range(10):
            print('nop')
    seg_model = tf.keras.models.load_model('segementation_model.h5')
    with open('hog_features.pickle', 'rb') as handle:
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
    hog_features = extract_hog_features(train_src, seg_model)
    print('Creating pickle..')
    with open('hog_features.pickle', 'wb') as handle:
        cPickle.dump(hog_features, handle, protocol=cPickle.HIGHEST_PROTOCOL)
if __name__ == '__main__':
    get_train_features()
    predict()