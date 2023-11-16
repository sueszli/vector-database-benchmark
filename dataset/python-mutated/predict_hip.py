from __future__ import absolute_import, division, print_function
import cv2
import numpy as np
import tensorflow as tf
import hipCNN
import mnistCNN

def classify_hip(input, classifier):
    if False:
        while True:
            i = 10
    input = np.array(input) / np.float32(255)
    input = input.astype('float32')
    predictions = classifier.predict(input_fn=tf.estimator.inputs.numpy_input_fn(x={'x': input}, num_epochs=1, shuffle=False))
    res = list(predictions)[0]
    return res['classes']

def predict_array(predict_x, classifier):
    if False:
        i = 10
        return i + 15
    predict_x = predict_x / np.float32(255)
    predictions = classifier.predict(input_fn=tf.estimator.inputs.numpy_input_fn(x={'x': predict_x}, num_epochs=1, shuffle=False))
    return list(predictions)[0]['probabilities'][1]

def locate_hip(i):
    if False:
        print('Hello World!')
    hip_classifier = tf.estimator.Estimator(model_fn=hipCNN.cnn_model_fn, model_dir='model')
    y = 0
    x = 0
    step = 9
    width_ratio = 5
    dim = int(i.shape[1] / width_ratio)
    width = i.shape[1]
    height = i.shape[0]
    vals = np.zeros((int((height - dim) / step) + 1, int((width - dim) / step) + 1))
    while y < height - dim:
        x = 0
        while x < width - dim:
            sub_image = i[y:y + dim, x:x + dim]
            sub_image = cv2.resize(sub_image, (28, 28))
            score = predict_array(sub_image, hip_classifier)
            vals[int(y / step), int(x / step)] = score
            x += step
        y += step
    coords = np.unravel_index(np.argmax(vals), vals.shape)
    best_y = coords[0]
    best_x = coords[1]
    best_image = i[best_y * step:best_y * step + dim, best_x * step:best_x * step + dim]
    small_hip = cv2.resize(best_image, (28, 28))
    small_hip = small_hip / np.float32(255)
    prepped_image = np.zeros((28, 28))
    for y in range(0, 28):
        for x in range(0, 28):
            if is_num(small_hip, y, x):
                prepped_image[y, x] = small_hip[y, x]
            else:
                prepped_image[y, x] = 0.8
    mnist_classifier = tf.estimator.Estimator(model_fn=mnistCNN.cnn_model_fn, model_dir='mnist')
    final_result = classify_hip(prepped_image, mnist_classifier)
    return final_result

def is_num(small_hip, y, x):
    if False:
        print('Hello World!')
    threshold = 0.4
    if small_hip[y, x] > threshold:
        return False
    left = x == 0 or np.amax(small_hip[y, 0:x]) > threshold
    right = np.amax(small_hip[y, x:28]) > threshold
    top = y == 0 or np.amax(small_hip[0:y, x]) > threshold
    bottom = np.amax(small_hip[y:28, x]) > threshold
    total = 0
    if left:
        total += 1
    if right:
        total += 1
    if top:
        total += 1
    if bottom:
        total += 1
    return total >= 3