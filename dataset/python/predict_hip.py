from __future__ import absolute_import, division, print_function
import cv2
import numpy as np
import tensorflow as tf
import hipCNN
import mnistCNN

def classify_hip(input, classifier):
    input = np.array(input)/np.float32(255)
    input = input.astype('float32')
    predictions = classifier.predict(
    input_fn=tf.estimator.inputs.numpy_input_fn(
    x={"x": input},
    num_epochs=1,
    shuffle=False))
    res = list(predictions)[0]
    # print(res['probabilities'])
    return res['classes']


def predict_array(predict_x, classifier):
    predict_x = predict_x/np.float32(255)
    predictions = classifier.predict(
    input_fn=tf.estimator.inputs.numpy_input_fn(
    x={"x": predict_x},
    num_epochs=1,
    shuffle=False))
    return list(predictions)[0]['probabilities'][1]

def locate_hip(i):

    hip_classifier = tf.estimator.Estimator(
        model_fn=hipCNN.cnn_model_fn, model_dir="model")

    # i = cv2.imread(file)
    # i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

    y = 0
    x = 0

    #hyperparameters
    # step sets the step size when scanning the Image
    # width_ratio sets the ratio of the image height to the hip # height
    step = 9
    width_ratio = 5

    dim = int(i.shape[1]/width_ratio)

    width = i.shape[1]
    height = i.shape[0]

    # vals stores the probability that each section of the image has a hip #
    vals = np.zeros((int((height-dim)/step)+1, int((width-dim)/step)+1))

    # iterate down the height of the image
    while y < height-dim:
        # print(y)
        x = 0
        # iterate across the width of the image
        while x < width-dim:
            # a subsection of the image is sampled and resized
            sub_image = i[y:y+dim, x:x+dim]
            sub_image = cv2.resize(sub_image, (28, 28))
            # a score is computed for the subsection that is higher
            # if it looks more like a hip number. This score is stored
            # in vals
            score = predict_array(sub_image, hip_classifier)
            vals[int(y/step), int(x/step)] = score
            x+=step
        y+=step

    # cv2.imshow("results", vals)
    # results = cv2.resize(vals, (300, 300))
    # cv2.imwrite("heatmap.png", results)
    # np.save("heatmap.npy", results)

    coords = np.unravel_index(np.argmax(vals), vals.shape)
    best_y = coords[0]
    best_x = coords[1]
    # find the subimage that had the best hip number score
    best_image = i[best_y*step:best_y*step+dim, best_x*step:best_x*step+dim]

    #adjust size of the hip number section
    small_hip = cv2.resize(best_image, (28,28))
    small_hip = small_hip/np.float32(255)

    # black image that will be prepared for mnist classification
    prepped_image = np.zeros((28,28))

    # fill prepped image so that the number is black
    # and the background is white
    for y in range(0,28):
        for x in range(0,28):
            if is_num(small_hip, y, x):
                prepped_image[y,x] = small_hip[y,x]
            else:
                prepped_image[y,x] = 0.80


    # view = cv2.resize(prepped_image, (300, 300))


    # cv2.imshow("hip", view)
    # np.save("hip.png", view)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=mnistCNN.cnn_model_fn, model_dir="mnist")

    final_result = classify_hip(prepped_image, mnist_classifier)
    # print("FINAL RESULT: " + str(final_result))

    # cv2.waitKey(0)
    return final_result

# Determines whether the point at y,xx in small_hip
# is part of the number portion of a hip number
#returns boolean
def is_num(small_hip, y, x):
    threshold = 0.4
    if (small_hip[y,x] > threshold):
        return False
    left = x==0 or np.amax(small_hip[y, 0:x]) > threshold
    right = np.amax(small_hip[y, x:28]) > threshold
    top = y==0 or np.amax(small_hip[0:y, x]) > threshold
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



# zach_pic = "../data/finish-line/bmps/marked/20190413_140509_011.bmp"
# finish_1 = "finish_1.png"
# finish_2 = "finish_2.png"
# finish_3 = "finish_3.png"
# finish_4 = "finish_4.png"
# # predict_file("../data/finish-line/bmps/train/eval/20190413_144457_1327_neg1.bmp")
# # locate_hip("../data/finish-line/bmps/marked/20190413_140509_011.bmp")
# locate_hip(finish_3)
# locate_hip(finish_4)
# locate_hip(finish_1)
