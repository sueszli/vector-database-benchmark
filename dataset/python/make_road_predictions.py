'''
This script generates a binary mask of where the CNN predicts the road to be.
'''
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.models import load_model
from keras.preprocessing import image as image_utils
import matplotlib.pyplot as plt
import os
import numpy as np

# Set up paths
IMAGES_DIR = '/home/ubuntu/csc420_data/data_road/testing/image_2'
MODEL_NAME = 'autoencodernov25.h5p'
OUTPUT_DIR = 'outputs'

i_shape = (248, 504)
x_shape_tensor = (248, 504, 3)
y_shape_tensor = (248, 504, 1)


def load_images(dir, grayscale=False):
    imgs = []
    for i in image_utils.list_pictures(dir):
        imgs.append(image_utils.img_to_array(
            image_utils.load_img(i, grayscale=grayscale, target_size=i_shape)))
    return image_utils.list_pictures(dir),np.array(imgs)


def load_data():
    img_name, x_data = load_images(IMAGES_DIR)
    return img_name, x_data

# Load the model
model = load_model('autoencodernov25.h5p')
img_name, x_test = load_data()
x_test = x_test.astype('float32') / 255.
# Predict the roads
decoded_imgs = model.predict(x_test, batch_size=8)

# Write the predictions.
for i in range(len(x_test)):
    plt.imsave(OUTPUT_DIR+os.sep+os.path.basename(img_name[i]),decoded_imgs[i].reshape(y_shape_tensor[0], y_shape_tensor[1]),cmap='gray')