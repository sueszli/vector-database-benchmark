"""
Title: Image segmentation with a U-Net-like architecture
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2019/03/20
Last modified: 2020/04/20
Description: Image segmentation model trained from scratch on the Oxford Pets dataset.
Accelerator: GPU
"""
'\n## Download the data\n'
'shell\n!wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz\n!wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz\n\ncurl -O https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz\ncurl -O https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz\n\ntar -xf images.tar.gz\ntar -xf annotations.tar.gz\n'
'\n## Prepare paths of input images and target segmentation masks\n'
import os
input_dir = 'images/'
target_dir = 'annotations/trimaps/'
img_size = (160, 160)
num_classes = 3
batch_size = 32
input_img_paths = sorted([os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith('.jpg')])
target_img_paths = sorted([os.path.join(target_dir, fname) for fname in os.listdir(target_dir) if fname.endswith('.png') and (not fname.startswith('.'))])
print('Number of samples:', len(input_img_paths))
for (input_path, target_path) in zip(input_img_paths[:10], target_img_paths[:10]):
    print(input_path, '|', target_path)
'\n## What does one input image and corresponding segmentation mask look like?\n'
from IPython.display import Image, display
from keras.utils import load_img
from PIL import ImageOps
display(Image(filename=input_img_paths[9]))
img = ImageOps.autocontrast(load_img(target_img_paths[9]))
display(img)
'\n## Prepare dataset to load & vectorize batches of data\n'
import keras
import numpy as np
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io

def get_dataset(batch_size, img_size, input_img_paths, target_img_paths, max_dataset_len=None):
    if False:
        i = 10
        return i + 15
    'Returns a TF Dataset.'

    def load_img_masks(input_img_path, target_img_path):
        if False:
            while True:
                i = 10
        input_img = tf_io.read_file(input_img_path)
        input_img = tf_io.decode_png(input_img, channels=3)
        input_img = tf_image.resize(input_img, img_size)
        input_img = tf_image.convert_image_dtype(input_img, 'float32')
        target_img = tf_io.read_file(target_img_path)
        target_img = tf_io.decode_png(target_img, channels=1)
        target_img = tf_image.resize(target_img, img_size, method='nearest')
        target_img = tf_image.convert_image_dtype(target_img, 'uint8')
        target_img -= 1
        return (input_img, target_img)
    if max_dataset_len:
        input_img_paths = input_img_paths[:max_dataset_len]
        target_img_paths = target_img_paths[:max_dataset_len]
    dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
    dataset = dataset.map(load_img_masks, num_parallel_calls=tf_data.AUTOTUNE)
    return dataset.batch(batch_size)
'\n## Prepare U-Net Xception-style model\n'
from keras import layers

def get_model(img_size, num_classes):
    if False:
        print('Hello World!')
    inputs = keras.Input(shape=img_size + (3,))
    x = layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    previous_block_activation = x
    for filters in [64, 128, 256]:
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        residual = layers.Conv2D(filters, 1, strides=2, padding='same')(previous_block_activation)
        x = layers.add([x, residual])
        previous_block_activation = x
    for filters in [256, 128, 64, 32]:
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D(2)(x)
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding='same')(residual)
        x = layers.add([x, residual])
        previous_block_activation = x
    outputs = layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(x)
    model = keras.Model(inputs, outputs)
    return model
model = get_model(img_size, num_classes)
model.summary()
'\n## Set aside a validation split\n'
import random
val_samples = 1000
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]
train_dataset = get_dataset(batch_size, img_size, train_input_img_paths, train_target_img_paths, max_dataset_len=1000)
valid_dataset = get_dataset(batch_size, img_size, val_input_img_paths, val_target_img_paths)
'\n## Train the model\n'
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
callbacks = [keras.callbacks.ModelCheckpoint('oxford_segmentation.keras', save_best_only=True)]
epochs = 15
model.fit(train_dataset, epochs=epochs, validation_data=valid_dataset, callbacks=callbacks)
'\n## Visualize predictions\n'
val_dataset = get_dataset(batch_size, img_size, val_input_img_paths, val_target_img_paths)
val_preds = model.predict(val_dataset)

def display_mask(i):
    if False:
        i = 10
        return i + 15
    "Quick utility to display a model's prediction."
    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = ImageOps.autocontrast(keras.utils.array_to_img(mask))
    display(img)
i = 10
display(Image(filename=val_input_img_paths[i]))
img = ImageOps.autocontrast(load_img(val_target_img_paths[i]))
display(img)
display_mask(i)