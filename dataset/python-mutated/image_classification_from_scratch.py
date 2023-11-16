"""
Title: Image classification from scratch
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/04/27
Last modified: 2023/11/09
Description: Training an image classifier from scratch on the Kaggle Cats vs Dogs dataset.
Accelerator: GPU
"""
'\n## Introduction\n\nThis example shows how to do image classification from scratch, starting from JPEG\nimage files on disk, without leveraging pre-trained weights or a pre-made Keras\nApplication model. We demonstrate the workflow on the Kaggle Cats vs Dogs binary\nclassification dataset.\n\nWe use the `image_dataset_from_directory` utility to generate the datasets, and\nwe use Keras image preprocessing layers for image standardization and data augmentation.\n'
'\n## Setup\n'
import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
"\n## Load the data: the Cats vs Dogs dataset\n\n### Raw data download\n\nFirst, let's download the 786M ZIP archive of the raw data:\n"
'shell\ncurl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip\n'
'shell\nunzip -q kagglecatsanddogs_5340.zip\nls\n'
'\nNow we have a `PetImages` folder which contain two subfolders, `Cat` and `Dog`. Each\nsubfolder contains image files for each category.\n'
'shell\nls PetImages\n'
'\n### Filter out corrupted images\n\nWhen working with lots of real-world image data, corrupted images are a common\noccurence. Let\'s filter out badly-encoded images that do not feature the string "JFIF"\nin their header.\n'
num_skipped = 0
for folder_name in ('Cat', 'Dog'):
    folder_path = os.path.join('PetImages', folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, 'rb')
            is_jfif = b'JFIF' in fobj.peek(10)
        finally:
            fobj.close()
        if not is_jfif:
            num_skipped += 1
            os.remove(fpath)
print(f'Deleted {num_skipped} images.')
'\n## Generate a `Dataset`\n'
image_size = (180, 180)
batch_size = 128
(train_ds, val_ds) = keras.utils.image_dataset_from_directory('PetImages', validation_split=0.2, subset='both', seed=1337, image_size=image_size, batch_size=batch_size)
'\n## Visualize the data\n\nHere are the first 9 images in the training dataset.\n'
plt.figure(figsize=(10, 10))
for (images, labels) in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype('uint8'))
        plt.title(int(labels[i]))
        plt.axis('off')
"\n## Using image data augmentation\n\nWhen you don't have a large image dataset, it's a good practice to artificially\nintroduce sample diversity by applying random yet realistic transformations to the\ntraining images, such as random horizontal flipping or small random rotations. This\nhelps expose the model to different aspects of the training data while slowing down\noverfitting.\n"
data_augmentation_layers = [layers.RandomFlip('horizontal'), layers.RandomRotation(0.1)]

def data_augmentation(images):
    if False:
        i = 10
        return i + 15
    for layer in data_augmentation_layers:
        images = layer(images)
    return images
"\nLet's visualize what the augmented samples look like, by applying `data_augmentation`\nrepeatedly to the first few images in the dataset:\n"
plt.figure(figsize=(10, 10))
for (images, _) in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(augmented_images[0]).astype('uint8'))
        plt.axis('off')
'\n## Standardizing the data\n\nOur image are already in a standard size (180x180), as they are being yielded as\ncontiguous `float32` batches by our dataset. However, their RGB channel values are in\nthe `[0, 255]` range. This is not ideal for a neural network;\nin general you should seek to make your input values small. Here, we will\nstandardize values to be in the `[0, 1]` by using a `Rescaling` layer at the start of\nour model.\n'
"\n## Two options to preprocess the data\n\nThere are two ways you could be using the `data_augmentation` preprocessor:\n\n**Option 1: Make it part of the model**, like this:\n\n```python\ninputs = keras.Input(shape=input_shape)\nx = data_augmentation(inputs)\nx = layers.Rescaling(1./255)(x)\n...  # Rest of the model\n```\n\nWith this option, your data augmentation will happen *on device*, synchronously\nwith the rest of the model execution, meaning that it will benefit from GPU\nacceleration.\n\nNote that data augmentation is inactive at test time, so the input samples will only be\naugmented during `fit()`, not when calling `evaluate()` or `predict()`.\n\nIf you're training on GPU, this may be a good option.\n\n**Option 2: apply it to the dataset**, so as to obtain a dataset that yields batches of\naugmented images, like this:\n\n```python\naugmented_train_ds = train_ds.map(\n    lambda x, y: (data_augmentation(x, training=True), y))\n```\n\nWith this option, your data augmentation will happen **on CPU**, asynchronously, and will\nbe buffered before going into the model.\n\nIf you're training on CPU, this is the better option, since it makes data augmentation\nasynchronous and non-blocking.\n\nIn our case, we'll go with the second option. If you're not sure\nwhich one to pick, this second option (asynchronous preprocessing) is always a solid choice.\n"
"\n## Configure the dataset for performance\n\nLet's apply data augmentation to our training dataset,\nand let's make sure to use buffered prefetching so we can yield data from disk without\nhaving I/O becoming blocking:\n"
train_ds = train_ds.map(lambda img, label: (data_augmentation(img), label), num_parallel_calls=tf_data.AUTOTUNE)
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)
"\n## Build a model\n\nWe'll build a small version of the Xception network. We haven't particularly tried to\noptimize the architecture; if you want to do a systematic search for the best model\nconfiguration, consider using\n[KerasTuner](https://github.com/keras-team/keras-tuner).\n\nNote that:\n\n- We start the model with the `data_augmentation` preprocessor, followed by a\n `Rescaling` layer.\n- We include a `Dropout` layer before the final classification layer.\n"

def make_model(input_shape, num_classes):
    if False:
        return 10
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    previous_block_activation = x
    for size in [256, 512, 728]:
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(size, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(size, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        residual = layers.Conv2D(size, 1, strides=2, padding='same')(previous_block_activation)
        x = layers.add([x, residual])
        previous_block_activation = x
    x = layers.SeparableConv2D(1024, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)
model = make_model(input_shape=image_size + (3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)
'\n## Train the model\n'
epochs = 25
callbacks = [keras.callbacks.ModelCheckpoint('save_at_{epoch}.keras')]
model.compile(optimizer=keras.optimizers.Adam(0.0003), loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=[keras.metrics.BinaryAccuracy(name='acc')])
model.fit(train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds)
'\nWe get to >90% validation accuracy after training for 25 epochs on the full dataset\n(in practice, you can train for 50+ epochs before validation performance starts degrading).\n'
'\n## Run inference on new data\n\nNote that data augmentation and dropout are inactive at inference time.\n'
img = keras.utils.load_img('PetImages/Cat/6779.jpg', target_size=image_size)
plt.imshow(img)
img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)
predictions = model.predict(img_array)
score = float(keras.ops.sigmoid(predictions[0]))
print(f'This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.')