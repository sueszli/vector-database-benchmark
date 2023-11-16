"""
Title: OCR model for reading Captchas
Author: [A_K_Nain](https://twitter.com/A_K_Nain)
Date created: 2020/06/14
Last modified: 2020/06/26
Description: How to implement an OCR model using CNNs, RNNs and CTC loss.
Accelerator: GPU
"""
'\n## Introduction\n\nThis example demonstrates a simple OCR model built with the Functional API. Apart from\ncombining CNN and RNN, it also illustrates how you can instantiate a new layer\nand use it as an "Endpoint layer" for implementing CTC loss. For a detailed\nguide to layer subclassing, please check out\n[this page](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)\nin the developer guides.\n'
'\n## Setup\n'
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
"\n## Load the data: [Captcha Images](https://www.kaggle.com/fournierp/captcha-version-2-images)\nLet's download the data.\n"
'shell\ncurl -LO https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip\nunzip -qq captcha_images_v2.zip\n'
'\nThe dataset contains 1040 captcha files as `png` images. The label for each sample is a string,\nthe name of the file (minus the file extension).\nWe will map each character in the string to an integer for training the model. Similary,\nwe will need to map the predictions of the model back to strings. For this purpose\nwe will maintain two dictionaries, mapping characters to integers, and integers to characters,\nrespectively.\n'
data_dir = Path('./captcha_images_v2/')
images = sorted(list(map(str, list(data_dir.glob('*.png')))))
labels = [img.split(os.path.sep)[-1].split('.png')[0] for img in images]
characters = set((char for label in labels for char in label))
characters = sorted(list(characters))
print('Number of images found: ', len(images))
print('Number of labels found: ', len(labels))
print('Number of unique characters: ', len(characters))
print('Characters present: ', characters)
batch_size = 16
img_width = 200
img_height = 50
downsample_factor = 4
max_length = max([len(label) for label in labels])
'\n## Preprocessing\n'
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

def split_data(images, labels, train_size=0.9, shuffle=True):
    if False:
        return 10
    size = len(images)
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    train_samples = int(size * train_size)
    (x_train, y_train) = (images[indices[:train_samples]], labels[indices[:train_samples]])
    (x_valid, y_valid) = (images[indices[train_samples:]], labels[indices[train_samples:]])
    return (x_train, x_valid, y_train, y_valid)
(x_train, x_valid, y_train, y_valid) = split_data(np.array(images), np.array(labels))

def encode_single_sample(img_path, label):
    if False:
        while True:
            i = 10
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    label = char_to_num(tf.strings.unicode_split(label, input_encoding='UTF-8'))
    return {'image': img, 'label': label}
'\n## Create `Dataset` objects\n'
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
validation_dataset = validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
'\n## Visualize the data\n'
(_, ax) = plt.subplots(4, 4, figsize=(10, 5))
for batch in train_dataset.take(1):
    images = batch['image']
    labels = batch['label']
    for i in range(16):
        img = (images[i] * 255).numpy().astype('uint8')
        label = tf.strings.reduce_join(num_to_char(labels[i])).numpy().decode('utf-8')
        ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap='gray')
        ax[i // 4, i % 4].set_title(label)
        ax[i // 4, i % 4].axis('off')
plt.show()
'\n## Model\n'

class CTCLayer(layers.Layer):

    def __init__(self, name=None):
        if False:
            while True:
                i = 10
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        if False:
            while True:
                i = 10
        batch_len = tf.cast(tf.shape(y_true)[0], dtype='int64')
        input_length = tf.cast(tf.shape(y_pred)[1], dtype='int64')
        label_length = tf.cast(tf.shape(y_true)[1], dtype='int64')
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype='int64')
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype='int64')
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred

def build_model():
    if False:
        print('Hello World!')
    input_img = layers.Input(shape=(img_width, img_height, 1), name='image', dtype='float32')
    labels = layers.Input(name='label', shape=(None,), dtype='float32')
    x = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name='Conv1')(input_img)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name='Conv2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    new_shape = (img_width // 4, img_height // 4 * 64)
    x = layers.Reshape(target_shape=new_shape, name='reshape')(x)
    x = layers.Dense(64, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
    x = layers.Dense(len(char_to_num.get_vocabulary()) + 1, activation='softmax', name='dense2')(x)
    output = CTCLayer(name='ctc_loss')(labels, x)
    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name='ocr_model_v1')
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt)
    return model
model = build_model()
model.summary()
'\n## Training\n'
epochs = 1
early_stopping_patience = 10
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True)
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=[early_stopping])
'\n## Inference\n\nYou can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/ocr-for-captcha)\nand try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/ocr-for-captcha).\n'
prediction_model = keras.models.Model(model.get_layer(name='image').input, model.get_layer(name='dense2').output)
prediction_model.summary()

def decode_batch_predictions(pred):
    if False:
        for i in range(10):
            print('nop')
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode('utf-8')
        output_text.append(res)
    return output_text
for batch in validation_dataset.take(1):
    batch_images = batch['image']
    batch_labels = batch['label']
    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)
    orig_texts = []
    for label in batch_labels:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode('utf-8')
        orig_texts.append(label)
    (_, ax) = plt.subplots(4, 4, figsize=(15, 5))
    for i in range(len(pred_texts)):
        img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
        img = img.T
        title = f'Prediction: {pred_texts[i]}'
        ax[i // 4, i % 4].imshow(img, cmap='gray')
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis('off')
plt.show()