"""
Title: Next-Frame Video Prediction with Convolutional LSTMs
Author: [Amogh Joshi](https://github.com/amogh7joshi)
Date created: 2021/06/02
Last modified: 2021/06/05
Description: How to build and train a convolutional LSTM model for next-frame video prediction.
Accelerator: GPU
"""
'\n## Introduction\n\nThe\n[Convolutional LSTM](https://papers.nips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf)\narchitectures bring together time series processing and computer vision by\nintroducing a convolutional recurrent cell in a LSTM layer. In this example, we will explore the\nConvolutional LSTM model in an application to next-frame prediction, the process\nof predicting what video frames come next given a series of past frames.\n'
'\n## Setup\n'
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
import io
import imageio
from IPython.display import Image, display
from ipywidgets import widgets, Layout, HBox
'\n## Dataset Construction\n\nFor this example, we will be using the\n[Moving MNIST](http://www.cs.toronto.edu/~nitish/unsupervised_video/)\ndataset.\n\nWe will download the dataset and then construct and\npreprocess training and validation sets.\n\nFor next-frame prediction, our model will be using a previous frame,\nwhich we\'ll call `f_n`, to predict a new frame, called `f_(n + 1)`.\nTo allow the model to create these predictions, we\'ll need to process\nthe data such that we have "shifted" inputs and outputs, where the\ninput data is frame `x_n`, being used to predict frame `y_(n + 1)`.\n'
fpath = keras.utils.get_file('moving_mnist.npy', 'http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy')
dataset = np.load(fpath)
dataset = np.swapaxes(dataset, 0, 1)
dataset = dataset[:1000, ...]
dataset = np.expand_dims(dataset, axis=-1)
indexes = np.arange(dataset.shape[0])
np.random.shuffle(indexes)
train_index = indexes[:int(0.9 * dataset.shape[0])]
val_index = indexes[int(0.9 * dataset.shape[0]):]
train_dataset = dataset[train_index]
val_dataset = dataset[val_index]
train_dataset = train_dataset / 255
val_dataset = val_dataset / 255

def create_shifted_frames(data):
    if False:
        i = 10
        return i + 15
    x = data[:, 0:data.shape[1] - 1, :, :]
    y = data[:, 1:data.shape[1], :, :]
    return (x, y)
(x_train, y_train) = create_shifted_frames(train_dataset)
(x_val, y_val) = create_shifted_frames(val_dataset)
print('Training Dataset Shapes: ' + str(x_train.shape) + ', ' + str(y_train.shape))
print('Validation Dataset Shapes: ' + str(x_val.shape) + ', ' + str(y_val.shape))
"\n## Data Visualization\n\nOur data consists of sequences of frames, each of which\nare used to predict the upcoming frame. Let's take a look\nat some of these sequential frames.\n"
(fig, axes) = plt.subplots(4, 5, figsize=(10, 8))
data_choice = np.random.choice(range(len(train_dataset)), size=1)[0]
for (idx, ax) in enumerate(axes.flat):
    ax.imshow(np.squeeze(train_dataset[data_choice][idx]), cmap='gray')
    ax.set_title(f'Frame {idx + 1}')
    ax.axis('off')
print(f'Displaying frames for example {data_choice}.')
plt.show()
'\n## Model Construction\n\nTo build a Convolutional LSTM model, we will use the\n`ConvLSTM2D` layer, which will accept inputs of shape\n`(batch_size, num_frames, width, height, channels)`, and return\na prediction movie of the same shape.\n'
inp = layers.Input(shape=(None, *x_train.shape[2:]))
x = layers.ConvLSTM2D(filters=64, kernel_size=(5, 5), padding='same', return_sequences=True, activation='relu')(inp)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(filters=64, kernel_size=(1, 1), padding='same', return_sequences=True, activation='relu')(x)
x = layers.Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same')(x)
model = keras.models.Model(inp, x)
model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam())
'\n## Model Training\n\nWith our model and data constructed, we can now train the model.\n'
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5)
epochs = 20
batch_size = 5
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), callbacks=[early_stopping, reduce_lr])
"\n## Frame Prediction Visualizations\n\nWith our model now constructed and trained, we can generate\nsome example frame predictions based on a new video.\n\nWe'll pick a random example from the validation set and\nthen choose the first ten frames from them. From there, we can\nallow the model to predict 10 new frames, which we can compare\nto the ground truth frame predictions.\n"
example = val_dataset[np.random.choice(range(len(val_dataset)), size=1)[0]]
frames = example[:10, ...]
original_frames = example[10:, ...]
for _ in range(10):
    new_prediction = model.predict(np.expand_dims(frames, axis=0))
    new_prediction = np.squeeze(new_prediction, axis=0)
    predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)
    frames = np.concatenate((frames, predicted_frame), axis=0)
(fig, axes) = plt.subplots(2, 10, figsize=(20, 4))
for (idx, ax) in enumerate(axes[0]):
    ax.imshow(np.squeeze(original_frames[idx]), cmap='gray')
    ax.set_title(f'Frame {idx + 11}')
    ax.axis('off')
new_frames = frames[10:, ...]
for (idx, ax) in enumerate(axes[1]):
    ax.imshow(np.squeeze(new_frames[idx]), cmap='gray')
    ax.set_title(f'Frame {idx + 11}')
    ax.axis('off')
plt.show()
"\n## Predicted Videos\n\nFinally, we'll pick a few examples from the validation set\nand construct some GIFs with them to see the model's\npredicted videos.\n\nYou can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/conv-lstm)\nand try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/conv-lstm).\n"
examples = val_dataset[np.random.choice(range(len(val_dataset)), size=5)]
predicted_videos = []
for example in examples:
    frames = example[:10, ...]
    original_frames = example[10:, ...]
    new_predictions = np.zeros(shape=(10, *frames[0].shape))
    for i in range(10):
        frames = example[:10 + i + 1, ...]
        new_prediction = model.predict(np.expand_dims(frames, axis=0))
        new_prediction = np.squeeze(new_prediction, axis=0)
        predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)
        new_predictions[i] = predicted_frame
    for frame_set in [original_frames, new_predictions]:
        current_frames = np.squeeze(frame_set)
        current_frames = current_frames[..., np.newaxis] * np.ones(3)
        current_frames = (current_frames * 255).astype(np.uint8)
        current_frames = list(current_frames)
        with io.BytesIO() as gif:
            imageio.mimsave(gif, current_frames, 'GIF', duration=200)
            predicted_videos.append(gif.getvalue())
print(' Truth\tPrediction')
for i in range(0, len(predicted_videos), 2):
    box = HBox([widgets.Image(value=predicted_videos[i]), widgets.Image(value=predicted_videos[i + 1])])
    display(box)