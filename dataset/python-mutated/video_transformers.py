"""
Title: Video Classification with Transformers
Author: [Sayak Paul](https://twitter.com/RisingSayak)
Converted to Keras 3 by: [Soumik Rakshit](http://github.com/soumik12345)
Date created: 2021/06/08
Last modified: 2023/22/07
Description: Training a video classifier with hybrid transformers.
Accelerator: GPU
"""
'\nThis example is a follow-up to the\n[Video Classification with a CNN-RNN Architecture](https://keras.io/examples/vision/video_classification/)\nexample. This time, we will be using a Transformer-based model\n([Vaswani et al.](https://arxiv.org/abs/1706.03762)) to classify videos. You can follow\n[this book chapter](https://livebook.manning.com/book/deep-learning-with-python-second-edition/chapter-11)\nin case you need an introduction to Transformers (with code). After reading this\nexample, you will know how to develop hybrid Transformer-based models for video\nclassification that operate on CNN feature maps.\n'
'shell\npip install -q git+https://github.com/keras-team/keras\npip install -q git+https://github.com/tensorflow/docs\n'
'\n## Data collection\n\nAs done in the [predecessor](https://keras.io/examples/vision/video_classification/) to\nthis example, we will be using a subsampled version of the\n[UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php),\na well-known benchmark dataset. In case you want to operate on a larger subsample or\neven the entire dataset, please refer to\n[this notebook](https://colab.research.google.com/github/sayakpaul/Action-Recognition-in-TensorFlow/blob/main/Data_Preparation_UCF101.ipynb).\n'
'shell\nwget -q https://github.com/sayakpaul/Action-Recognition-in-TensorFlow/releases/download/v1.0.0/ucf101_top5.tar.gz\ntar -xf ucf101_top5.tar.gz\n'
'\n## Setup\n'
import os
os.environ['KERAS_BACKEND'] = 'jax'
import keras
from keras import layers
from keras.applications.densenet import DenseNet121
from tensorflow_docs.vis import embed
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio
import cv2
'\n## Define hyperparameters\n'
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 1024
IMG_SIZE = 128
EPOCHS = 5
"\n## Data preparation\n\nWe will mostly be following the same data preparation steps in this example, except for\nthe following changes:\n\n* We reduce the image size to 128x128 instead of 224x224 to speed up computation.\n* Instead of using a pre-trained [InceptionV3](https://arxiv.org/abs/1512.00567) network,\nwe use a pre-trained\n[DenseNet121](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf)\nfor feature extraction.\n* We directly pad shorter videos to length `MAX_SEQ_LENGTH`.\n\nFirst, let's load up the\n[DataFrames](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html).\n"
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
print(f'Total videos for training: {len(train_df)}')
print(f'Total videos for testing: {len(test_df)}')
center_crop_layer = layers.CenterCrop(IMG_SIZE, IMG_SIZE)

def crop_center(frame):
    if False:
        print('Hello World!')
    cropped = center_crop_layer(frame[None, ...])
    cropped = keras.ops.convert_to_numpy(cropped)
    cropped = keras.ops.squeeze(cropped)
    return cropped

def load_video(path, max_frames=0, offload_to_cpu=False):
    if False:
        return 10
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            (ret, frame) = cap.read()
            if not ret:
                break
            frame = frame[:, :, [2, 1, 0]]
            frame = crop_center(frame)
            if offload_to_cpu and keras.backend.backend() == 'torch':
                frame = frame.to('cpu')
            frames.append(frame)
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    if offload_to_cpu and keras.backend.backend() == 'torch':
        return np.array([frame.to('cpu').numpy() for frame in frames])
    return np.array(frames)

def build_feature_extractor():
    if False:
        while True:
            i = 10
    feature_extractor = DenseNet121(weights='imagenet', include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    preprocess_input = keras.applications.densenet.preprocess_input
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name='feature_extractor')
feature_extractor = build_feature_extractor()
label_processor = keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.unique(train_df['tag']), mask_token=None)
print(label_processor.get_vocabulary())

def prepare_all_videos(df, root_dir):
    if False:
        while True:
            i = 10
    num_samples = len(df)
    video_paths = df['video_name'].values.tolist()
    labels = df['tag'].values
    labels = label_processor(labels[..., None]).numpy()
    frame_features = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype='float32')
    for (idx, path) in enumerate(video_paths):
        frames = load_video(os.path.join(root_dir, path))
        if len(frames) < MAX_SEQ_LENGTH:
            diff = MAX_SEQ_LENGTH - len(frames)
            padding = np.zeros((diff, IMG_SIZE, IMG_SIZE, 3))
            frames = np.concatenate(frames, padding)
        frames = frames[None, ...]
        temp_frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype='float32')
        for (i, batch) in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                if np.mean(batch[j, :]) > 0.0:
                    temp_frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
                else:
                    temp_frame_features[i, j, :] = 0.0
        frame_features[idx,] = temp_frame_features.squeeze()
    return (frame_features, labels)
'\nCalling `prepare_all_videos()` on `train_df` and `test_df` takes ~20 minutes to\ncomplete. For this reason, to save time, here we download already preprocessed NumPy arrays:\n'
'shell\n!wget -q https://git.io/JZmf4 -O top5_data_prepared.tar.gz\n!tar -xf top5_data_prepared.tar.gz\n'
(train_data, train_labels) = (np.load('train_data.npy'), np.load('train_labels.npy'))
(test_data, test_labels) = (np.load('test_data.npy'), np.load('test_labels.npy'))
print(f'Frame features in train set: {train_data.shape}')
'\n## Building the Transformer-based model\n\nWe will be building on top of the code shared in\n[this book chapter](https://livebook.manning.com/book/deep-learning-with-python-second-edition/chapter-11) of\n[Deep Learning with Python (Second ed.)](https://www.manning.com/books/deep-learning-with-python)\nby François Chollet.\n\nFirst, self-attention layers that form the basic blocks of a Transformer are\norder-agnostic. Since videos are ordered sequences of frames, we need our\nTransformer model to take into account order information.\nWe do this via **positional encoding**.\nWe simply embed the positions of the frames present inside videos with an\n[`Embedding` layer](https://keras.io/api/layers/core_layers/embedding). We then\nadd these positional embeddings to the precomputed CNN feature maps.\n'

class PositionalEmbedding(layers.Layer):

    def __init__(self, sequence_length, output_dim, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        inputs = keras.backend.cast(inputs, self.compute_dtype)
        length = keras.backend.shape(inputs)[1]
        positions = keras.ops.numpy.arange(start=0, stop=length, step=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions
'\nNow, we can create a subclassed layer for the Transformer.\n'

class TransformerEncoder(layers.Layer):

    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.3)
        self.dense_proj = keras.Sequential([layers.Dense(dense_dim, activation=keras.activations.gelu), layers.Dense(embed_dim)])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if False:
            while True:
                i = 10
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
'\n## Utility functions for training\n'

def get_compiled_model(shape):
    if False:
        for i in range(10):
            print('nop')
    sequence_length = MAX_SEQ_LENGTH
    embed_dim = NUM_FEATURES
    dense_dim = 4
    num_heads = 1
    classes = len(label_processor.get_vocabulary())
    inputs = keras.Input(shape=shape)
    x = PositionalEmbedding(sequence_length, embed_dim, name='frame_position_embedding')(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name='transformer_layer')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def run_experiment():
    if False:
        while True:
            i = 10
    filepath = '/tmp/video_classifier.weights.h5'
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, save_weights_only=True, save_best_only=True, verbose=1)
    model = get_compiled_model(train_data.shape[1:])
    history = model.fit(train_data, train_labels, validation_split=0.15, epochs=EPOCHS, callbacks=[checkpoint])
    model.load_weights(filepath)
    (_, accuracy) = model.evaluate(test_data, test_labels)
    print(f'Test accuracy: {round(accuracy * 100, 2)}%')
    return model
'\n## Model training and inference\n'
trained_model = run_experiment()
'\n**Note**: This model has ~4.23 Million parameters, which is way more than the sequence\nmodel (99918 parameters) we used in the prequel of this example.  This kind of\nTransformer model works best with a larger dataset and a longer pre-training schedule.\n'

def prepare_single_video(frames):
    if False:
        return 10
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype='float32')
    if len(frames) < MAX_SEQ_LENGTH:
        diff = MAX_SEQ_LENGTH - len(frames)
        padding = np.zeros((diff, IMG_SIZE, IMG_SIZE, 3))
        frames = np.concatenate(frames, padding)
    frames = frames[None, ...]
    for (i, batch) in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            if np.mean(batch[j, :]) > 0.0:
                frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
            else:
                frame_features[i, j, :] = 0.0
    return frame_features

def predict_action(path):
    if False:
        i = 10
        return i + 15
    class_vocab = label_processor.get_vocabulary()
    frames = load_video(os.path.join('test', path), offload_to_cpu=True)
    frame_features = prepare_single_video(frames)
    probabilities = trained_model.predict(frame_features)[0]
    (plot_x_axis, plot_y_axis) = ([], [])
    for i in np.argsort(probabilities)[::-1]:
        plot_x_axis.append(class_vocab[i])
        plot_y_axis.append(probabilities[i])
        print(f'  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%')
    plt.bar(plot_x_axis, plot_y_axis, label=plot_x_axis)
    plt.xlabel('class_label')
    plt.xlabel('Probability')
    plt.show()
    return frames

def to_gif(images):
    if False:
        print('Hello World!')
    converted_images = images.astype(np.uint8)
    imageio.mimsave('animation.gif', converted_images, fps=10)
    return embed.embed_file('animation.gif')
test_video = np.random.choice(test_df['video_name'].values.tolist())
print(f'Test video path: {test_video}')
test_frames = predict_action(test_video)
to_gif(test_frames[:MAX_SEQ_LENGTH])
'\nThe performance of our model is far from optimal, because it was trained on a\nsmall dataset.\n'