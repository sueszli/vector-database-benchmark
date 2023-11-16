"""
Title: Speaker Recognition
Author: [Fadi Badine](https://twitter.com/fadibadine)
Date created: 14/06/2020
Last modified: 19/07/2023
Description: Classify speakers using Fast Fourier Transform (FFT) and a 1D Convnet.
Accelerator: GPU
Converted to Keras 3 by: [Fadi Badine](https://twitter.com/fadibadine)
"""
'\n## Introduction\n\nThis example demonstrates how to create a model to classify speakers from the\nfrequency domain representation of speech recordings, obtained via Fast Fourier\nTransform (FFT).\n\nIt shows the following:\n\n- How to use `tf.data` to load, preprocess and feed audio streams into a model\n- How to create a 1D convolutional network with residual\nconnections for audio classification.\n\nOur process:\n\n- We prepare a dataset of speech samples from different speakers, with the speaker as label.\n- We add background noise to these samples to augment our data.\n- We take the FFT of these samples.\n- We train a 1D convnet to predict the correct speaker given a noisy FFT speech sample.\n\nNote:\n\n- This example should be run with TensorFlow 2.3 or higher, or `tf-nightly`.\n- The noise samples in the dataset need to be resampled to a sampling rate of 16000 Hz\nbefore using the code in this example. In order to do this, you will need to have\ninstalled `ffmpeg`.\n'
'\n## Setup\n'
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import shutil
import numpy as np
import tensorflow as tf
import keras
from pathlib import Path
from IPython.display import display, Audio
'shell\nkaggle datasets download -d kongaevans/speaker-recognition-dataset\nunzip -qq speaker-recognition-dataset.zip\n'
DATASET_ROOT = '16000_pcm_speeches'
AUDIO_SUBFOLDER = 'audio'
NOISE_SUBFOLDER = 'noise'
DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)
DATASET_NOISE_PATH = os.path.join(DATASET_ROOT, NOISE_SUBFOLDER)
VALID_SPLIT = 0.1
SHUFFLE_SEED = 43
SAMPLING_RATE = 16000
SCALE = 0.5
BATCH_SIZE = 128
EPOCHS = 1
"\n## Data preparation\n\nThe dataset is composed of 7 folders, divided into 2 groups:\n\n- Speech samples, with 5 folders for 5 different speakers. Each folder contains\n1500 audio files, each 1 second long and sampled at 16000 Hz.\n- Background noise samples, with 2 folders and a total of 6 files. These files\nare longer than 1 second (and originally not sampled at 16000 Hz, but we will resample them to 16000 Hz).\nWe will use those 6 files to create 354 1-second-long noise samples to be used for training.\n\nLet's sort these 2 categories into 2 folders:\n\n- An `audio` folder which will contain all the per-speaker speech sample folders\n- A `noise` folder which will contain all the noise samples\n"
'\nBefore sorting the audio and noise categories into 2 folders,\nwe have the following directory structure:\n\n```\nmain_directory/\n...speaker_a/\n...speaker_b/\n...speaker_c/\n...speaker_d/\n...speaker_e/\n...other/\n..._background_noise_/\n```\n\nAfter sorting, we end up with the following structure:\n\n```\nmain_directory/\n...audio/\n......speaker_a/\n......speaker_b/\n......speaker_c/\n......speaker_d/\n......speaker_e/\n...noise/\n......other/\n......_background_noise_/\n```\n'
for folder in os.listdir(DATASET_ROOT):
    if os.path.isdir(os.path.join(DATASET_ROOT, folder)):
        if folder in [AUDIO_SUBFOLDER, NOISE_SUBFOLDER]:
            continue
        elif folder in ['other', '_background_noise_']:
            shutil.move(os.path.join(DATASET_ROOT, folder), os.path.join(DATASET_NOISE_PATH, folder))
        else:
            shutil.move(os.path.join(DATASET_ROOT, folder), os.path.join(DATASET_AUDIO_PATH, folder))
'\n## Noise preparation\n\nIn this section:\n\n- We load all noise samples (which should have been resampled to 16000)\n- We split those noise samples to chunks of 16000 samples which\ncorrespond to 1 second duration each\n'
noise_paths = []
for subdir in os.listdir(DATASET_NOISE_PATH):
    subdir_path = Path(DATASET_NOISE_PATH) / subdir
    if os.path.isdir(subdir_path):
        noise_paths += [os.path.join(subdir_path, filepath) for filepath in os.listdir(subdir_path) if filepath.endswith('.wav')]
if not noise_paths:
    raise RuntimeError(f'Could not find any files at {DATASET_NOISE_PATH}')
print('Found {} files belonging to {} directories'.format(len(noise_paths), len(os.listdir(DATASET_NOISE_PATH))))
'\nResample all noise samples to 16000 Hz.\nNote that this requires `ffmpeg`.\n'
command = 'for dir in `ls -1 ' + DATASET_NOISE_PATH + '`; do for file in `ls -1 ' + DATASET_NOISE_PATH + '/$dir/*.wav`; do sample_rate=`ffprobe -hide_banner -loglevel panic -show_streams $file | grep sample_rate | cut -f2 -d=`; if [ $sample_rate -ne 16000 ]; then ffmpeg -hide_banner -loglevel panic -y -i $file -ar 16000 temp.wav; mv temp.wav $file; fi; done; done'
os.system(command)

def load_noise_sample(path):
    if False:
        i = 10
        return i + 15
    (sample, sampling_rate) = tf.audio.decode_wav(tf.io.read_file(path), desired_channels=1)
    if sampling_rate == SAMPLING_RATE:
        slices = int(sample.shape[0] / SAMPLING_RATE)
        sample = tf.split(sample[:slices * SAMPLING_RATE], slices)
        return sample
    else:
        print('Sampling rate for {} is incorrect. Ignoring it'.format(path))
        return None
noises = []
for path in noise_paths:
    sample = load_noise_sample(path)
    if sample:
        noises.extend(sample)
noises = tf.stack(noises)
print('{} noise files were split into {} noise samples where each is {} sec. long'.format(len(noise_paths), noises.shape[0], noises.shape[1] // SAMPLING_RATE))
'\n## Dataset generation\n'

def paths_and_labels_to_dataset(audio_paths, labels):
    if False:
        for i in range(10):
            print('nop')
    'Constructs a dataset of audios and labels.'
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: path_to_audio(x), num_parallel_calls=tf.data.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))

def path_to_audio(path):
    if False:
        for i in range(10):
            print('nop')
    'Reads and decodes an audio file.'
    audio = tf.io.read_file(path)
    (audio, _) = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    return audio

def add_noise(audio, noises=None, scale=0.5):
    if False:
        return 10
    if noises is not None:
        tf_rnd = tf.random.uniform((tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32)
        noise = tf.gather(noises, tf_rnd, axis=0)
        prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
        prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)
        audio = audio + noise * prop * scale
    return audio

def audio_to_fft(audio):
    if False:
        while True:
            i = 10
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64))
    fft = tf.expand_dims(fft, axis=-1)
    return tf.math.abs(fft[:, :audio.shape[1] // 2, :])
class_names = os.listdir(DATASET_AUDIO_PATH)
print('Our class names: {}'.format(class_names))
audio_paths = []
labels = []
for (label, name) in enumerate(class_names):
    print('Processing speaker {}'.format(name))
    dir_path = Path(DATASET_AUDIO_PATH) / name
    speaker_sample_paths = [os.path.join(dir_path, filepath) for filepath in os.listdir(dir_path) if filepath.endswith('.wav')]
    audio_paths += speaker_sample_paths
    labels += [label] * len(speaker_sample_paths)
print('Found {} files belonging to {} classes.'.format(len(audio_paths), len(class_names)))
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(labels)
num_val_samples = int(VALID_SPLIT * len(audio_paths))
print('Using {} files for training.'.format(len(audio_paths) - num_val_samples))
train_audio_paths = audio_paths[:-num_val_samples]
train_labels = labels[:-num_val_samples]
print('Using {} files for validation.'.format(num_val_samples))
valid_audio_paths = audio_paths[-num_val_samples:]
valid_labels = labels[-num_val_samples:]
train_ds = paths_and_labels_to_dataset(train_audio_paths, train_labels)
train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(BATCH_SIZE)
valid_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)
train_ds = train_ds.map(lambda x, y: (add_noise(x, noises, scale=SCALE), y), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
valid_ds = valid_ds.map(lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE)
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)
'\n## Model Definition\n'

def residual_block(x, filters, conv_num=3, activation='relu'):
    if False:
        print('Hello World!')
    s = keras.layers.Conv1D(filters, 1, padding='same')(x)
    for i in range(conv_num - 1):
        x = keras.layers.Conv1D(filters, 3, padding='same')(x)
        x = keras.layers.Activation(activation)(x)
    x = keras.layers.Conv1D(filters, 3, padding='same')(x)
    x = keras.layers.Add()([x, s])
    x = keras.layers.Activation(activation)(x)
    return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)

def build_model(input_shape, num_classes):
    if False:
        return 10
    inputs = keras.layers.Input(shape=input_shape, name='input')
    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)
    x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax', name='output')(x)
    return keras.models.Model(inputs=inputs, outputs=outputs)
model = build_model((SAMPLING_RATE // 2, 1), len(class_names))
model.summary()
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_save_filename = 'model.keras'
earlystopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(model_save_filename, monitor='val_accuracy', save_best_only=True)
'\n## Training\n'
history = model.fit(train_ds, epochs=EPOCHS, validation_data=valid_ds, callbacks=[earlystopping_cb, mdlcheckpoint_cb])
'\n## Evaluation\n'
print(model.evaluate(valid_ds))
'\nWe get ~ 98% validation accuracy.\n'
"\n## Demonstration\n\nLet's take some samples and:\n\n- Predict the speaker\n- Compare the prediction with the real speaker\n- Listen to the audio to see that despite the samples being noisy,\nthe model is still pretty accurate\n"
SAMPLES_TO_DISPLAY = 10
test_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
test_ds = test_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(BATCH_SIZE)
test_ds = test_ds.map(lambda x, y: (add_noise(x, noises, scale=SCALE), y), num_parallel_calls=tf.data.AUTOTUNE)
for (audios, labels) in test_ds.take(1):
    ffts = audio_to_fft(audios)
    y_pred = model.predict(ffts)
    rnd = np.random.randint(0, BATCH_SIZE, SAMPLES_TO_DISPLAY)
    audios = audios.numpy()[rnd, :, :]
    labels = labels.numpy()[rnd]
    y_pred = np.argmax(y_pred, axis=-1)[rnd]
    for index in range(SAMPLES_TO_DISPLAY):
        print('Speaker:\x1b{} {}\x1b[0m\tPredicted:\x1b{} {}\x1b[0m'.format('[92m' if labels[index] == y_pred[index] else '[91m', class_names[labels[index]], '[92m' if labels[index] == y_pred[index] else '[91m', class_names[y_pred[index]]))
        display(Audio(audios[index, :, :].squeeze(), rate=SAMPLING_RATE))