"""
Title: English speaker accent recognition using Transfer Learning
Author: [Fadi Badine](https://twitter.com/fadibadine)
Converted to Keras 3 by: [Fadi Badine](https://twitter.com/fadibadine)
Date created: 2022/04/16
Last modified: 2023/07/19
Description: Training a model to classify UK & Ireland accents using feature extraction from Yamnet.
Accelerator: GPU
"""
'\n## Introduction\n\nThe following example shows how to use feature extraction in order to\ntrain a model to classify the English accent spoken in an audio wave.\n\nInstead of training a model from scratch, transfer learning enables us to\ntake advantage of existing state-of-the-art deep learning models and use them as feature extractors.\n\nOur process:\n\n* Use a TF Hub pre-trained model (Yamnet) and apply it as part of the tf.data pipeline which transforms\nthe audio files into feature vectors.\n* Train a dense model on the feature vectors.\n* Use the trained model for inference on a new audio file.\n\nNote:\n\n* We need to install TensorFlow IO in order to resample audio files to 16 kHz as required by Yamnet model.\n* In the test section, ffmpeg is used to convert the mp3 file to wav.\n\nYou can install TensorFlow IO with the following command:\n'
'shell\npip install -U -q tensorflow_io\n'
'\n## Configuration\n'
SEED = 1337
EPOCHS = 1
BATCH_SIZE = 64
VALIDATION_RATIO = 0.1
MODEL_NAME = 'uk_irish_accent_recognition'
CACHE_DIR = None
URL_PATH = 'https://www.openslr.org/resources/83/'
zip_files = {0: 'irish_english_male.zip', 1: 'midlands_english_female.zip', 2: 'midlands_english_male.zip', 3: 'northern_english_female.zip', 4: 'northern_english_male.zip', 5: 'scottish_english_female.zip', 6: 'scottish_english_male.zip', 7: 'southern_english_female.zip', 8: 'southern_english_male.zip', 9: 'welsh_english_female.zip', 10: 'welsh_english_male.zip'}
gender_agnostic_categories = ['ir', 'mi', 'no', 'sc', 'so', 'we']
class_names = ['Irish', 'Midlands', 'Northern', 'Scottish', 'Southern', 'Welsh', 'Not a speech']
'\n## Imports\n'
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import io
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import keras
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from IPython.display import Audio
keras.utils.set_random_seed(SEED)
DATASET_DESTINATION = os.path.join(CACHE_DIR if CACHE_DIR else '~/.keras/', 'datasets')
'\n## Yamnet Model\n\nYamnet is an audio event classifier trained on the AudioSet dataset to predict audio\nevents from the AudioSet ontology. It is available on TensorFlow Hub.\n\nYamnet accepts a 1-D tensor of audio samples with a sample rate of 16 kHz.\nAs output, the model returns a 3-tuple:\n\n* Scores of shape `(N, 521)` representing the scores of the 521 classes.\n* Embeddings of shape `(N, 1024)`.\n* The log-mel spectrogram of the entire audio frame.\n\nWe will use the embeddings, which are the features extracted from the audio samples, as the input to our dense model.\n\nFor more detailed information about Yamnet, please refer to its [TensorFlow Hub](https://tfhub.dev/google/yamnet/1) page.\n'
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
'\n## Dataset\n\nThe dataset used is the\n[Crowdsourced high-quality UK and Ireland English Dialect speech data set](https://openslr.org/83/)\nwhich consists of a total of 17,877 high-quality audio wav files.\n\nThis dataset includes over 31 hours of recording from 120 volunteers who self-identify as\nnative speakers of Southern England, Midlands, Northern England, Wales, Scotland and Ireland.\n\nFor more info, please refer to the above link or to the following paper:\n[Open-source Multi-speaker Corpora of the English Accents in the British Isles](https://aclanthology.org/2020.lrec-1.804.pdf)\n'
'\n## Download the data\n'
line_index_file = keras.utils.get_file(fname='line_index_file', origin=URL_PATH + 'line_index_all.csv')
for i in zip_files:
    fname = zip_files[i].split('.')[0]
    url = URL_PATH + zip_files[i]
    zip_file = keras.utils.get_file(fname=fname, origin=url, extract=True)
    os.remove(zip_file)
'\n## Load the data in a Dataframe\n\nOf the 3 columns (ID, filename and transcript), we are only interested in the filename column in order to read the audio file.\nWe will ignore the other two.\n'
dataframe = pd.read_csv(line_index_file, names=['id', 'filename', 'transcript'], usecols=['filename'])
dataframe.head()
'\nLet\'s now preprocess the dataset by:\n\n* Adjusting the filename (removing a leading space & adding ".wav" extension to the\nfilename).\n* Creating a label using the first 2 characters of the filename which indicate the\naccent.\n* Shuffling the samples.\n'

def preprocess_dataframe(dataframe):
    if False:
        print('Hello World!')
    dataframe['filename'] = dataframe.apply(lambda row: row['filename'].strip(), axis=1)
    dataframe['label'] = dataframe.apply(lambda row: gender_agnostic_categories.index(row['filename'][:2]), axis=1)
    dataframe['filename'] = dataframe.apply(lambda row: os.path.join(DATASET_DESTINATION, row['filename'] + '.wav'), axis=1)
    dataframe = dataframe.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return dataframe
dataframe = preprocess_dataframe(dataframe)
dataframe.head()
"\n## Prepare training & validation sets\n\nLet's split the samples creating training and validation sets.\n"
split = int(len(dataframe) * (1 - VALIDATION_RATIO))
train_df = dataframe[:split]
valid_df = dataframe[split:]
print(f'We have {train_df.shape[0]} training samples & {valid_df.shape[0]} validation ones')
"\n## Prepare a TensorFlow Dataset\n\nNext, we need to create a `tf.data.Dataset`.\nThis is done by creating a `dataframe_to_dataset` function that does the following:\n\n* Create a dataset using filenames and labels.\n* Get the Yamnet embeddings by calling another function `filepath_to_embeddings`.\n* Apply caching, reshuffling and setting batch size.\n\nThe `filepath_to_embeddings` does the following:\n\n* Load audio file.\n* Resample audio to 16 kHz.\n* Generate scores and embeddings from Yamnet model.\n* Since Yamnet generates multiple samples for each audio file,\nthis function also duplicates the label for all the generated samples\nthat have `score=0` (speech) whereas sets the label for the others as\n'other' indicating that this audio segment is not a speech and we won't label it as one of the accents.\n\nThe below `load_16k_audio_file` is copied from the following tutorial\n[Transfer learning with YAMNet for environmental sound classification](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio)\n"

@tf.function
def load_16k_audio_wav(filename):
    if False:
        while True:
            i = 10
    file_content = tf.io.read_file(filename)
    (audio_wav, sample_rate) = tf.audio.decode_wav(file_content, desired_channels=1)
    audio_wav = tf.squeeze(audio_wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    audio_wav = tfio.audio.resample(audio_wav, rate_in=sample_rate, rate_out=16000)
    return audio_wav

def filepath_to_embeddings(filename, label):
    if False:
        print('Hello World!')
    audio_wav = load_16k_audio_wav(filename)
    (scores, embeddings, _) = yamnet_model(audio_wav)
    embeddings_num = tf.shape(embeddings)[0]
    labels = tf.repeat(label, embeddings_num)
    labels = tf.where(tf.argmax(scores, axis=1) == 0, label, len(class_names) - 1)
    return (embeddings, tf.one_hot(labels, len(class_names)))

def dataframe_to_dataset(dataframe, batch_size=64):
    if False:
        i = 10
        return i + 15
    dataset = tf.data.Dataset.from_tensor_slices((dataframe['filename'], dataframe['label']))
    dataset = dataset.map(lambda x, y: filepath_to_embeddings(x, y), num_parallel_calls=tf.data.experimental.AUTOTUNE).unbatch()
    return dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
train_ds = dataframe_to_dataset(train_df)
valid_ds = dataframe_to_dataset(valid_df)
"\n## Build the model\n\nThe model that we use consists of:\n\n* An input layer which is the embedding output of the Yamnet classifier.\n* 4 dense hidden layers and 4 dropout layers.\n* An output dense layer.\n\nThe model's hyperparameters were selected using\n[KerasTuner](https://keras.io/keras_tuner/).\n"
keras.backend.clear_session()

def build_and_compile_model():
    if False:
        i = 10
        return i + 15
    inputs = keras.layers.Input(shape=(1024,), name='embedding')
    x = keras.layers.Dense(256, activation='relu', name='dense_1')(inputs)
    x = keras.layers.Dropout(0.15, name='dropout_1')(x)
    x = keras.layers.Dense(384, activation='relu', name='dense_2')(x)
    x = keras.layers.Dropout(0.2, name='dropout_2')(x)
    x = keras.layers.Dense(192, activation='relu', name='dense_3')(x)
    x = keras.layers.Dropout(0.25, name='dropout_3')(x)
    x = keras.layers.Dense(384, activation='relu', name='dense_4')(x)
    x = keras.layers.Dropout(0.2, name='dropout_4')(x)
    outputs = keras.layers.Dense(len(class_names), activation='softmax', name='ouput')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='accent_recognition')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1.9644e-05), loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy', keras.metrics.AUC(name='auc')])
    return model
model = build_and_compile_model()
model.summary()
'\n## Class weights calculation\n\nSince the dataset is quite unbalanced, we wil use `class_weight` argument during training.\n\nGetting the class weights is a little tricky because even though we know the number of\naudio files for each class, it does not represent the number of samples for that class\nsince Yamnet transforms each audio file into multiple audio samples of 0.96 seconds each.\nSo every audio file will be split into a number of samples that is proportional to its length.\n\nTherefore, to get those weights, we have to calculate the number of samples for each class\nafter preprocessing through Yamnet.\n'
class_counts = tf.zeros(shape=(len(class_names),), dtype=tf.int32)
for (x, y) in iter(train_ds):
    class_counts = class_counts + tf.math.bincount(tf.cast(tf.math.argmax(y, axis=1), tf.int32), minlength=len(class_names))
class_weight = {i: tf.math.reduce_sum(class_counts).numpy() / class_counts[i].numpy() for i in range(len(class_counts))}
print(class_weight)
'\n## Callbacks\n\nWe use Keras callbacks in order to:\n\n* Stop whenever the validation AUC stops improving.\n* Save the best model.\n* Call TensorBoard in order to later view the training and validation logs.\n'
early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True)
model_checkpoint_cb = keras.callbacks.ModelCheckpoint(MODEL_NAME + '.keras', monitor='val_auc', save_best_only=True)
tensorboard_cb = keras.callbacks.TensorBoard(os.path.join(os.curdir, 'logs', model.name))
callbacks = [early_stopping_cb, model_checkpoint_cb, tensorboard_cb]
'\n## Training\n'
history = model.fit(train_ds, epochs=EPOCHS, validation_data=valid_ds, class_weight=class_weight, callbacks=callbacks, verbose=2)
"\n## Results\n\nLet's plot the training and validation AUC and accuracy.\n"
(fig, axs) = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
axs[0].plot(range(EPOCHS), history.history['accuracy'], label='Training')
axs[0].plot(range(EPOCHS), history.history['val_accuracy'], label='Validation')
axs[0].set_xlabel('Epochs')
axs[0].set_title('Training & Validation Accuracy')
axs[0].legend()
axs[0].grid(True)
axs[1].plot(range(EPOCHS), history.history['auc'], label='Training')
axs[1].plot(range(EPOCHS), history.history['val_auc'], label='Validation')
axs[1].set_xlabel('Epochs')
axs[1].set_title('Training & Validation AUC')
axs[1].legend()
axs[1].grid(True)
plt.show()
'\n## Evaluation\n'
(train_loss, train_acc, train_auc) = model.evaluate(train_ds)
(valid_loss, valid_acc, valid_auc) = model.evaluate(valid_ds)
"\nLet's try to compare our model's performance to Yamnet's using one of Yamnet metrics (d-prime)\nYamnet achieved a d-prime value of 2.318.\nLet's check our model's performance.\n"

def d_prime(auc):
    if False:
        print('Hello World!')
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime
print('train d-prime: {0:.3f}, validation d-prime: {1:.3f}'.format(d_prime(train_auc), d_prime(valid_auc)))
'\nWe can see that the model achieves the following results:\n\nResults    | Training  | Validation\n-----------|-----------|------------\nAccuracy   | 54%       | 51%\nAUC        | 0.91      | 0.89\nd-prime    | 1.882     | 1.740\n\n'
"\n## Confusion Matrix\n\nLet's now plot the confusion matrix for the validation dataset.\n\nThe confusion matrix lets us see, for every class, not only how many samples were correctly classified,\nbut also which other classes were the samples confused with.\n\nIt allows us to calculate the precision and recall for every class.\n"
x_valid = None
y_valid = None
for (x, y) in iter(valid_ds):
    if x_valid is None:
        x_valid = x.numpy()
        y_valid = y.numpy()
    else:
        x_valid = np.concatenate((x_valid, x.numpy()), axis=0)
        y_valid = np.concatenate((y_valid, y.numpy()), axis=0)
y_pred = model.predict(x_valid)
confusion_mtx = tf.math.confusion_matrix(np.argmax(y_valid, axis=1), np.argmax(y_pred, axis=1))
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, xticklabels=class_names, yticklabels=class_names, annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.title('Validation Confusion Matrix')
plt.show()
'\n## Precision & recall\n\nFor every class:\n\n* Recall is the ratio of correctly classified samples i.e. it shows how many samples\nof this specific class, the model is able to detect.\nIt is the ratio of diagonal elements to the sum of all elements in the row.\n* Precision shows the accuracy of the classifier. It is the ratio of correctly predicted\nsamples among the ones classified as belonging to this class.\nIt is the ratio of diagonal elements to the sum of all elements in the column.\n'
for (i, label) in enumerate(class_names):
    precision = confusion_mtx[i, i] / np.sum(confusion_mtx[:, i])
    recall = confusion_mtx[i, i] / np.sum(confusion_mtx[i, :])
    print('{0:15} Precision:{1:.2f}%; Recall:{2:.2f}%'.format(label, precision * 100, recall * 100))
"\n## Run inference on test data\n\nLet's now run a test on a single audio file.\nLet's check this example from [The Scottish Voice](https://www.thescottishvoice.org.uk/home/)\n\nWe will:\n\n* Download the mp3 file.\n* Convert it to a 16k wav file.\n* Run the model on the wav file.\n* Plot the results.\n"
filename = 'audio-sample-Stuart'
url = 'https://www.thescottishvoice.org.uk/files/cm/files/'
if os.path.exists(filename + '.wav') == False:
    print(f'Downloading {filename}.mp3 from {url}')
    command = f'wget {url}{filename}.mp3'
    os.system(command)
    print(f'Converting mp3 to wav and resampling to 16 kHZ')
    command = f'ffmpeg -hide_banner -loglevel panic -y -i {filename}.mp3 -acodec pcm_s16le -ac 1 -ar 16000 {filename}.wav'
    os.system(command)
filename = filename + '.wav'
'\nThe below function `yamnet_class_names_from_csv` was copied and very slightly changed\nfrom this [Yamnet Notebook](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/yamnet.ipynb).\n'

def yamnet_class_names_from_csv(yamnet_class_map_csv_text):
    if False:
        while True:
            i = 10
    'Returns list of class names corresponding to score vector.'
    yamnet_class_map_csv = io.StringIO(yamnet_class_map_csv_text)
    yamnet_class_names = [name for (class_index, mid, name) in csv.reader(yamnet_class_map_csv)]
    yamnet_class_names = yamnet_class_names[1:]
    return yamnet_class_names
yamnet_class_map_path = yamnet_model.class_map_path().numpy()
yamnet_class_names = yamnet_class_names_from_csv(tf.io.read_file(yamnet_class_map_path).numpy().decode('utf-8'))

def calculate_number_of_non_speech(scores):
    if False:
        for i in range(10):
            print('nop')
    number_of_non_speech = tf.math.reduce_sum(tf.where(tf.math.argmax(scores, axis=1, output_type=tf.int32) != 0, 1, 0))
    return number_of_non_speech

def filename_to_predictions(filename):
    if False:
        print('Hello World!')
    audio_wav = load_16k_audio_wav(filename)
    (scores, embeddings, mel_spectrogram) = yamnet_model(audio_wav)
    print('Out of {} samples, {} are not speech'.format(scores.shape[0], calculate_number_of_non_speech(scores)))
    predictions = model.predict(embeddings)
    return (audio_wav, predictions, mel_spectrogram)
"\nLet's run the model on the audio file:\n"
(audio_wav, predictions, mel_spectrogram) = filename_to_predictions(filename)
infered_class = class_names[predictions.mean(axis=0).argmax()]
print(f'The main accent is: {infered_class} English')
'\nListen to the audio\n'
Audio(audio_wav, rate=16000)
'\nThe below function was copied from this [Yamnet notebook](tinyurl.com/4a8xn7at) and adjusted to our need.\n\nThis function plots the following:\n\n* Audio waveform\n* Mel spectrogram\n* Predictions for every time step\n'
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(audio_wav)
plt.xlim([0, len(audio_wav)])
plt.subplot(3, 1, 2)
plt.imshow(mel_spectrogram.numpy().T, aspect='auto', interpolation='nearest', origin='lower')
mean_predictions = np.mean(predictions, axis=0)
top_class_indices = np.argsort(mean_predictions)[::-1]
plt.subplot(3, 1, 3)
plt.imshow(predictions[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')
patch_padding = 0.025 / 2 / 0.01
plt.xlim([-patch_padding - 0.5, predictions.shape[0] + patch_padding - 0.5])
yticks = range(0, len(class_names), 1)
plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
_ = plt.ylim(-0.5 + np.array([len(class_names), 0]))