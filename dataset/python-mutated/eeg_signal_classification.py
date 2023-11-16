"""
Title: Electroencephalogram Signal Classification for action identification
Author: [Suvaditya Mukherjee](https://github.com/suvadityamuk)
Date created: 2022/11/03
Last modified: 2022/11/05
Description: Training a Convolutional model to classify EEG signals produced by exposure to certain stimuli.
Accelerator: GPU
"""
'\n## Introduction\n\nThe following example explores how we can make a Convolution-based Neural Network to\nperform classification on Electroencephalogram signals captured when subjects were\nexposed to different stimuli.\nWe train a model from scratch since such signal-classification models are fairly scarce\nin pre-trained format.\nThe data we use is sourced from the UC Berkeley-Biosense Lab where the data was collected\nfrom 15 subjects at the same time.\nOur process is as follows:\n\n- Load the [UC Berkeley-Biosense Synchronized Brainwave Dataset](https://www.kaggle.com/datasets/berkeley-biosense/synchronized-brainwave-dataset)\n- Visualize random samples from the data\n- Pre-process, collate and scale the data to finally make a `tf.data.Dataset`\n- Prepare class weights in order to tackle major imbalances\n- Create a Conv1D and Dense-based model to perform classification\n- Define callbacks and hyperparameters\n- Train the model\n- Plot metrics from History and perform evaluation\n\nThis example needs the following external dependencies (Gdown, Scikit-learn, Pandas,\nNumpy, Matplotlib). You can install it via the following commands.\n\nGdown is an external package used to download large files from Google Drive. To know\nmore, you can refer to its [PyPi page here](https://pypi.org/project/gdown)\n'
'\n## Setup and Data Downloads\n\nFirst, lets install our dependencies:\n'
'shell\npip install gdown -q\npip install sklearn -q\npip install pandas -q\npip install numpy -q\npip install matplotlib -q\n'
'\nNext, lets download our dataset.\nThe gdown package makes it easy to download the data from Google Drive:\n'
'shell\ngdown 1V5B7Bt6aJm0UHbR7cRKBEK8jx7lYPVuX\n# gdown will download eeg-data.csv onto the local drive for use. Total size of\n# eeg-data.csv is 105.7 MB\n'
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import keras
from keras import layers
import tensorflow as tf
from sklearn import preprocessing, model_selection
import random
QUALITY_THRESHOLD = 128
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 2
'\n## Read data from `eeg-data.csv`\n\nWe use the Pandas library to read the `eeg-data.csv` file and display the first 5 rows\nusing the `.head()` command\n'
eeg = pd.read_csv('eeg-data.csv')
'\nWe remove unlabeled samples from our dataset as they do not contribute to the model. We\nalso perform a `.drop()` operation on the columns that are not required for training data\npreparation\n'
unlabeled_eeg = eeg[eeg['label'] == 'unlabeled']
eeg = eeg.loc[eeg['label'] != 'unlabeled']
eeg = eeg.loc[eeg['label'] != 'everyone paired']
eeg.drop(['indra_time', 'Unnamed: 0', 'browser_latency', 'reading_time', 'attention_esense', 'meditation_esense', 'updatedAt', 'createdAt'], axis=1, inplace=True)
eeg.reset_index(drop=True, inplace=True)
eeg.head()
'\nIn the data, the samples recorded are given a score from 0 to 128 based on how\nwell-calibrated the sensor was (0 being best, 200 being worst). We filter the values\nbased on an arbitrary cutoff limit of 128.\n'

def convert_string_data_to_values(value_string):
    if False:
        i = 10
        return i + 15
    str_list = json.loads(value_string)
    return str_list
eeg['raw_values'] = eeg['raw_values'].apply(convert_string_data_to_values)
eeg = eeg.loc[eeg['signal_quality'] < QUALITY_THRESHOLD]
eeg.head()
'\n## Visualize one random sample from the data\n'
'\nWe visualize one sample from the data to understand how the stimulus-induced signal looks\nlike\n'

def view_eeg_plot(idx):
    if False:
        while True:
            i = 10
    data = eeg.loc[idx, 'raw_values']
    plt.plot(data)
    plt.title(f'Sample random plot')
    plt.show()
view_eeg_plot(7)
'\n## Pre-process and collate data\n'
'\nThere are a total of 67 different labels present in the data, where there are numbered\nsub-labels. We collate them under a single label as per their numbering and replace them\nin the data itself. Following this process, we perform simple Label encoding to get them\nin an integer format.\n'
print('Before replacing labels')
print(eeg['label'].unique(), '\n')
print(len(eeg['label'].unique()), '\n')
eeg.replace({'label': {'blink1': 'blink', 'blink2': 'blink', 'blink3': 'blink', 'blink4': 'blink', 'blink5': 'blink', 'math1': 'math', 'math2': 'math', 'math3': 'math', 'math4': 'math', 'math5': 'math', 'math6': 'math', 'math7': 'math', 'math8': 'math', 'math9': 'math', 'math10': 'math', 'math11': 'math', 'math12': 'math', 'thinkOfItems-ver1': 'thinkOfItems', 'thinkOfItems-ver2': 'thinkOfItems', 'video-ver1': 'video', 'video-ver2': 'video', 'thinkOfItemsInstruction-ver1': 'thinkOfItemsInstruction', 'thinkOfItemsInstruction-ver2': 'thinkOfItemsInstruction', 'colorRound1-1': 'colorRound1', 'colorRound1-2': 'colorRound1', 'colorRound1-3': 'colorRound1', 'colorRound1-4': 'colorRound1', 'colorRound1-5': 'colorRound1', 'colorRound1-6': 'colorRound1', 'colorRound2-1': 'colorRound2', 'colorRound2-2': 'colorRound2', 'colorRound2-3': 'colorRound2', 'colorRound2-4': 'colorRound2', 'colorRound2-5': 'colorRound2', 'colorRound2-6': 'colorRound2', 'colorRound3-1': 'colorRound3', 'colorRound3-2': 'colorRound3', 'colorRound3-3': 'colorRound3', 'colorRound3-4': 'colorRound3', 'colorRound3-5': 'colorRound3', 'colorRound3-6': 'colorRound3', 'colorRound4-1': 'colorRound4', 'colorRound4-2': 'colorRound4', 'colorRound4-3': 'colorRound4', 'colorRound4-4': 'colorRound4', 'colorRound4-5': 'colorRound4', 'colorRound4-6': 'colorRound4', 'colorRound5-1': 'colorRound5', 'colorRound5-2': 'colorRound5', 'colorRound5-3': 'colorRound5', 'colorRound5-4': 'colorRound5', 'colorRound5-5': 'colorRound5', 'colorRound5-6': 'colorRound5', 'colorInstruction1': 'colorInstruction', 'colorInstruction2': 'colorInstruction', 'readyRound1': 'readyRound', 'readyRound2': 'readyRound', 'readyRound3': 'readyRound', 'readyRound4': 'readyRound', 'readyRound5': 'readyRound', 'colorRound1': 'colorRound', 'colorRound2': 'colorRound', 'colorRound3': 'colorRound', 'colorRound4': 'colorRound', 'colorRound5': 'colorRound'}}, inplace=True)
print('After replacing labels')
print(eeg['label'].unique())
print(len(eeg['label'].unique()))
le = preprocessing.LabelEncoder()
le.fit(eeg['label'])
eeg['label'] = le.transform(eeg['label'])
'\nWe extract the number of unique classes present in the data\n'
num_classes = len(eeg['label'].unique())
print(num_classes)
'\nWe now visualize the number of samples present in each class using a Bar plot.\n'
plt.bar(range(num_classes), eeg['label'].value_counts())
plt.title('Number of samples per class')
plt.show()
'\n## Scale and split data\n'
'\nWe perform a simple Min-Max scaling to bring the value-range between 0 and 1. We do not\nuse Standard Scaling as the data does not follow a Gaussian distribution.\n'
scaler = preprocessing.MinMaxScaler()
series_list = [scaler.fit_transform(np.asarray(i).reshape(-1, 1)) for i in eeg['raw_values']]
labels_list = [i for i in eeg['label']]
'\nWe now create a Train-test split with a 15% holdout set. Following this, we reshape the\ndata to create a sequence of length 512. We also convert the labels from their current\nlabel-encoded form to a one-hot encoding to enable use of several different\n`keras.metrics` functions.\n'
(x_train, x_test, y_train, y_test) = model_selection.train_test_split(series_list, labels_list, test_size=0.15, random_state=42, shuffle=True)
print(f'Length of x_train : {len(x_train)}\nLength of x_test : {len(x_test)}\nLength of y_train : {len(y_train)}\nLength of y_test : {len(y_test)}')
x_train = np.asarray(x_train).astype(np.float32).reshape(-1, 512, 1)
y_train = np.asarray(y_train).astype(np.float32).reshape(-1, 1)
y_train = keras.utils.to_categorical(y_train)
x_test = np.asarray(x_test).astype(np.float32).reshape(-1, 512, 1)
y_test = np.asarray(y_test).astype(np.float32).reshape(-1, 1)
y_test = keras.utils.to_categorical(y_test)
'\n## Prepare `tf.data.Dataset`\n'
'\nWe now create a `tf.data.Dataset` from this data to prepare it for training. We also\nshuffle and batch the data for use later.\n'
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
'\n## Make Class Weights using Naive method\n'
'\nAs we can see from the plot of number of samples per class, the dataset is imbalanced.\nHence, we **calculate weights for each class** to make sure that the model is trained in\na fair manner without preference to any specific class due to greater number of samples.\n\nWe use a naive method to calculate these weights, finding an **inverse proportion** of\neach class and using that as the weight.\n'
vals_dict = {}
for i in eeg['label']:
    if i in vals_dict.keys():
        vals_dict[i] += 1
    else:
        vals_dict[i] = 1
total = sum(vals_dict.values())
weight_dict = {k: 1 - v / total for (k, v) in vals_dict.items()}
print(weight_dict)
'\n## Define simple function to plot all the metrics present in a `keras.callbacks.History`\nobject\n'

def plot_history_metrics(history: keras.callbacks.History):
    if False:
        while True:
            i = 10
    total_plots = len(history.history)
    cols = total_plots // 2
    rows = total_plots // cols
    if total_plots % cols != 0:
        rows += 1
    pos = range(1, total_plots + 1)
    plt.figure(figsize=(15, 10))
    for (i, (key, value)) in enumerate(history.history.items()):
        plt.subplot(rows, cols, pos[i])
        plt.plot(range(len(value)), value)
        plt.title(str(key))
    plt.show()
'\n## Define function to generate Convolutional model\n'

def create_model():
    if False:
        for i in range(10):
            print('nop')
    input_layer = keras.Input(shape=(512, 1))
    x = layers.Conv1D(filters=32, kernel_size=3, strides=2, activation='relu', padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=64, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=128, kernel_size=5, strides=2, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=256, kernel_size=5, strides=2, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=512, kernel_size=7, strides=2, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=1024, kernel_size=7, strides=2, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(2048, activation='relu', kernel_regularizer=keras.regularizers.L2())(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.L2())(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.L2())(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs=input_layer, outputs=output_layer)
'\n## Get Model summary\n'
conv_model = create_model()
conv_model.summary()
'\n## Define callbacks, optimizer, loss and metrics\n'
'\nWe set the number of epochs at 30 after performing extensive experimentation. It was seen\nthat this was the optimal number, after performing Early-Stopping analysis as well.\nWe define a Model Checkpoint callback to make sure that we only get the best model\nweights.\nWe also define a ReduceLROnPlateau as there were several cases found during\nexperimentation where the loss stagnated after a certain point. On the other hand, a\ndirect LRScheduler was found to be too aggressive in its decay.\n'
epochs = 30
callbacks = [keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='loss'), keras.callbacks.ReduceLROnPlateau(monitor='val_top_k_categorical_accuracy', factor=0.2, patience=2, min_lr=1e-06)]
optimizer = keras.optimizers.Adam(amsgrad=True, learning_rate=0.001)
loss = keras.losses.CategoricalCrossentropy()
'\n## Compile model and call `model.fit()`\n'
'\nWe use the `Adam` optimizer since it is commonly considered the best choice for\npreliminary training, and was found to be the best optimizer.\nWe use `CategoricalCrossentropy` as the loss as our labels are in a one-hot-encoded form.\n\nWe define the `TopKCategoricalAccuracy(k=3)`, `AUC`, `Precision` and `Recall` metrics to\nfurther aid in understanding the model better.\n'
conv_model.compile(optimizer=optimizer, loss=loss, metrics=[keras.metrics.TopKCategoricalAccuracy(k=3), keras.metrics.AUC(), keras.metrics.Precision(), keras.metrics.Recall()])
conv_model_history = conv_model.fit(train_dataset, epochs=epochs, callbacks=callbacks, validation_data=test_dataset, class_weight=weight_dict)
'\n## Visualize model metrics during training\n'
'\nWe use the function defined above to see model metrics during training.\n'
plot_history_metrics(conv_model_history)
'\n## Evaluate model on test data\n'
(loss, accuracy, auc, precision, recall) = conv_model.evaluate(test_dataset)
print(f'Loss : {loss}')
print(f'Top 3 Categorical Accuracy : {accuracy}')
print(f'Area under the Curve (ROC) : {auc}')
print(f'Precision : {precision}')
print(f'Recall : {recall}')

def view_evaluated_eeg_plots(model):
    if False:
        i = 10
        return i + 15
    start_index = random.randint(10, len(eeg))
    end_index = start_index + 11
    data = eeg.loc[start_index:end_index, 'raw_values']
    data_array = [scaler.fit_transform(np.asarray(i).reshape(-1, 1)) for i in data]
    data_array = [np.asarray(data_array).astype(np.float32).reshape(-1, 512, 1)]
    original_labels = eeg.loc[start_index:end_index, 'label']
    predicted_labels = np.argmax(model.predict(data_array, verbose=0), axis=1)
    original_labels = [le.inverse_transform(np.array(label).reshape(-1))[0] for label in original_labels]
    predicted_labels = [le.inverse_transform(np.array(label).reshape(-1))[0] for label in predicted_labels]
    total_plots = 12
    cols = total_plots // 3
    rows = total_plots // cols
    if total_plots % cols != 0:
        rows += 1
    pos = range(1, total_plots + 1)
    fig = plt.figure(figsize=(20, 10))
    for (i, (plot_data, og_label, pred_label)) in enumerate(zip(data, original_labels, predicted_labels)):
        plt.subplot(rows, cols, pos[i])
        plt.plot(plot_data)
        plt.title(f'Actual Label : {og_label}\nPredicted Label : {pred_label}')
        fig.subplots_adjust(hspace=0.5)
    plt.show()
view_evaluated_eeg_plots(conv_model)