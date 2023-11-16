"""
Title: Timeseries classification from scratch
Author: [hfawaz](https://github.com/hfawaz/)
Date created: 2020/07/21
Last modified: 2021/07/16
Description: Training a timeseries classifier from scratch on the FordA dataset from the UCR/UEA archive.
Accelerator: GPU
"""
'\n## Introduction\n\nThis example shows how to do timeseries classification from scratch, starting from raw\nCSV timeseries files on disk. We demonstrate the workflow on the FordA dataset from the\n[UCR/UEA archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/).\n\n'
'\n## Setup\n\n'
import keras
import numpy as np
import matplotlib.pyplot as plt
'\n## Load the data: the FordA dataset\n\n### Dataset description\n\nThe dataset we are using here is called FordA.\nThe data comes from the UCR archive.\nThe dataset contains 3601 training instances and another 1320 testing instances.\nEach timeseries corresponds to a measurement of engine noise captured by a motor sensor.\nFor this task, the goal is to automatically detect the presence of a specific issue with\nthe engine. The problem is a balanced binary classification task. The full description of\nthis dataset can be found [here](http://www.j-wichard.de/publications/FordPaper.pdf).\n\n### Read the TSV data\n\nWe will use the `FordA_TRAIN` file for training and the\n`FordA_TEST` file for testing. The simplicity of this dataset\nallows us to demonstrate effectively how to use ConvNets for timeseries classification.\nIn this file, the first column corresponds to the label.\n'

def readucr(filename):
    if False:
        print('Hello World!')
    data = np.loadtxt(filename, delimiter='\t')
    y = data[:, 0]
    x = data[:, 1:]
    return (x, y.astype(int))
root_url = 'https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/'
(x_train, y_train) = readucr(root_url + 'FordA_TRAIN.tsv')
(x_test, y_test) = readucr(root_url + 'FordA_TEST.tsv')
'\n## Visualize the data\n\nHere we visualize one timeseries example for each class in the dataset.\n\n'
classes = np.unique(np.concatenate((y_train, y_test), axis=0))
plt.figure()
for c in classes:
    c_x_train = x_train[y_train == c]
    plt.plot(c_x_train[0], label='class ' + str(c))
plt.legend(loc='best')
plt.show()
plt.close()
'\n## Standardize the data\n\nOur timeseries are already in a single length (500). However, their values are\nusually in various ranges. This is not ideal for a neural network;\nin general we should seek to make the input values normalized.\nFor this specific dataset, the data is already z-normalized: each timeseries sample\nhas a mean equal to zero and a standard deviation equal to one. This type of\nnormalization is very common for timeseries classification problems, see\n[Bagnall et al. (2016)](https://link.springer.com/article/10.1007/s10618-016-0483-9).\n\nNote that the timeseries data used here are univariate, meaning we only have one channel\nper timeseries example.\nWe will therefore transform the timeseries into a multivariate one with one channel\nusing a simple reshaping via numpy.\nThis will allow us to construct a model that is easily applicable to multivariate time\nseries.\n'
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
'\nFinally, in order to use `sparse_categorical_crossentropy`, we will have to count\nthe number of classes beforehand.\n'
num_classes = len(np.unique(y_train))
'\nNow we shuffle the training set because we will be using the `validation_split` option\nlater when training.\n'
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]
'\nStandardize the labels to positive integers.\nThe expected labels will then be 0 and 1.\n'
y_train[y_train == -1] = 0
y_test[y_test == -1] = 0
'\n## Build a model\n\nWe build a Fully Convolutional Neural Network originally proposed in\n[this paper](https://arxiv.org/abs/1611.06455).\nThe implementation is based on the TF 2 version provided\n[here](https://github.com/hfawaz/dl-4-tsc/).\nThe following hyperparameters (kernel_size, filters, the usage of BatchNorm) were found\nvia random search using [KerasTuner](https://github.com/keras-team/keras-tuner).\n\n'

def make_model(input_shape):
    if False:
        return 10
    input_layer = keras.layers.Input(input_shape)
    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)
    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)
    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)
    gap = keras.layers.GlobalAveragePooling1D()(conv3)
    output_layer = keras.layers.Dense(num_classes, activation='softmax')(gap)
    return keras.models.Model(inputs=input_layer, outputs=output_layer)
model = make_model(input_shape=x_train.shape[1:])
keras.utils.plot_model(model, show_shapes=True)
'\n## Train the model\n\n'
epochs = 500
batch_size = 32
callbacks = [keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss'), keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=0.0001), keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1)]
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, validation_split=0.2, verbose=1)
'\n## Evaluate model on test data\n'
model = keras.models.load_model('best_model.keras')
(test_loss, test_acc) = model.evaluate(x_test, y_test)
print('Test accuracy', test_acc)
print('Test loss', test_loss)
"\n## Plot the model's training and validation loss\n"
metric = 'sparse_categorical_accuracy'
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history['val_' + metric])
plt.title('model ' + metric)
plt.ylabel(metric, fontsize='large')
plt.xlabel('epoch', fontsize='large')
plt.legend(['train', 'val'], loc='best')
plt.show()
plt.close()
'\nWe can see how the training accuracy reaches almost 0.95 after 100 epochs.\nHowever, by observing the validation accuracy we can see how the network still needs\ntraining until it reaches almost 0.97 for both the validation and the training accuracy\nafter 200 epochs. Beyond the 200th epoch, if we continue on training, the validation\naccuracy will start decreasing while the training accuracy will continue on increasing:\nthe model starts overfitting.\n'