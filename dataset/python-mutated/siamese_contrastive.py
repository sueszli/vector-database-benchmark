"""
Title: Image similarity estimation using a Siamese Network with a contrastive loss
Author: Mehdi
Date created: 2021/05/06
Last modified: 2022/09/10
Description: Similarity learning using a siamese network trained with a contrastive loss.
Accelerator: GPU
"""
'\n## Introduction\n\n[Siamese Networks](https://en.wikipedia.org/wiki/Siamese_neural_network)\nare neural networks which share weights between two or more sister networks,\neach producing embedding vectors of its respective inputs.\n\nIn supervised similarity learning, the networks are then trained to maximize the\ncontrast (distance) between embeddings of inputs of different classes, while minimizing the distance between\nembeddings of similar classes, resulting in embedding spaces that reflect\nthe class segmentation of the training inputs.\n'
'\n## Setup\n'
import random
import numpy as np
import keras
from keras import ops
import matplotlib.pyplot as plt
'\n## Hyperparameters\n'
epochs = 10
batch_size = 16
margin = 1
'\n## Load the MNIST dataset\n'
((x_train_val, y_train_val), (x_test, y_test)) = keras.datasets.mnist.load_data()
x_train_val = x_train_val.astype('float32')
x_test = x_test.astype('float32')
'\n## Define training and validation sets\n'
(x_train, x_val) = (x_train_val[:30000], x_train_val[30000:])
(y_train, y_val) = (y_train_val[:30000], y_train_val[30000:])
del x_train_val, y_train_val
'\n## Create pairs of images\n\nWe will train the model to differentiate between digits of different classes. For\nexample, digit `0` needs to be differentiated from the rest of the\ndigits (`1` through `9`), digit `1` - from `0` and `2` through `9`, and so on.\nTo carry this out, we will select N random images from class A (for example,\nfor digit `0`) and pair them with N random images from another class B\n(for example, for digit `1`). Then, we can repeat this process for all classes\nof digits (until digit `9`). Once we have paired digit `0` with other digits,\nwe can repeat this process for the remaining classes for the rest of the digits\n(from `1` until `9`).\n'

def make_pairs(x, y):
    if False:
        i = 10
        return i + 15
    "Creates a tuple containing image pairs with corresponding label.\n\n    Arguments:\n        x: List containing images, each index in this list corresponds to one image.\n        y: List containing labels, each label with datatype of `int`.\n\n    Returns:\n        Tuple containing two numpy arrays as (pairs_of_samples, labels),\n        where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and\n        labels are a binary array of shape (2len(x)).\n    "
    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]
    pairs = []
    labels = []
    for idx1 in range(len(x)):
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[label1])
        x2 = x[idx2]
        pairs += [[x1, x2]]
        labels += [0]
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)
        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]
        pairs += [[x1, x2]]
        labels += [1]
    return (np.array(pairs), np.array(labels).astype('float32'))
(pairs_train, labels_train) = make_pairs(x_train, y_train)
(pairs_val, labels_val) = make_pairs(x_val, y_val)
(pairs_test, labels_test) = make_pairs(x_test, y_test)
'\nWe get:\n\n**pairs_train.shape = (60000, 2, 28, 28)**\n\n- We have 60,000 pairs\n- Each pair contains 2 images\n- Each image has shape `(28, 28)`\n'
'\nSplit the training pairs\n'
x_train_1 = pairs_train[:, 0]
x_train_2 = pairs_train[:, 1]
'\nSplit the validation pairs\n'
x_val_1 = pairs_val[:, 0]
x_val_2 = pairs_val[:, 1]
'\nSplit the test pairs\n'
x_test_1 = pairs_test[:, 0]
x_test_2 = pairs_test[:, 1]
'\n## Visualize pairs and their labels\n'

def visualize(pairs, labels, to_show=6, num_col=3, predictions=None, test=False):
    if False:
        i = 10
        return i + 15
    "Creates a plot of pairs and labels, and prediction if it's test dataset.\n\n    Arguments:\n        pairs: Numpy Array, of pairs to visualize, having shape\n               (Number of pairs, 2, 28, 28).\n        to_show: Int, number of examples to visualize (default is 6)\n                `to_show` must be an integral multiple of `num_col`.\n                 Otherwise it will be trimmed if it is greater than num_col,\n                 and incremented if if it is less then num_col.\n        num_col: Int, number of images in one row - (default is 3)\n                 For test and train respectively, it should not exceed 3 and 7.\n        predictions: Numpy Array of predictions with shape (to_show, 1) -\n                     (default is None)\n                     Must be passed when test=True.\n        test: Boolean telling whether the dataset being visualized is\n              train dataset or test dataset - (default False).\n\n    Returns:\n        None.\n    "
    num_row = to_show // num_col if to_show // num_col != 0 else 1
    to_show = num_row * num_col
    (fig, axes) = plt.subplots(num_row, num_col, figsize=(5, 5))
    for i in range(to_show):
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]
        ax.imshow(ops.concatenate([pairs[i][0], pairs[i][1]], axis=1), cmap='gray')
        ax.set_axis_off()
        if test:
            ax.set_title('True: {} | Pred: {:.5f}'.format(labels[i], predictions[i][0]))
        else:
            ax.set_title('Label: {}'.format(labels[i]))
    if test:
        plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
    else:
        plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    plt.show()
'\nInspect training pairs\n'
visualize(pairs_train[:-1], labels_train[:-1], to_show=4, num_col=4)
'\nInspect validation pairs\n'
visualize(pairs_val[:-1], labels_val[:-1], to_show=4, num_col=4)
'\nInspect test pairs\n'
visualize(pairs_test[:-1], labels_test[:-1], to_show=4, num_col=4)
'\n## Define the model\n\nThere are two input layers, each leading to its own network, which\nproduces embeddings. A `Lambda` layer then merges them using an\n[Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) and the\nmerged output is fed to the final network.\n'

def euclidean_distance(vects):
    if False:
        for i in range(10):
            print('nop')
    'Find the Euclidean distance between two vectors.\n\n    Arguments:\n        vects: List containing two tensors of same length.\n\n    Returns:\n        Tensor containing euclidean distance\n        (as floating point value) between vectors.\n    '
    (x, y) = vects
    sum_square = ops.sum(ops.square(x - y), axis=1, keepdims=True)
    return ops.sqrt(ops.maximum(sum_square, keras.backend.epsilon()))
input = keras.layers.Input((28, 28, 1))
x = keras.layers.BatchNormalization()(input)
x = keras.layers.Conv2D(4, (5, 5), activation='tanh')(x)
x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
x = keras.layers.Conv2D(16, (5, 5), activation='tanh')(x)
x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
x = keras.layers.Flatten()(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dense(10, activation='tanh')(x)
embedding_network = keras.Model(input, x)
input_1 = keras.layers.Input((28, 28, 1))
input_2 = keras.layers.Input((28, 28, 1))
tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)
merge_layer = keras.layers.Lambda(euclidean_distance, output_shape=(1,))([tower_1, tower_2])
normal_layer = keras.layers.BatchNormalization()(merge_layer)
output_layer = keras.layers.Dense(1, activation='sigmoid')(normal_layer)
siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)
'\n## Define the contrastive Loss\n'

def loss(margin=1):
    if False:
        print('Hello World!')
    "Provides 'contrastive_loss' an enclosing scope with variable 'margin'.\n\n    Arguments:\n        margin: Integer, defines the baseline for distance for which pairs\n                should be classified as dissimilar. - (default is 1).\n\n    Returns:\n        'contrastive_loss' function with data ('margin') attached.\n    "

    def contrastive_loss(y_true, y_pred):
        if False:
            print('Hello World!')
        'Calculates the contrastive loss.\n\n        Arguments:\n            y_true: List of labels, each label is of type float32.\n            y_pred: List of predictions of same length as of y_true,\n                    each label is of type float32.\n\n        Returns:\n            A tensor containing contrastive loss as floating point value.\n        '
        square_pred = ops.square(y_pred)
        margin_square = ops.square(ops.maximum(margin - y_pred, 0))
        return ops.mean((1 - y_true) * square_pred + y_true * margin_square)
    return contrastive_loss
'\n## Compile the model with the contrastive loss\n'
siamese.compile(loss=loss(margin=margin), optimizer='RMSprop', metrics=['accuracy'])
siamese.summary()
'\n## Train the model\n'
history = siamese.fit([x_train_1, x_train_2], labels_train, validation_data=([x_val_1, x_val_2], labels_val), batch_size=batch_size, epochs=epochs)
'\n## Visualize results\n'

def plt_metric(history, metric, title, has_valid=True):
    if False:
        for i in range(10):
            print('nop')
    "Plots the given 'metric' from 'history'.\n\n    Arguments:\n        history: history attribute of History object returned from Model.fit.\n        metric: Metric to plot, a string value present as key in 'history'.\n        title: A string to be used as title of plot.\n        has_valid: Boolean, true if valid data was passed to Model.fit else false.\n\n    Returns:\n        None.\n    "
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history['val_' + metric])
        plt.legend(['train', 'validation'], loc='upper left')
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.show()
plt_metric(history=history.history, metric='accuracy', title='Model accuracy')
plt_metric(history=history.history, metric='loss', title='Contrastive Loss')
'\n## Evaluate the model\n'
results = siamese.evaluate([x_test_1, x_test_2], labels_test)
print('test loss, test acc:', results)
'\n## Visualize the predictions\n'
predictions = siamese.predict([x_test_1, x_test_2])
visualize(pairs_test, labels_test, to_show=3, predictions=predictions, test=True)
'\n**Example available on HuggingFace**\n| Trained Model | Demo |\n| :--: | :--: |\n| [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Siamese%20Network-black.svg)](https://huggingface.co/keras-io/siamese-contrastive) | [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Siamese%20Network-black.svg)](https://huggingface.co/spaces/keras-io/siamese-contrastive) |\n'