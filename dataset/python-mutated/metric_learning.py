"""
Title: Metric learning for image similarity search
Author: [Mat Kelcey](https://twitter.com/mat_kelcey)
Date created: 2020/06/05
Last modified: 2020/06/09
Description: Example of using similarity metric learning on CIFAR-10 images.
Accelerator: GPU
"""
'\n## Overview\n\nMetric learning aims to train models that can embed inputs into a high-dimensional space\nsuch that "similar" inputs, as defined by the training scheme, are located close to each\nother. These models once trained can produce embeddings for downstream systems where such\nsimilarity is useful; examples include as a ranking signal for search or as a form of\npretrained embedding model for another supervised problem.\n\nFor a more detailed overview of metric learning see:\n\n* [What is metric learning?](http://contrib.scikit-learn.org/metric-learn/introduction.html)\n* ["Using crossentropy for metric learning" tutorial](https://www.youtube.com/watch?v=Jb4Ewl5RzkI)\n'
'\n## Setup\n\nSet Keras backend to tensorflow. \n'
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import defaultdict
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
import keras
from keras import layers
'\n## Dataset\n\nFor this example we will be using the\n[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.\n'
from keras.datasets import cifar10
((x_train, y_train), (x_test, y_test)) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
y_train = np.squeeze(y_train)
x_test = x_test.astype('float32') / 255.0
y_test = np.squeeze(y_test)
'\nTo get a sense of the dataset we can visualise a grid of 25 random examples.\n\n\n'
height_width = 32

def show_collage(examples):
    if False:
        i = 10
        return i + 15
    box_size = height_width + 2
    (num_rows, num_cols) = examples.shape[:2]
    collage = Image.new(mode='RGB', size=(num_cols * box_size, num_rows * box_size), color=(250, 250, 250))
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            array = (np.array(examples[row_idx, col_idx]) * 255).astype(np.uint8)
            collage.paste(Image.fromarray(array), (col_idx * box_size, row_idx * box_size))
    collage = collage.resize((2 * num_cols * box_size, 2 * num_rows * box_size))
    return collage
sample_idxs = np.random.randint(0, 50000, size=(5, 5))
examples = x_train[sample_idxs]
show_collage(examples)
"\nMetric learning provides training data not as explicit `(X, y)` pairs but instead uses\nmultiple instances that are related in the way we want to express similarity. In our\nexample we will use instances of the same class to represent similarity; a single\ntraining instance will not be one image, but a pair of images of the same class. When\nreferring to the images in this pair we'll use the common metric learning names of the\n`anchor` (a randomly chosen image) and the `positive` (another randomly chosen image of\nthe same class).\n\nTo facilitate this we need to build a form of lookup that maps from classes to the\ninstances of that class. When generating data for training we will sample from this\nlookup.\n"
class_idx_to_train_idxs = defaultdict(list)
for (y_train_idx, y) in enumerate(y_train):
    class_idx_to_train_idxs[y].append(y_train_idx)
class_idx_to_test_idxs = defaultdict(list)
for (y_test_idx, y) in enumerate(y_test):
    class_idx_to_test_idxs[y].append(y_test_idx)
'\nFor this example we are using the simplest approach to training; a batch will consist of\n`(anchor, positive)` pairs spread across the classes. The goal of learning will be to\nmove the anchor and positive pairs closer together and further away from other instances\nin the batch. In this case the batch size will be dictated by the number of classes; for\nCIFAR-10 this is 10.\n'
num_classes = 10

class AnchorPositivePairs(keras.utils.Sequence):

    def __init__(self, num_batches):
        if False:
            while True:
                i = 10
        super().__init__()
        self.num_batches = num_batches

    def __len__(self):
        if False:
            while True:
                i = 10
        return self.num_batches

    def __getitem__(self, _idx):
        if False:
            while True:
                i = 10
        x = np.empty((2, num_classes, height_width, height_width, 3), dtype=np.float32)
        for class_idx in range(num_classes):
            examples_for_class = class_idx_to_train_idxs[class_idx]
            anchor_idx = random.choice(examples_for_class)
            positive_idx = random.choice(examples_for_class)
            while positive_idx == anchor_idx:
                positive_idx = random.choice(examples_for_class)
            x[0, class_idx] = x_train[anchor_idx]
            x[1, class_idx] = x_train[positive_idx]
        return x
'\nWe can visualise a batch in another collage. The top row shows randomly chosen anchors\nfrom the 10 classes, the bottom row shows the corresponding 10 positives.\n'
examples = next(iter(AnchorPositivePairs(num_batches=1)))
show_collage(examples)
'\n## Embedding model\n\nWe define a custom model with a `train_step` that first embeds both anchors and positives\nand then uses their pairwise dot products as logits for a softmax.\n'

class EmbeddingModel(keras.Model):

    def train_step(self, data):
        if False:
            while True:
                i = 10
        if isinstance(data, tuple):
            data = data[0]
        (anchors, positives) = (data[0], data[1])
        with tf.GradientTape() as tape:
            anchor_embeddings = self(anchors, training=True)
            positive_embeddings = self(positives, training=True)
            similarities = keras.ops.einsum('ae,pe->ap', anchor_embeddings, positive_embeddings)
            temperature = 0.2
            similarities /= temperature
            sparse_labels = keras.ops.arange(num_classes)
            loss = self.compiled_loss(sparse_labels, similarities)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(sparse_labels, similarities)
        return {m.name: m.result() for m in self.metrics}
'\nNext we describe the architecture that maps from an image to an embedding. This model\nsimply consists of a sequence of 2d convolutions followed by global pooling with a final\nlinear projection to an embedding space. As is common in metric learning we normalise the\nembeddings so that we can use simple dot products to measure similarity. For simplicity\nthis model is intentionally small.\n'
inputs = layers.Input(shape=(height_width, height_width, 3))
x = layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu')(inputs)
x = layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='relu')(x)
x = layers.Conv2D(filters=128, kernel_size=3, strides=2, activation='relu')(x)
x = layers.GlobalAveragePooling2D()(x)
embeddings = layers.Dense(units=8, activation=None)(x)
embeddings = layers.UnitNormalization()(embeddings)
model = EmbeddingModel(inputs, embeddings)
'\nFinally we run the training. On a Google Colab GPU instance this takes about a minute.\n'
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
history = model.fit(AnchorPositivePairs(num_batches=1000), epochs=20)
plt.plot(history.history['loss'])
plt.show()
'\n## Testing\n\nWe can review the quality of this model by applying it to the test set and considering\nnear neighbours in the embedding space.\n\nFirst we embed the test set and calculate all near neighbours. Recall that since the\nembeddings are unit length we can calculate cosine similarity via dot products.\n'
near_neighbours_per_example = 10
embeddings = model.predict(x_test)
gram_matrix = np.einsum('ae,be->ab', embeddings, embeddings)
near_neighbours = np.argsort(gram_matrix.T)[:, -(near_neighbours_per_example + 1):]
'\nAs a visual check of these embeddings we can build a collage of the near neighbours for 5\nrandom examples. The first column of the image below is a randomly selected image, the\nfollowing 10 columns show the nearest neighbours in order of similarity.\n'
num_collage_examples = 5
examples = np.empty((num_collage_examples, near_neighbours_per_example + 1, height_width, height_width, 3), dtype=np.float32)
for row_idx in range(num_collage_examples):
    examples[row_idx, 0] = x_test[row_idx]
    anchor_near_neighbours = reversed(near_neighbours[row_idx][:-1])
    for (col_idx, nn_idx) in enumerate(anchor_near_neighbours):
        examples[row_idx, col_idx + 1] = x_test[nn_idx]
show_collage(examples)
'\nWe can also get a quantified view of the performance by considering the correctness of\nnear neighbours in terms of a confusion matrix.\n\nLet us sample 10 examples from each of the 10 classes and consider their near neighbours\nas a form of prediction; that is, does the example and its near neighbours share the same\nclass?\n\nWe observe that each animal class does generally well, and is confused the most with the\nother animal classes. The vehicle classes follow the same pattern.\n'
confusion_matrix = np.zeros((num_classes, num_classes))
for class_idx in range(num_classes):
    example_idxs = class_idx_to_test_idxs[class_idx][:10]
    for y_test_idx in example_idxs:
        for nn_idx in near_neighbours[y_test_idx][:-1]:
            nn_class_idx = y_test[nn_idx]
            confusion_matrix[class_idx, nn_class_idx] += 1
labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='vertical')
plt.show()