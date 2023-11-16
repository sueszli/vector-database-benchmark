"""
Title: Image similarity estimation using a Siamese Network with a triplet loss
Authors: [Hazem Essam](https://twitter.com/hazemessamm) and [Santiago L. Valdarrama](https://twitter.com/svpino)
Date created: 2021/03/25
Last modified: 2021/03/25
Description: Training a Siamese Network to compare the similarity of images using a triplet loss function.
Accelerator: GPU
"""
'\n## Introduction\n\nA [Siamese Network](https://en.wikipedia.org/wiki/Siamese_neural_network) is a type of network architecture that\ncontains two or more identical subnetworks used to generate feature vectors for each input and compare them.\n\nSiamese Networks can be applied to different use cases, like detecting duplicates, finding anomalies, and face recognition.\n\nThis example uses a Siamese Network with three identical subnetworks. We will provide three images to the model, where\ntwo of them will be similar (_anchor_ and _positive_ samples), and the third will be unrelated (a _negative_ example.)\nOur goal is for the model to learn to estimate the similarity between images.\n\nFor the network to learn, we use a triplet loss function. You can find an introduction to triplet loss in the\n[FaceNet paper](https://arxiv.org/pdf/1503.03832.pdf) by Schroff et al,. 2015. In this example, we define the triplet\nloss function as follows:\n\n`L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)`\n\nThis example uses the [Totally Looks Like dataset](https://sites.google.com/view/totally-looks-like-dataset)\nby [Rosenfeld et al., 2018](https://arxiv.org/pdf/1803.01485v3.pdf).\n'
'\n## Setup\n'
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from pathlib import Path
from keras import layers
from keras import optimizers
from keras import metrics
from keras import Model
from keras.applications import resnet
target_shape = (200, 200)
'\n## Load the dataset\n\nWe are going to load the *Totally Looks Like* dataset and unzip it inside the `~/.keras` directory\nin the local environment.\n\nThe dataset consists of two separate files:\n\n* `left.zip` contains the images that we will use as the anchor.\n* `right.zip` contains the images that we will use as the positive sample (an image that looks like the anchor).\n'
cache_dir = Path(Path.home()) / '.keras'
anchor_images_path = cache_dir / 'left'
positive_images_path = cache_dir / 'right'
'shell\ngdown --id 1jvkbTr_giSP3Ru8OwGNCg6B4PvVbcO34\ngdown --id 1EzBZUb_mh_Dp_FKD0P4XiYYSd0QBH5zW\nunzip -oq left.zip -d $cache_dir\nunzip -oq right.zip -d $cache_dir\n'
"\n## Preparing the data\n\nWe are going to use a `tf.data` pipeline to load the data and generate the triplets that we\nneed to train the Siamese network.\n\nWe'll set up the pipeline using a zipped list with anchor, positive, and negative filenames as\nthe source. The pipeline will load and preprocess the corresponding images.\n"

def preprocess_image(filename):
    if False:
        for i in range(10):
            print('nop')
    '\n    Load the specified file as a JPEG image, preprocess it and\n    resize it to the target shape.\n    '
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image

def preprocess_triplets(anchor, positive, negative):
    if False:
        print('Hello World!')
    '\n    Given the filenames corresponding to the three images, load and\n    preprocess them.\n    '
    return (preprocess_image(anchor), preprocess_image(positive), preprocess_image(negative))
"\nLet's setup our data pipeline using a zipped list with an anchor, positive,\nand negative image filename as the source. The output of the pipeline\ncontains the same triplet with every image loaded and preprocessed.\n"
anchor_images = sorted([str(anchor_images_path / f) for f in os.listdir(anchor_images_path)])
positive_images = sorted([str(positive_images_path / f) for f in os.listdir(positive_images_path)])
image_count = len(anchor_images)
anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)
rng = np.random.RandomState(seed=42)
rng.shuffle(anchor_images)
rng.shuffle(positive_images)
negative_images = anchor_images + positive_images
np.random.RandomState(seed=32).shuffle(negative_images)
negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)
negative_dataset = negative_dataset.shuffle(buffer_size=4096)
dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.map(preprocess_triplets)
train_dataset = dataset.take(round(image_count * 0.8))
val_dataset = dataset.skip(round(image_count * 0.8))
train_dataset = train_dataset.batch(32, drop_remainder=False)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(32, drop_remainder=False)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
"\nLet's take a look at a few examples of triplets. Notice how the first two images\nlook alike while the third one is always different.\n"

def visualize(anchor, positive, negative):
    if False:
        return 10
    'Visualize a few triplets from the supplied batches.'

    def show(ax, image):
        if False:
            for i in range(10):
                print('nop')
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig = plt.figure(figsize=(9, 9))
    axs = fig.subplots(3, 3)
    for i in range(3):
        show(axs[i, 0], anchor[i])
        show(axs[i, 1], positive[i])
        show(axs[i, 2], negative[i])
visualize(*list(train_dataset.take(1).as_numpy_iterator())[0])
'\n## Setting up the embedding generator model\n\nOur Siamese Network will generate embeddings for each of the images of the\ntriplet. To do this, we will use a ResNet50 model pretrained on ImageNet and\nconnect a few `Dense` layers to it so we can learn to separate these\nembeddings.\n\nWe will freeze the weights of all the layers of the model up until the layer `conv5_block1_out`.\nThis is important to avoid affecting the weights that the model has already learned.\nWe are going to leave the bottom few layers trainable, so that we can fine-tune their weights\nduring training.\n'
base_cnn = resnet.ResNet50(weights='imagenet', input_shape=target_shape + (3,), include_top=False)
flatten = layers.Flatten()(base_cnn.output)
dense1 = layers.Dense(512, activation='relu')(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(256, activation='relu')(dense1)
dense2 = layers.BatchNormalization()(dense2)
output = layers.Dense(256)(dense2)
embedding = Model(base_cnn.input, output, name='Embedding')
trainable = False
for layer in base_cnn.layers:
    if layer.name == 'conv5_block1_out':
        trainable = True
    layer.trainable = trainable
'\n## Setting up the Siamese Network model\n\nThe Siamese network will receive each of the triplet images as an input,\ngenerate the embeddings, and output the distance between the anchor and the\npositive embedding, as well as the distance between the anchor and the negative\nembedding.\n\nTo compute the distance, we can use a custom layer `DistanceLayer` that\nreturns both values as a tuple.\n'

class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        if False:
            i = 10
            return i + 15
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)
anchor_input = layers.Input(name='anchor', shape=target_shape + (3,))
positive_input = layers.Input(name='positive', shape=target_shape + (3,))
negative_input = layers.Input(name='negative', shape=target_shape + (3,))
distances = DistanceLayer()(embedding(resnet.preprocess_input(anchor_input)), embedding(resnet.preprocess_input(positive_input)), embedding(resnet.preprocess_input(negative_input)))
siamese_network = Model(inputs=[anchor_input, positive_input, negative_input], outputs=distances)
"\n## Putting everything together\n\nWe now need to implement a model with custom training loop so we can compute\nthe triplet loss using the three embeddings produced by the Siamese network.\n\nLet's create a `Mean` metric instance to track the loss of the training process.\n"

class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name='loss')

    def call(self, inputs):
        if False:
            print('Hello World!')
        return self.siamese_network(inputs)

    def train_step(self, data):
        if False:
            while True:
                i = 10
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}

    def test_step(self, data):
        if False:
            for i in range(10):
                print('nop')
        loss = self._compute_loss(data)
        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}

    def _compute_loss(self, data):
        if False:
            i = 10
            return i + 15
        (ap_distance, an_distance) = self.siamese_network(data)
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        if False:
            print('Hello World!')
        return [self.loss_tracker]
'\n## Training\n\nWe are now ready to train our model.\n'
siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(0.0001))
siamese_model.fit(train_dataset, epochs=10, validation_data=val_dataset)
"\n## Inspecting what the network has learned\n\nAt this point, we can check how the network learned to separate the embeddings\ndepending on whether they belong to similar images.\n\nWe can use [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) to measure the\nsimilarity between embeddings.\n\nLet's pick a sample from the dataset to check the similarity between the\nembeddings generated for each image.\n"
sample = next(iter(train_dataset))
visualize(*sample)
(anchor, positive, negative) = sample
(anchor_embedding, positive_embedding, negative_embedding) = (embedding(resnet.preprocess_input(anchor)), embedding(resnet.preprocess_input(positive)), embedding(resnet.preprocess_input(negative)))
'\nFinally, we can compute the cosine similarity between the anchor and positive\nimages and compare it with the similarity between the anchor and the negative\nimages.\n\nWe should expect the similarity between the anchor and positive images to be\nlarger than the similarity between the anchor and the negative images.\n'
cosine_similarity = metrics.CosineSimilarity()
positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
print('Positive similarity:', positive_similarity.numpy())
negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
print('Negative similarity', negative_similarity.numpy())
'\n## Summary\n\n1. The `tf.data` API enables you to build efficient input pipelines for your model. It is\nparticularly useful if you have a large dataset. You can learn more about `tf.data`\npipelines in [tf.data: Build TensorFlow input pipelines](https://www.tensorflow.org/guide/data).\n\n2. In this example, we use a pre-trained ResNet50 as part of the subnetwork that generates\nthe feature embeddings. By using [transfer learning](https://www.tensorflow.org/guide/keras/transfer_learning?hl=en),\nwe can significantly reduce the training time and size of the dataset.\n\n3. Notice how we are [fine-tuning](https://www.tensorflow.org/guide/keras/transfer_learning?hl=en#fine-tuning)\nthe weights of the final layers of the ResNet50 network but keeping the rest of the layers untouched.\nUsing the name assigned to each layer, we can freeze the weights to a certain point and keep the last few layers open.\n\n4. We can create custom layers by creating a class that inherits from `tf.keras.layers.Layer`,\nas we did in the `DistanceLayer` class.\n\n5. We used a cosine similarity metric to measure how to 2 output embeddings are similar to each other.\n\n6. You can implement a custom training loop by overriding the `train_step()` method. `train_step()` uses\n[`tf.GradientTape`](https://www.tensorflow.org/api_docs/python/tf/GradientTape),\nwhich records every operation that you perform inside it. In this example, we use it to access the\ngradients passed to the optimizer to update the model weights at every step. For more details, check out the\n[Intro to Keras for researchers](https://keras.io/getting_started/intro_to_keras_for_researchers/)\nand [Writing a training loop from scratch](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch?hl=en).\n\n'