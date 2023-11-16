"""
Title: Semantic Image Clustering
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Date created: 2021/02/28
Last modified: 2021/02/28
Description: Semantic Clustering by Adopting Nearest neighbors (SCAN) algorithm.
Accelerator: GPU
"""
'\n## Introduction\n\nThis example demonstrates how to apply the [Semantic Clustering by Adopting Nearest neighbors\n(SCAN)](https://arxiv.org/abs/2005.12320) algorithm (Van Gansbeke et al., 2020) on the\n[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. The algorithm consists of\ntwo phases:\n\n1. Self-supervised visual representation learning of images, in which we use the\n[simCLR](https://arxiv.org/abs/2002.05709) technique.\n2. Clustering of the learned visual representation vectors to maximize the agreement\nbetween the cluster assignments of neighboring vectors.\n'
'\n## Setup\n'
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from collections import defaultdict
import numpy as np
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
from tqdm import tqdm
'\n## Prepare the data\n'
num_classes = 10
input_shape = (32, 32, 3)
((x_train, y_train), (x_test, y_test)) = keras.datasets.cifar10.load_data()
x_data = np.concatenate([x_train, x_test])
y_data = np.concatenate([y_train, y_test])
print('x_data shape:', x_data.shape, '- y_data shape:', y_data.shape)
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
'\n## Define hyperparameters\n'
target_size = 32
representation_dim = 512
projection_units = 128
num_clusters = 20
k_neighbours = 5
tune_encoder_during_clustering = False
'\n## Implement data preprocessing\n\nThe data preprocessing step resizes the input images to the desired `target_size` and applies\nfeature-wise normalization. Note that, when using `keras.applications.ResNet50V2` as the\nvisual encoder, resizing the images into 255 x 255 inputs would lead to more accurate results\nbut require a longer time to train.\n'
data_preprocessing = keras.Sequential([layers.Resizing(target_size, target_size), layers.Normalization()])
data_preprocessing.layers[-1].adapt(x_data)
'\n## Data augmentation\n\nUnlike simCLR, which randomly picks a single data augmentation function to apply to an input\nimage, we apply a set of data augmentation functions randomly to the input image.\n(You can experiment with other image augmentation techniques by following\nthe [data augmentation tutorial](https://www.tensorflow.org/tutorials/images/data_augmentation).)\n'
data_augmentation = keras.Sequential([layers.RandomTranslation(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), fill_mode='nearest'), layers.RandomFlip(mode='horizontal'), layers.RandomRotation(factor=0.15, fill_mode='nearest'), layers.RandomZoom(height_factor=(-0.3, 0.1), width_factor=(-0.3, 0.1), fill_mode='nearest')])
'\nDisplay a random image\n'
image_idx = np.random.choice(range(x_data.shape[0]))
image = x_data[image_idx]
image_class = classes[y_data[image_idx][0]]
plt.figure(figsize=(3, 3))
plt.imshow(x_data[image_idx].astype('uint8'))
plt.title(image_class)
_ = plt.axis('off')
'\nDisplay a sample of augmented versions of the image\n'
plt.figure(figsize=(10, 10))
for i in range(9):
    augmented_images = data_augmentation(np.array([image]))
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype('uint8'))
    plt.axis('off')
'\n## Self-supervised representation learning\n'
'\n### Implement the vision encoder\n'

def create_encoder(representation_dim):
    if False:
        print('Hello World!')
    encoder = keras.Sequential([keras.applications.ResNet50V2(include_top=False, weights=None, pooling='avg'), layers.Dense(representation_dim)])
    return encoder
'\n### Implement the unsupervised contrastive loss\n'

class RepresentationLearner(keras.Model):

    def __init__(self, encoder, projection_units, num_augmentations, temperature=1.0, dropout_rate=0.1, l2_normalize=False, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.encoder = encoder
        self.projector = keras.Sequential([layers.Dropout(dropout_rate), layers.Dense(units=projection_units, use_bias=False), layers.BatchNormalization(), layers.ReLU()])
        self.num_augmentations = num_augmentations
        self.temperature = temperature
        self.l2_normalize = l2_normalize
        self.loss_tracker = keras.metrics.Mean(name='loss')

    @property
    def metrics(self):
        if False:
            return 10
        return [self.loss_tracker]

    def compute_contrastive_loss(self, feature_vectors, batch_size):
        if False:
            while True:
                i = 10
        num_augmentations = keras.ops.shape(feature_vectors)[0] // batch_size
        if self.l2_normalize:
            feature_vectors = keras.utils.normalize(feature_vectors)
        logits = tf.linalg.matmul(feature_vectors, feature_vectors, transpose_b=True) / self.temperature
        logits_max = keras.ops.max(logits, axis=1)
        logits = logits - logits_max
        targets = keras.ops.tile(tf.eye(batch_size), [num_augmentations, num_augmentations])
        return keras.losses.categorical_crossentropy(y_true=targets, y_pred=logits, from_logits=True)

    def call(self, inputs):
        if False:
            i = 10
            return i + 15
        preprocessed = data_preprocessing(inputs)
        augmented = []
        for _ in range(self.num_augmentations):
            augmented.append(data_augmentation(preprocessed))
        augmented = layers.Concatenate(axis=0)(augmented)
        features = self.encoder(augmented)
        return self.projector(features)

    def train_step(self, inputs):
        if False:
            print('Hello World!')
        batch_size = keras.ops.shape(inputs)[0]
        with tf.GradientTape() as tape:
            feature_vectors = self(inputs, training=True)
            loss = self.compute_contrastive_loss(feature_vectors, batch_size)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_tracker.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        if False:
            return 10
        batch_size = keras.ops.shape(inputs)[0]
        feature_vectors = self(inputs, training=False)
        loss = self.compute_contrastive_loss(feature_vectors, batch_size)
        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}
'\n### Train the model\n'
encoder = create_encoder(representation_dim)
representation_learner = RepresentationLearner(encoder, projection_units, num_augmentations=2, temperature=0.1)
lr_scheduler = keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.001, decay_steps=500, alpha=0.1)
representation_learner.compile(optimizer=keras.optimizers.AdamW(learning_rate=lr_scheduler, weight_decay=0.0001), jit_compile=False)
history = representation_learner.fit(x=x_data, batch_size=512, epochs=50)
'\nPlot training loss\n'
plt.plot(history.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
'\n## Compute the nearest neighbors\n'
'\n### Generate the embeddings for the images\n'
batch_size = 500
feature_vectors = encoder.predict(x_data, batch_size=batch_size, verbose=1)
feature_vectors = keras.utils.normalize(feature_vectors)
'\n### Find the *k* nearest neighbours for each embedding\n'
neighbours = []
num_batches = feature_vectors.shape[0] // batch_size
for batch_idx in tqdm(range(num_batches)):
    start_idx = batch_idx * batch_size
    end_idx = start_idx + batch_size
    current_batch = feature_vectors[start_idx:end_idx]
    similarities = tf.linalg.matmul(current_batch, feature_vectors, transpose_b=True)
    (_, indices) = keras.ops.top_k(similarities, k=k_neighbours + 1, sorted=True)
    neighbours.append(indices[..., 1:])
neighbours = np.reshape(np.array(neighbours), (-1, k_neighbours))
"\nLet's display some neighbors on each row\n"
nrows = 4
ncols = k_neighbours + 1
plt.figure(figsize=(12, 12))
position = 1
for _ in range(nrows):
    anchor_idx = np.random.choice(range(x_data.shape[0]))
    neighbour_indicies = neighbours[anchor_idx]
    indices = [anchor_idx] + neighbour_indicies.tolist()
    for j in range(ncols):
        plt.subplot(nrows, ncols, position)
        plt.imshow(x_data[indices[j]].astype('uint8'))
        plt.title(classes[y_data[indices[j]][0]])
        plt.axis('off')
        position += 1
'\nYou notice that images on each row are visually similar, and belong to similar classes.\n'
'\n## Semantic clustering with nearest neighbours\n'
'\n### Implement clustering consistency loss\n\nThis loss tries to make sure that neighbours have the same clustering assignments.\n'

class ClustersConsistencyLoss(keras.losses.Loss):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()

    def __call__(self, target, similarity, sample_weight=None):
        if False:
            return 10
        target = keras.ops.ones_like(similarity)
        loss = keras.losses.binary_crossentropy(y_true=target, y_pred=similarity, from_logits=True)
        return keras.ops.mean(loss)
'\n### Implement the clusters entropy loss\n\nThis loss tries to make sure that cluster distribution is roughly uniformed, to avoid\nassigning most of the instances to one cluster.\n'

class ClustersEntropyLoss(keras.losses.Loss):

    def __init__(self, entropy_loss_weight=1.0):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.entropy_loss_weight = entropy_loss_weight

    def __call__(self, target, cluster_probabilities, sample_weight=None):
        if False:
            while True:
                i = 10
        num_clusters = keras.ops.cast(keras.ops.shape(cluster_probabilities)[-1], 'float32')
        target = keras.ops.log(num_clusters)
        cluster_probabilities = keras.ops.mean(cluster_probabilities, axis=0)
        cluster_probabilities = keras.ops.clip(cluster_probabilities, 1e-08, 1.0)
        entropy = -keras.ops.sum(cluster_probabilities * keras.ops.log(cluster_probabilities))
        loss = target - entropy
        return loss
'\n### Implement clustering model\n\nThis model takes a raw image as an input, generated its feature vector using the trained\nencoder, and produces a probability distribution of the clusters given the feature vector\nas the cluster assignments.\n'

def create_clustering_model(encoder, num_clusters, name=None):
    if False:
        while True:
            i = 10
    inputs = keras.Input(shape=input_shape)
    preprocessed = data_preprocessing(inputs)
    augmented = data_augmentation(preprocessed)
    features = encoder(augmented)
    outputs = layers.Dense(units=num_clusters, activation='softmax')(features)
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model
'\n### Implement clustering learner\n\nThis model receives the input `anchor` image and its `neighbours`, produces the clusters\nassignments for them using the `clustering_model`, and produces two outputs:\n1. `similarity`: the similarity between the cluster assignments of the `anchor` image and\nits `neighbours`. This output is fed to the `ClustersConsistencyLoss`.\n2. `anchor_clustering`: cluster assignments of the `anchor` images. This is fed to the `ClustersEntropyLoss`.\n'

def create_clustering_learner(clustering_model):
    if False:
        print('Hello World!')
    anchor = keras.Input(shape=input_shape, name='anchors')
    neighbours = keras.Input(shape=tuple([k_neighbours]) + input_shape, name='neighbours')
    neighbours_reshaped = keras.ops.reshape(neighbours, tuple([-1]) + input_shape)
    anchor_clustering = clustering_model(anchor)
    neighbours_clustering = clustering_model(neighbours_reshaped)
    neighbours_clustering = keras.ops.reshape(neighbours_clustering, (-1, k_neighbours, keras.ops.shape(neighbours_clustering)[-1]))
    similarity = keras.ops.einsum('bij,bkj->bik', keras.ops.expand_dims(anchor_clustering, axis=1), neighbours_clustering)
    similarity = layers.Lambda(lambda x: keras.ops.squeeze(x, axis=1), name='similarity')(similarity)
    model = keras.Model(inputs=[anchor, neighbours], outputs=[similarity, anchor_clustering], name='clustering_learner')
    return model
'\n### Train model\n'
for layer in encoder.layers:
    layer.trainable = tune_encoder_during_clustering
clustering_model = create_clustering_model(encoder, num_clusters, name='clustering')
clustering_learner = create_clustering_learner(clustering_model)
losses = [ClustersConsistencyLoss(), ClustersEntropyLoss(entropy_loss_weight=5)]
inputs = {'anchors': x_data, 'neighbours': tf.gather(x_data, neighbours)}
labels = np.ones(shape=x_data.shape[0])
clustering_learner.compile(optimizer=keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=0.0001), loss=losses, jit_compile=False)
clustering_learner.fit(x=inputs, y=labels, batch_size=512, epochs=50)
'\nPlot training loss\n'
plt.plot(history.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
'\n## Cluster analysis\n'
'\n### Assign images to clusters\n'
clustering_probs = clustering_model.predict(x_data, batch_size=batch_size, verbose=1)
cluster_assignments = keras.ops.argmax(clustering_probs, axis=-1).numpy()
cluster_confidence = keras.ops.max(clustering_probs, axis=-1).numpy()
"\nLet's compute the cluster sizes\n"
clusters = defaultdict(list)
for (idx, c) in enumerate(cluster_assignments):
    clusters[c].append((idx, cluster_confidence[idx]))
non_empty_clusters = defaultdict(list)
for c in clusters.keys():
    if clusters[c]:
        non_empty_clusters[c] = clusters[c]
for c in range(num_clusters):
    print('cluster', c, ':', len(clusters[c]))
'\n### Visualize cluster images\n\nDisplay the *prototypes*—instances with the highest clustering confidence—of each cluster:\n'
num_images = 8
plt.figure(figsize=(15, 15))
position = 1
for c in non_empty_clusters.keys():
    cluster_instances = sorted(non_empty_clusters[c], key=lambda kv: kv[1], reverse=True)
    for j in range(num_images):
        image_idx = cluster_instances[j][0]
        plt.subplot(len(non_empty_clusters), num_images, position)
        plt.imshow(x_data[image_idx].astype('uint8'))
        plt.title(classes[y_data[image_idx][0]])
        plt.axis('off')
        position += 1
'\n### Compute clustering accuracy\n\nFirst, we assign a label for each cluster based on the majority label of its images.\nThen, we compute the accuracy of each cluster by dividing the number of image with the\nmajority label by the size of the cluster.\n'
cluster_label_counts = dict()
for c in range(num_clusters):
    cluster_label_counts[c] = [0] * num_classes
    instances = clusters[c]
    for (i, _) in instances:
        cluster_label_counts[c][y_data[i][0]] += 1
    cluster_label_idx = np.argmax(cluster_label_counts[c])
    correct_count = np.max(cluster_label_counts[c])
    cluster_size = len(clusters[c])
    accuracy = np.round(correct_count / cluster_size * 100, 2) if cluster_size > 0 else 0
    cluster_label = classes[cluster_label_idx]
    print('cluster', c, 'label is:', cluster_label, ' -  accuracy:', accuracy, '%')
'\n## Conclusion\n\nTo improve the accuracy results, you can: 1) increase the number\nof epochs in the representation learning and the clustering phases; 2)\nallow the encoder weights to be tuned during the clustering phase; and 3) perform a final\nfine-tuning step through self-labeling, as described in the [original SCAN paper](https://arxiv.org/abs/2005.12320).\nNote that unsupervised image clustering techniques are not expected to outperform the accuracy\nof supervised image classification techniques, rather showing that they can learn the semantics\nof the images and group them into clusters that are similar to their original classes.\n'