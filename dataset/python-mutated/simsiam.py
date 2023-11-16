"""
Title: Self-supervised contrastive learning with SimSiam
Author: [Sayak Paul](https://twitter.com/RisingSayak)
Date created: 2021/03/19
Last modified: 2021/03/20
Description: Implementation of a self-supervised learning method for computer vision.
Accelerator: GPU
"""
'\nSelf-supervised learning (SSL) is an interesting branch of study in the field of\nrepresentation learning. SSL systems try to formulate a supervised signal from a corpus\nof unlabeled data points.  An example is we train a deep neural network to predict the\nnext word from a given set of words. In literature, these tasks are known as *pretext\ntasks* or *auxiliary tasks*. If we [train such a network](https://arxiv.org/abs/1801.06146) on a huge dataset (such as\nthe [Wikipedia text corpus](https://www.corpusdata.org/wikipedia.asp)) it learns very effective\nrepresentations that transfer well to downstream tasks. Language models like\n[BERT](https://arxiv.org/abs/1810.04805), [GPT-3](https://arxiv.org/abs/2005.14165),\n[ELMo](https://allennlp.org/elmo) all benefit from this.\n\nMuch like the language models we can train computer vision models using similar\napproaches. To make things work in computer vision, we need to formulate the learning\ntasks such that the underlying model (a deep neural network) is able to make sense of the\nsemantic information present in vision data. One such task is to a model to _contrast_\nbetween two different versions of the same image. The hope is that in this way the model\nwill have learn representations where the similar images are grouped as together possible\nwhile the dissimilar images are further away.\n\nIn this example, we will be implementing one such system called **SimSiam** proposed in\n[Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566). It\nis implemented as the following:\n\n1. We create two different versions of the same dataset with a stochastic data\naugmentation pipeline. Note that the random initialization seed needs to be the same\nduring create these versions.\n2. We take a ResNet without any classification head (**backbone**) and we add a shallow\nfully-connected network (**projection head**) on top of it. Collectively, this is known\nas the **encoder**.\n3. We pass the output of the encoder through a **predictor** which is again a shallow\nfully-connected network having an\n[AutoEncoder](https://en.wikipedia.org/wiki/Autoencoder) like structure.\n4. We then train our encoder to maximize the cosine similarity between the two different\nversions of our dataset.\n'
'\n## Setup\n'
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import layers
from keras import regularizers
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
'\n## Define hyperparameters\n'
AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 128
EPOCHS = 5
CROP_TO = 32
SEED = 26
PROJECT_DIM = 2048
LATENT_DIM = 512
WEIGHT_DECAY = 0.0005
'\n## Load the CIFAR-10 dataset\n'
((x_train, y_train), (x_test, y_test)) = keras.datasets.cifar10.load_data()
print(f'Total training examples: {len(x_train)}')
print(f'Total test examples: {len(x_test)}')
'\n## Defining our data augmentation pipeline\n\nAs studied in [SimCLR](https://arxiv.org/abs/2002.05709) having the right data\naugmentation pipeline is critical for SSL systems to work effectively in computer vision.\nTwo particular augmentation transforms that seem to matter the most are: 1.) Random\nresized crops and 2.) Color distortions. Most of the other SSL systems for computer\nvision (such as [BYOL](https://arxiv.org/abs/2006.07733),\n[MoCoV2](https://arxiv.org/abs/2003.04297), [SwAV](https://arxiv.org/abs/2006.09882),\netc.) include these in their training pipelines.\n'

def flip_random_crop(image):
    if False:
        return 10
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, (CROP_TO, CROP_TO, 3))
    return image

def color_jitter(x, strength=[0.4, 0.4, 0.4, 0.1]):
    if False:
        return 10
    x = tf.image.random_brightness(x, max_delta=0.8 * strength[0])
    x = tf.image.random_contrast(x, lower=1 - 0.8 * strength[1], upper=1 + 0.8 * strength[1])
    x = tf.image.random_saturation(x, lower=1 - 0.8 * strength[2], upper=1 + 0.8 * strength[2])
    x = tf.image.random_hue(x, max_delta=0.2 * strength[3])
    x = tf.clip_by_value(x, 0, 255)
    return x

def color_drop(x):
    if False:
        while True:
            i = 10
    x = tf.image.rgb_to_grayscale(x)
    x = tf.tile(x, [1, 1, 3])
    return x

def random_apply(func, x, p):
    if False:
        while True:
            i = 10
    if tf.random.uniform([], minval=0, maxval=1) < p:
        return func(x)
    else:
        return x

def custom_augment(image):
    if False:
        print('Hello World!')
    image = flip_random_crop(image)
    image = random_apply(color_jitter, image, p=0.8)
    image = random_apply(color_drop, image, p=0.2)
    return image
"\nIt should be noted that an augmentation pipeline is generally dependent on various\nproperties of the dataset we are dealing with. For example, if images in the dataset are\nheavily object-centric then taking random crops with a very high probability may hurt the\ntraining performance.\n\nLet's now apply our augmentation pipeline to our dataset and visualize a few outputs.\n"
'\n## Convert the data into TensorFlow `Dataset` objects\n\nHere we create two different versions of our dataset *without* any ground-truth labels.\n'
ssl_ds_one = tf.data.Dataset.from_tensor_slices(x_train)
ssl_ds_one = ssl_ds_one.shuffle(1024, seed=SEED).map(custom_augment, num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO)
ssl_ds_two = tf.data.Dataset.from_tensor_slices(x_train)
ssl_ds_two = ssl_ds_two.shuffle(1024, seed=SEED).map(custom_augment, num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO)
ssl_ds = tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two))
sample_images_one = next(iter(ssl_ds_one))
plt.figure(figsize=(10, 10))
for n in range(25):
    ax = plt.subplot(5, 5, n + 1)
    plt.imshow(sample_images_one[n].numpy().astype('int'))
    plt.axis('off')
plt.show()
sample_images_two = next(iter(ssl_ds_two))
plt.figure(figsize=(10, 10))
for n in range(25):
    ax = plt.subplot(5, 5, n + 1)
    plt.imshow(sample_images_two[n].numpy().astype('int'))
    plt.axis('off')
plt.show()
'\nNotice that the images in `samples_images_one` and `sample_images_two` are essentially\nthe same but are augmented differently.\n'
'\n## Defining the encoder and the predictor\n\nWe use an implementation of ResNet20 that is specifically configured for the CIFAR10\ndataset. The code is taken from the\n[keras-idiomatic-programmer](https://github.com/GoogleCloudPlatform/keras-idiomatic-programmer/blob/master/zoo/resnet/resnet_cifar10_v2.py) repository. The hyperparameters of\nthese architectures have been referred from Section 3 and Appendix A of [the original\npaper](https://arxiv.org/abs/2011.10566).\n'
'shell\nwget -q https://shorturl.at/QS369 -O resnet_cifar10_v2.py\n'
import resnet_cifar10_v2
N = 2
DEPTH = N * 9 + 2
NUM_BLOCKS = (DEPTH - 2) // 9 - 1

def get_encoder():
    if False:
        while True:
            i = 10
    inputs = layers.Input((CROP_TO, CROP_TO, 3))
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1)(inputs)
    x = resnet_cifar10_v2.stem(x)
    x = resnet_cifar10_v2.learner(x, NUM_BLOCKS)
    x = layers.GlobalAveragePooling2D(name='backbone_pool')(x)
    x = layers.Dense(PROJECT_DIM, use_bias=False, kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(PROJECT_DIM, use_bias=False, kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)
    outputs = layers.BatchNormalization()(x)
    return keras.Model(inputs, outputs, name='encoder')

def get_predictor():
    if False:
        print('Hello World!')
    model = keras.Sequential([layers.Input((PROJECT_DIM,)), layers.Dense(LATENT_DIM, use_bias=False, kernel_regularizer=regularizers.l2(WEIGHT_DECAY)), layers.ReLU(), layers.BatchNormalization(), layers.Dense(PROJECT_DIM)], name='predictor')
    return model
'\n## Defining the (pre-)training loop\n\nOne of the main reasons behind training networks with these kinds of approaches is to\nutilize the learned representations for downstream tasks like classification. This is why\nthis particular training phase is also referred to as _pre-training_.\n\nWe start by defining the loss function.\n'

def compute_loss(p, z):
    if False:
        while True:
            i = 10
    z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    return -tf.reduce_mean(tf.reduce_sum(p * z, axis=1))
'\nWe then define our training loop by overriding the `train_step()` function of the\n`keras.Model` class.\n'

class SimSiam(keras.Model):

    def __init__(self, encoder, predictor):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.loss_tracker = keras.metrics.Mean(name='loss')

    @property
    def metrics(self):
        if False:
            print('Hello World!')
        return [self.loss_tracker]

    def train_step(self, data):
        if False:
            i = 10
            return i + 15
        (ds_one, ds_two) = data
        with tf.GradientTape() as tape:
            (z1, z2) = (self.encoder(ds_one), self.encoder(ds_two))
            (p1, p2) = (self.predictor(z1), self.predictor(z2))
            loss = compute_loss(p1, z2) / 2 + compute_loss(p2, z1) / 2
        learnable_params = self.encoder.trainable_variables + self.predictor.trainable_variables
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))
        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}
'\n## Pre-training our networks\n\nIn the interest of this example, we will train the model for only 5 epochs. In reality,\nthis should at least be 100 epochs.\n'
num_training_samples = len(x_train)
steps = EPOCHS * (num_training_samples // BATCH_SIZE)
lr_decayed_fn = keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.03, decay_steps=steps)
early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
simsiam = SimSiam(get_encoder(), get_predictor())
simsiam.compile(optimizer=keras.optimizers.SGD(lr_decayed_fn, momentum=0.6))
history = simsiam.fit(ssl_ds, epochs=EPOCHS, callbacks=[early_stopping])
plt.plot(history.history['loss'])
plt.grid()
plt.title('Negative Cosine Similairty')
plt.show()
'\nIf your solution gets very close to -1 (minimum value of our loss) very quickly with a\ndifferent dataset and a different backbone architecture that is likely because of\n*representation collapse*. It is a phenomenon where the encoder yields similar output for\nall the images. In that case additional hyperparameter tuning is required especially in\nthe following areas:\n\n* Strength of the color distortions and their probabilities.\n* Learning rate and its schedule.\n* Architecture of both the backbone and their projection head.\n\n'
'\n## Evaluating our SSL method\n\nThe most popularly used method to evaluate a SSL method in computer vision (or any other\npre-training method as such) is to learn a linear classifier on the frozen features of\nthe trained backbone model (in this case it is ResNet20) and evaluate the classifier on\nunseen images. Other methods include\n[fine-tuning](https://keras.io/guides/transfer_learning/) on the source dataset or even a\ntarget dataset with 5% or 10% labels present. Practically, we can use the backbone model\nfor any downstream task such as semantic segmentation, object detection, and so on where\nthe backbone models are usually pre-trained with *pure supervised learning*.\n'
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
train_ds = train_ds.shuffle(1024).map(lambda x, y: (flip_random_crop(x), y), num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTO)
backbone = keras.Model(simsiam.encoder.input, simsiam.encoder.get_layer('backbone_pool').output)
backbone.trainable = False
inputs = layers.Input((CROP_TO, CROP_TO, 3))
x = backbone(inputs, training=False)
outputs = layers.Dense(10, activation='softmax')(x)
linear_model = keras.Model(inputs, outputs, name='linear_model')
linear_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=keras.optimizers.SGD(lr_decayed_fn, momentum=0.9))
history = linear_model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=[early_stopping])
(_, test_acc) = linear_model.evaluate(test_ds)
print('Test accuracy: {:.2f}%'.format(test_acc * 100))
'\n\n## Notes\n* More data and longer pre-training schedule benefit SSL in general.\n* SSL is particularly very helpful when you do not have access to very limited *labeled*\ntraining data but you can manage to build a large corpus of unlabeled data. Recently,\nusing an SSL method called [SwAV](https://arxiv.org/abs/2006.09882), a group of\nresearchers at Facebook trained a [RegNet](https://arxiv.org/abs/2006.09882) on 2 Billion\nimages. They were able to achieve downstream performance very close to those achieved by\npure supervised pre-training. For some downstream tasks, their method even outperformed\nthe supervised counterparts. You can check out [their\npaper](https://arxiv.org/pdf/2103.01988.pdf) to know the details.\n* If you are interested to understand why contrastive SSL helps networks learn meaningful\nrepresentations, you can check out the following resources:\n   * [Self-supervised learning: The dark matter of\nintelligence](https://ai.facebook.com/blog/self-supervised-learning-the-dark-matter-of-intelligence/)\n   * [Understanding self-supervised learning using controlled datasets with known\nstructure](https://sslneuips20.github.io/files/CameraReadys%203-77/64/CameraReady/Understanding_self_supervised_learning.pdf)\n\n'