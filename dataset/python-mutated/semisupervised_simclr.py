"""
Title: Semi-supervised image classification using contrastive pretraining with SimCLR
Author: [András Béres](https://www.linkedin.com/in/andras-beres-789190210)
Date created: 2021/04/24
Last modified: 2021/04/24
Description: Contrastive pretraining with SimCLR for semi-supervised image classification on the STL-10 dataset.
Accelerator: GPU
"""
'\n## Introduction\n\n### Semi-supervised learning\n\nSemi-supervised learning is a machine learning paradigm that deals with\n**partially labeled datasets**. When applying deep learning in the real world,\none usually has to gather a large dataset to make it work well. However, while\nthe cost of labeling scales linearly with the dataset size (labeling each\nexample takes a constant time), model performance only scales\n[sublinearly](https://arxiv.org/abs/2001.08361) with it. This means that\nlabeling more and more samples becomes less and less cost-efficient, while\ngathering unlabeled data is generally cheap, as it is usually readily available\nin large quantities.\n\nSemi-supervised learning offers to solve this problem by only requiring a\npartially labeled dataset, and by being label-efficient by utilizing the\nunlabeled examples for learning as well.\n\nIn this example, we will pretrain an encoder with contrastive learning on the\n[STL-10](https://ai.stanford.edu/~acoates/stl10/) semi-supervised dataset using\nno labels at all, and then fine-tune it using only its labeled subset.\n\n### Contrastive learning\n\nOn the highest level, the main idea behind contrastive learning is to **learn\nrepresentations that are invariant to image augmentations** in a self-supervised\nmanner. One problem with this objective is that it has a trivial degenerate\nsolution: the case where the representations are constant, and do not depend at all on the\ninput images.\n\nContrastive learning avoids this trap by modifying the objective in the\nfollowing way: it pulls representations of augmented versions/views of the same\nimage closer to each other (contracting positives), while simultaneously pushing\ndifferent images away from each other (contrasting negatives) in representation\nspace.\n\nOne such contrastive approach is [SimCLR](https://arxiv.org/abs/2002.05709),\nwhich essentially identifies the core components needed to optimize this\nobjective, and can achieve high performance by scaling this simple approach.\n\nAnother approach is [SimSiam](https://arxiv.org/abs/2011.10566)\n([Keras example](https://keras.io/examples/vision/simsiam/)),\nwhose main difference from\nSimCLR is that the former does not use any negatives in its loss. Therefore, it does not\nexplicitly prevent the trivial solution, and, instead, avoids it implicitly by\narchitecture design (asymmetric encoding paths using a predictor network and\nbatch normalization (BatchNorm) are applied in the final layers).\n\nFor further reading about SimCLR, check out\n[the official Google AI blog post](https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html),\nand for an overview of self-supervised learning across both vision and language\ncheck out\n[this blog post](https://ai.facebook.com/blog/self-supervised-learning-the-dark-matter-of-intelligence/).\n'
'\n## Setup\n'
import resource
(low, high) = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras import layers
'\n## Hyperparameter setup\n'
unlabeled_dataset_size = 100000
labeled_dataset_size = 5000
image_size = 96
image_channels = 3
num_epochs = 1
batch_size = 525
width = 128
temperature = 0.1
contrastive_augmentation = {'min_area': 0.25, 'brightness': 0.6, 'jitter': 0.2}
classification_augmentation = {'min_area': 0.75, 'brightness': 0.3, 'jitter': 0.1}
'\n## Dataset\n\nDuring training we will simultaneously load a large batch of unlabeled images along with a\nsmaller batch of labeled images.\n'

def prepare_dataset():
    if False:
        while True:
            i = 10
    steps_per_epoch = (unlabeled_dataset_size + labeled_dataset_size) // batch_size
    unlabeled_batch_size = unlabeled_dataset_size // steps_per_epoch
    labeled_batch_size = labeled_dataset_size // steps_per_epoch
    print(f'batch size is {unlabeled_batch_size} (unlabeled) + {labeled_batch_size} (labeled)')
    unlabeled_train_dataset = tfds.load('stl10', split='unlabelled', as_supervised=True, shuffle_files=False).shuffle(buffer_size=10 * unlabeled_batch_size).batch(unlabeled_batch_size)
    labeled_train_dataset = tfds.load('stl10', split='train', as_supervised=True, shuffle_files=False).shuffle(buffer_size=10 * labeled_batch_size).batch(labeled_batch_size)
    test_dataset = tfds.load('stl10', split='test', as_supervised=True).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    train_dataset = tf.data.Dataset.zip((unlabeled_train_dataset, labeled_train_dataset)).prefetch(buffer_size=tf.data.AUTOTUNE)
    return (train_dataset, labeled_train_dataset, test_dataset)
(train_dataset, labeled_train_dataset, test_dataset) = prepare_dataset()
'\n## Image augmentations\n\nThe two most important image augmentations for contrastive learning are the\nfollowing:\n\n- Cropping: forces the model to encode different parts of the same image\nsimilarly, we implement it with the\n[RandomTranslation](https://keras.io/api/layers/preprocessing_layers/image_preprocessing/random_translation/)\nand\n[RandomZoom](https://keras.io/api/layers/preprocessing_layers/image_preprocessing/random_zoom/)\nlayers\n- Color jitter: prevents a trivial color histogram-based solution to the task by\ndistorting color histograms. A principled way to implement that is by affine\ntransformations in color space.\n\nIn this example we use random horizontal flips as well. Stronger augmentations\nare applied for contrastive learning, along with weaker ones for supervised\nclassification to avoid overfitting on the few labeled examples.\n\nWe implement random color jitter as a custom preprocessing layer. Using\npreprocessing layers for data augmentation has the following two advantages:\n\n- The data augmentation will run on GPU in batches, so the training will not be\nbottlenecked by the data pipeline in environments with constrained CPU\nresources (such as a Colab Notebook, or a personal machine)\n- Deployment is easier as the data preprocessing pipeline is encapsulated in the\nmodel, and does not have to be reimplemented when deploying it\n'

class RandomColorAffine(layers.Layer):

    def __init__(self, brightness=0, jitter=0, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.brightness = brightness
        self.jitter = jitter

    def get_config(self):
        if False:
            print('Hello World!')
        config = super().get_config()
        config.update({'brightness': self.brightness, 'jitter': self.jitter})
        return config

    def call(self, images, training=True):
        if False:
            while True:
                i = 10
        if training:
            batch_size = tf.shape(images)[0]
            brightness_scales = 1 + tf.random.uniform((batch_size, 1, 1, 1), minval=-self.brightness, maxval=self.brightness)
            jitter_matrices = tf.random.uniform((batch_size, 1, 3, 3), minval=-self.jitter, maxval=self.jitter)
            color_transforms = tf.eye(3, batch_shape=[batch_size, 1]) * brightness_scales + jitter_matrices
            images = tf.clip_by_value(tf.matmul(images, color_transforms), 0, 1)
        return images

def get_augmenter(min_area, brightness, jitter):
    if False:
        print('Hello World!')
    zoom_factor = 1.0 - math.sqrt(min_area)
    return keras.Sequential([keras.Input(shape=(image_size, image_size, image_channels)), layers.Rescaling(1 / 255, dtype='uint8'), layers.RandomFlip('horizontal'), layers.RandomTranslation(zoom_factor / 2, zoom_factor / 2), layers.RandomZoom((-zoom_factor, 0.0), (-zoom_factor, 0.0)), RandomColorAffine(brightness, jitter)])

def visualize_augmentations(num_images):
    if False:
        return 10
    images = next(iter(train_dataset))[0][0][:num_images]
    augmented_images = zip(images, get_augmenter(**classification_augmentation)(images), get_augmenter(**contrastive_augmentation)(images), get_augmenter(**contrastive_augmentation)(images))
    row_titles = ['Original:', 'Weakly augmented:', 'Strongly augmented:', 'Strongly augmented:']
    plt.figure(figsize=(num_images * 2.2, 4 * 2.2), dpi=100)
    for (column, image_row) in enumerate(augmented_images):
        for (row, image) in enumerate(image_row):
            plt.subplot(4, num_images, row * num_images + column + 1)
            plt.imshow(image)
            if column == 0:
                plt.title(row_titles[row], loc='left')
            plt.axis('off')
    plt.tight_layout()
visualize_augmentations(num_images=8)
'\n## Encoder architecture\n'

def get_encoder():
    if False:
        return 10
    return keras.Sequential([keras.Input(shape=(image_size, image_size, image_channels)), layers.Conv2D(width, kernel_size=3, strides=2, activation='relu'), layers.Conv2D(width, kernel_size=3, strides=2, activation='relu'), layers.Conv2D(width, kernel_size=3, strides=2, activation='relu'), layers.Conv2D(width, kernel_size=3, strides=2, activation='relu'), layers.Flatten(), layers.Dense(width, activation='relu')], name='encoder')
'\n## Supervised baseline model\n\nA baseline supervised model is trained using random initialization.\n'
baseline_model = keras.Sequential([keras.Input(shape=(image_size, image_size, image_channels)), get_augmenter(**classification_augmentation), get_encoder(), layers.Dense(10)], name='baseline_model')
baseline_model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[keras.metrics.SparseCategoricalAccuracy(name='acc')])
baseline_history = baseline_model.fit(labeled_train_dataset, epochs=num_epochs, validation_data=test_dataset)
print('Maximal validation accuracy: {:.2f}%'.format(max(baseline_history.history['val_acc']) * 100))
'\n## Self-supervised model for contrastive pretraining\n\nWe pretrain an encoder on unlabeled images with a contrastive loss.\nA nonlinear projection head is attached to the top of the encoder, as it\nimproves the quality of representations of the encoder.\n\nWe use the InfoNCE/NT-Xent/N-pairs loss, which can be interpreted in the\nfollowing way:\n\n1. We treat each image in the batch as if it had its own class.\n2. Then, we have two examples (a pair of augmented views) for each "class".\n3. Each view\'s representation is compared to every possible pair\'s one (for both\n  augmented versions).\n4. We use the temperature-scaled cosine similarity of compared representations as\n  logits.\n5. Finally, we use categorical cross-entropy as the "classification" loss\n\nThe following two metrics are used for monitoring the pretraining performance:\n\n- [Contrastive accuracy (SimCLR Table 5)](https://arxiv.org/abs/2002.05709):\nSelf-supervised metric, the ratio of cases in which the representation of an\nimage is more similar to its differently augmented version\'s one, than to the\nrepresentation of any other image in the current batch. Self-supervised\nmetrics can be used for hyperparameter tuning even in the case when there are\nno labeled examples.\n- [Linear probing accuracy](https://arxiv.org/abs/1603.08511): Linear probing is\na popular metric to evaluate self-supervised classifiers. It is computed as\nthe accuracy of a logistic regression classifier trained on top of the\nencoder\'s features. In our case, this is done by training a single dense layer\non top of the frozen encoder. Note that contrary to traditional approach where\nthe classifier is trained after the pretraining phase, in this example we\ntrain it during pretraining. This might slightly decrease its accuracy, but\nthat way we can monitor its value during training, which helps with\nexperimentation and debugging.\n\nAnother widely used supervised metric is the\n[KNN accuracy](https://arxiv.org/abs/1805.01978), which is the accuracy of a KNN\nclassifier trained on top of the encoder\'s features, which is not implemented in\nthis example.\n'

class ContrastiveModel(keras.Model):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.temperature = temperature
        self.contrastive_augmenter = get_augmenter(**contrastive_augmentation)
        self.classification_augmenter = get_augmenter(**classification_augmentation)
        self.encoder = get_encoder()
        self.projection_head = keras.Sequential([keras.Input(shape=(width,)), layers.Dense(width, activation='relu'), layers.Dense(width)], name='projection_head')
        self.linear_probe = keras.Sequential([layers.Input(shape=(width,)), layers.Dense(10)], name='linear_probe')
        self.encoder.summary()
        self.projection_head.summary()
        self.linear_probe.summary()

    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        if False:
            while True:
                i = 10
        super().compile(**kwargs)
        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer
        self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.contrastive_loss_tracker = keras.metrics.Mean(name='c_loss')
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(name='c_acc')
        self.probe_loss_tracker = keras.metrics.Mean(name='p_loss')
        self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy(name='p_acc')

    @property
    def metrics(self):
        if False:
            i = 10
            return i + 15
        return [self.contrastive_loss_tracker, self.contrastive_accuracy, self.probe_loss_tracker, self.probe_accuracy]

    def contrastive_loss(self, projections_1, projections_2):
        if False:
            return 10
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(contrastive_labels, tf.transpose(similarities))
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(contrastive_labels, similarities, from_logits=True)
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(contrastive_labels, tf.transpose(similarities), from_logits=True)
        return (loss_1_2 + loss_2_1) / 2

    def train_step(self, data):
        if False:
            print('Hello World!')
        ((unlabeled_images, _), (labeled_images, labels)) = data
        images = tf.concat((unlabeled_images, labeled_images), axis=0)
        augmented_images_1 = self.contrastive_augmenter(images, training=True)
        augmented_images_2 = self.contrastive_augmenter(images, training=True)
        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1, training=True)
            features_2 = self.encoder(augmented_images_2, training=True)
            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        gradients = tape.gradient(contrastive_loss, self.encoder.trainable_weights + self.projection_head.trainable_weights)
        self.contrastive_optimizer.apply_gradients(zip(gradients, self.encoder.trainable_weights + self.projection_head.trainable_weights))
        self.contrastive_loss_tracker.update_state(contrastive_loss)
        preprocessed_images = self.classification_augmenter(labeled_images, training=True)
        with tf.GradientTape() as tape:
            features = self.encoder(preprocessed_images, training=False)
            class_logits = self.linear_probe(features, training=True)
            probe_loss = self.probe_loss(labels, class_logits)
        gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        self.probe_optimizer.apply_gradients(zip(gradients, self.linear_probe.trainable_weights))
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if False:
            return 10
        (labeled_images, labels) = data
        preprocessed_images = self.classification_augmenter(labeled_images, training=False)
        features = self.encoder(preprocessed_images, training=False)
        class_logits = self.linear_probe(features, training=False)
        probe_loss = self.probe_loss(labels, class_logits)
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)
        return {m.name: m.result() for m in self.metrics[2:]}
pretraining_model = ContrastiveModel()
pretraining_model.compile(contrastive_optimizer=keras.optimizers.Adam(), probe_optimizer=keras.optimizers.Adam())
pretraining_history = pretraining_model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
print('Maximal validation accuracy: {:.2f}%'.format(max(pretraining_history.history['val_p_acc']) * 100))
'\n## Supervised finetuning of the pretrained encoder\n\nWe then finetune the encoder on the labeled examples, by attaching\na single randomly initalized fully connected classification layer on its top.\n'
finetuning_model = keras.Sequential([layers.Input(shape=(image_size, image_size, image_channels)), get_augmenter(**classification_augmentation), pretraining_model.encoder, layers.Dense(10)], name='finetuning_model')
finetuning_model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[keras.metrics.SparseCategoricalAccuracy(name='acc')])
finetuning_history = finetuning_model.fit(labeled_train_dataset, epochs=num_epochs, validation_data=test_dataset)
print('Maximal validation accuracy: {:.2f}%'.format(max(finetuning_history.history['val_acc']) * 100))
'\n## Comparison against the baseline\n'

def plot_training_curves(pretraining_history, finetuning_history, baseline_history):
    if False:
        while True:
            i = 10
    for (metric_key, metric_name) in zip(['acc', 'loss'], ['accuracy', 'loss']):
        plt.figure(figsize=(8, 5), dpi=100)
        plt.plot(baseline_history.history[f'val_{metric_key}'], label='supervised baseline')
        plt.plot(pretraining_history.history[f'val_p_{metric_key}'], label='self-supervised pretraining')
        plt.plot(finetuning_history.history[f'val_{metric_key}'], label='supervised finetuning')
        plt.legend()
        plt.title(f'Classification {metric_name} during training')
        plt.xlabel('epochs')
        plt.ylabel(f'validation {metric_name}')
plot_training_curves(pretraining_history, finetuning_history, baseline_history)
'\nBy comparing the training curves, we can see that when using contrastive\npretraining, a higher validation accuracy can be reached, paired with a lower\nvalidation loss, which means that the pretrained network was able to generalize\nbetter when seeing only a small amount of labeled examples.\n'
'\n## Improving further\n\n### Architecture\n\nThe experiment in the original paper demonstrated that increasing the width and depth of the\nmodels improves performance at a higher rate than for supervised learning. Also,\nusing a [ResNet-50](https://keras.io/api/applications/resnet/#resnet50-function)\nencoder is quite standard in the literature. However keep in mind, that more\npowerful models will not only increase training time but will also require more\nmemory and will limit the maximal batch size you can use.\n\nIt has [been](https://arxiv.org/abs/1905.09272)\n[reported](https://arxiv.org/abs/1911.05722) that the usage of BatchNorm layers\ncould sometimes degrade performance, as it introduces an intra-batch dependency\nbetween samples, which is why I did not have used them in this example. In my\nexperiments however, using BatchNorm, especially in the projection head,\nimproves performance.\n\n### Hyperparameters\n\nThe hyperparameters used in this example have been tuned manually for this task and\narchitecture. Therefore, without changing them, only marginal gains can be expected\nfrom further hyperparameter tuning.\n\nHowever for a different task or model architecture these would need tuning, so\nhere are my notes on the most important ones:\n\n- **Batch size**: since the objective can be interpreted as a classification\nover a batch of images (loosely speaking), the batch size is actually a more\nimportant hyperparameter than usual. The higher, the better.\n- **Temperature**: the temperature defines the "softness" of the softmax\ndistribution that is used in the cross-entropy loss, and is an important\nhyperparameter. Lower values generally lead to a higher contrastive accuracy.\nA recent trick (in [ALIGN](https://arxiv.org/abs/2102.05918)) is to learn\nthe temperature\'s value as well (which can be done by defining it as a\ntf.Variable, and applying gradients on it). Even though this provides a good baseline\nvalue, in my experiments the learned temperature was somewhat lower\nthan optimal, as it is optimized with respect to the contrastive loss, which is not a\nperfect proxy for representation quality.\n- **Image augmentation strength**: during pretraining stronger augmentations\nincrease the difficulty of the task, however after a point too strong\naugmentations will degrade performance. During finetuning stronger\naugmentations reduce overfitting while in my experience too strong\naugmentations decrease the performance gains from pretraining. The whole data\naugmentation pipeline can be seen as an important hyperparameter of the\nalgorithm, implementations of other custom image augmentation layers in Keras\ncan be found in\n[this repository](https://github.com/beresandras/image-augmentation-layers-keras).\n- **Learning rate schedule**: a constant schedule is used here, but it is\nquite common in the literature to use a\n[cosine decay schedule](https://www.tensorflow.org/api_docs/python/tf/keras/experimental/CosineDecay),\nwhich can further improve performance.\n- **Optimizer**: Adam is used in this example, as it provides good performance\nwith default parameters. SGD with momentum requires more tuning, however it\ncould slightly increase performance.\n'
'\n## Related works\n\nOther instance-level (image-level) contrastive learning methods:\n\n- [MoCo](https://arxiv.org/abs/1911.05722)\n([v2](https://arxiv.org/abs/2003.04297),\n[v3](https://arxiv.org/abs/2104.02057)): uses a momentum-encoder as well,\nwhose weights are an exponential moving average of the target encoder\n- [SwAV](https://arxiv.org/abs/2006.09882): uses clustering instead of pairwise\ncomparison\n- [BarlowTwins](https://arxiv.org/abs/2103.03230): uses a cross\ncorrelation-based objective instead of pairwise comparison\n\nKeras implementations of **MoCo** and **BarlowTwins** can be found in\n[this repository](https://github.com/beresandras/contrastive-classification-keras),\nwhich includes a Colab notebook.\n\nThere is also a new line of works, which optimize a similar objective, but\nwithout the use of any negatives:\n\n- [BYOL](https://arxiv.org/abs/2006.07733): momentum-encoder + no negatives\n- [SimSiam](https://arxiv.org/abs/2011.10566)\n([Keras example](https://keras.io/examples/vision/simsiam/)):\nno momentum-encoder + no negatives\n\nIn my experience, these methods are more brittle (they can collapse to a constant\nrepresentation, I could not get them to work using this encoder architecture).\nEven though they are generally more dependent on the\n[model](https://generallyintelligent.ai/understanding-self-supervised-contrastive-learning.html)\n[architecture](https://arxiv.org/abs/2010.10241), they can improve\nperformance at smaller batch sizes.\n\nYou can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/semi-supervised-classification-simclr)\nand try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/semi-supervised-classification).\n'