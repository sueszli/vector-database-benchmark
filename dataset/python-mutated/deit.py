"""
Title: Distilling Vision Transformers
Author: [Sayak Paul](https://twitter.com/RisingSayak)
Date created: 2022/04/05
Last modified: 2023/09/16
Description: Distillation of Vision Transformers through attention.
Accelerator: GPU
"""
"\n## Introduction\n\nIn the original *Vision Transformers* (ViT) paper\n([Dosovitskiy et al.](https://arxiv.org/abs/2010.11929)),\nthe authors concluded that to perform on par with Convolutional Neural Networks (CNNs),\nViTs need to be pre-trained on larger datasets. The larger the better. This is mainly\ndue to the lack of inductive biases in the ViT architecture -- unlike CNNs,\nthey don't have layers that exploit locality. In a follow-up paper\n([Steiner et al.](https://arxiv.org/abs/2106.10270)),\nthe authors show that it is possible to substantially improve the performance of ViTs\nwith stronger regularization and longer training.\n\nMany groups have proposed different ways to deal with the problem\nof data-intensiveness of ViT training.\nOne such way was shown in the *Data-efficient image Transformers*,\n(DeiT) paper ([Touvron et al.](https://arxiv.org/abs/2012.12877)). The\nauthors introduced a distillation technique that is specific to transformer-based vision\nmodels. DeiT is among the first works to show that it's possible to train ViTs well\nwithout using larger datasets.\n\nIn this example, we implement the distillation recipe proposed in DeiT. This\nrequires us to slightly tweak the original ViT architecture and write a custom training\nloop to implement the distillation recipe.\n\nTo comfortably navigate through this example, you'll be expected to know how a ViT and\nknowledge distillation work. The following are good resources in case you needed a\nrefresher:\n\n* [ViT on keras.io](https://keras.io/examples/vision/image_classification_with_vision_transformer)\n* [Knowledge distillation on keras.io](https://keras.io/examples/vision/knowledge_distillation/)\n"
'\n## Imports\n'
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras import layers
tfds.disable_progress_bar()
keras.utils.set_random_seed(42)
'\n## Constants\n'
MODEL_TYPE = 'deit_distilled_tiny_patch16_224'
RESOLUTION = 224
PATCH_SIZE = 16
NUM_PATCHES = (RESOLUTION // PATCH_SIZE) ** 2
LAYER_NORM_EPS = 1e-06
PROJECTION_DIM = 192
NUM_HEADS = 3
NUM_LAYERS = 12
MLP_UNITS = [PROJECTION_DIM * 4, PROJECTION_DIM]
DROPOUT_RATE = 0.0
DROP_PATH_RATE = 0.1
NUM_EPOCHS = 20
BASE_LR = 0.0005
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 256
AUTO = tf.data.AUTOTUNE
NUM_CLASSES = 5
"\nYou probably noticed that `DROPOUT_RATE` has been set 0.0. Dropout has been used\nin the implementation to keep it complete. For smaller models (like the one used in\nthis example), you don't need it, but for bigger models, using dropout helps.\n"
"\n## Load the `tf_flowers` dataset and prepare preprocessing utilities\n\nThe authors use an array of different augmentation techniques, including MixUp\n([Zhang et al.](https://arxiv.org/abs/1710.09412)),\nRandAugment ([Cubuk et al.](https://arxiv.org/abs/1909.13719)),\nand so on. However, to keep the example simple to work through, we'll discard them.\n"

def preprocess_dataset(is_training=True):
    if False:
        return 10

    def fn(image, label):
        if False:
            for i in range(10):
                print('nop')
        if is_training:
            image = tf.image.resize(image, (RESOLUTION + 20, RESOLUTION + 20))
            image = tf.image.random_crop(image, (RESOLUTION, RESOLUTION, 3))
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize(image, (RESOLUTION, RESOLUTION))
        label = tf.one_hot(label, depth=NUM_CLASSES)
        return (image, label)
    return fn

def prepare_dataset(dataset, is_training=True):
    if False:
        print('Hello World!')
    if is_training:
        dataset = dataset.shuffle(BATCH_SIZE * 10)
    dataset = dataset.map(preprocess_dataset(is_training), num_parallel_calls=AUTO)
    return dataset.batch(BATCH_SIZE).prefetch(AUTO)
(train_dataset, val_dataset) = tfds.load('tf_flowers', split=['train[:90%]', 'train[90%:]'], as_supervised=True)
num_train = train_dataset.cardinality()
num_val = val_dataset.cardinality()
print(f'Number of training examples: {num_train}')
print(f'Number of validation examples: {num_val}')
train_dataset = prepare_dataset(train_dataset, is_training=True)
val_dataset = prepare_dataset(val_dataset, is_training=False)
"\n## Implementing the DeiT variants of ViT\n\nSince DeiT is an extension of ViT it'd make sense to first implement ViT and then extend\nit to support DeiT's components.\n\nFirst, we'll implement a layer for Stochastic Depth\n([Huang et al.](https://arxiv.org/abs/1603.09382))\nwhich is used in DeiT for regularization.\n"

class StochasticDepth(layers.Layer):

    def __init__(self, drop_prop, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.drop_prob = drop_prop

    def call(self, x, training=True):
        if False:
            return 10
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return x / keep_prob * random_tensor
        return x
"\nNow, we'll implement the MLP and Transformer blocks.\n"

def mlp(x, dropout_rate: float, hidden_units):
    if False:
        print('Hello World!')
    'FFN for a Transformer block.'
    for (idx, units) in enumerate(hidden_units):
        x = layers.Dense(units, activation=tf.nn.gelu if idx == 0 else None)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def transformer(drop_prob: float, name: str) -> keras.Model:
    if False:
        print('Hello World!')
    'Transformer block with pre-norm.'
    num_patches = NUM_PATCHES + 2 if 'distilled' in MODEL_TYPE else NUM_PATCHES + 1
    encoded_patches = layers.Input((num_patches, PROJECTION_DIM))
    x1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(encoded_patches)
    attention_output = layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=DROPOUT_RATE)(x1, x1)
    attention_output = StochasticDepth(drop_prob)(attention_output) if drop_prob else attention_output
    x2 = layers.Add()([attention_output, encoded_patches])
    x3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x2)
    x4 = mlp(x3, hidden_units=MLP_UNITS, dropout_rate=DROPOUT_RATE)
    x4 = StochasticDepth(drop_prob)(x4) if drop_prob else x4
    outputs = layers.Add()([x2, x4])
    return keras.Model(encoded_patches, outputs, name=name)
"\nWe'll now implement a `ViTClassifier` class building on top of the components we just\ndeveloped. Here we'll be following the original pooling strategy used in the ViT paper --\nuse a class token and use the feature representations corresponding to it for\nclassification.\n"

class ViTClassifier(keras.Model):
    """Vision Transformer base class."""

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.projection = keras.Sequential([layers.Conv2D(filters=PROJECTION_DIM, kernel_size=(PATCH_SIZE, PATCH_SIZE), strides=(PATCH_SIZE, PATCH_SIZE), padding='VALID', name='conv_projection'), layers.Reshape(target_shape=(NUM_PATCHES, PROJECTION_DIM), name='flatten_projection')], name='projection')
        init_shape = (1, NUM_PATCHES + 1, PROJECTION_DIM)
        self.positional_embedding = tf.Variable(tf.zeros(init_shape), name='pos_embedding')
        dpr = [x for x in tf.linspace(0.0, DROP_PATH_RATE, NUM_LAYERS)]
        self.transformer_blocks = [transformer(drop_prob=dpr[i], name=f'transformer_block_{i}') for i in range(NUM_LAYERS)]
        initial_value = tf.zeros((1, 1, PROJECTION_DIM))
        self.cls_token = tf.Variable(initial_value=initial_value, trainable=True, name='cls')
        self.dropout = layers.Dropout(DROPOUT_RATE)
        self.layer_norm = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)
        self.head = layers.Dense(NUM_CLASSES, name='classification_head')

    def call(self, inputs):
        if False:
            return 10
        n = tf.shape(inputs)[0]
        projected_patches = self.projection(inputs)
        cls_token = tf.tile(self.cls_token, (n, 1, 1))
        cls_token = tf.cast(cls_token, projected_patches.dtype)
        projected_patches = tf.concat([cls_token, projected_patches], axis=1)
        encoded_patches = self.positional_embedding + projected_patches
        encoded_patches = self.dropout(encoded_patches)
        for transformer_module in self.transformer_blocks:
            encoded_patches = transformer_module(encoded_patches)
        representation = self.layer_norm(encoded_patches)
        encoded_patches = representation[:, 0]
        output = self.head(encoded_patches)
        return output
"\nThis class can be used standalone as ViT and is end-to-end trainable. Just remove the\n`distilled` phrase in `MODEL_TYPE` and it should work with `vit_tiny = ViTClassifier()`.\nLet's now extend it to DeiT. The following figure presents the schematic of DeiT (taken\nfrom the DeiT paper):\n\n![](https://i.imgur.com/5lmg2Xs.png)\n\nApart from the class token, DeiT has another token for distillation. During distillation,\nthe logits corresponding to the class token are compared to the true labels, and the\nlogits corresponding to the distillation token are compared to the teacher's predictions.\n"

class ViTDistilled(ViTClassifier):

    def __init__(self, regular_training=False, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.num_tokens = 2
        self.regular_training = regular_training
        init_value = tf.zeros((1, 1, PROJECTION_DIM))
        self.dist_token = tf.Variable(init_value, name='dist_token')
        self.positional_embedding = tf.Variable(tf.zeros((1, NUM_PATCHES + self.num_tokens, PROJECTION_DIM)), name='pos_embedding')
        self.head = layers.Dense(NUM_CLASSES, name='classification_head')
        self.head_dist = layers.Dense(NUM_CLASSES, name='distillation_head')

    def call(self, inputs, training=False):
        if False:
            for i in range(10):
                print('nop')
        n = tf.shape(inputs)[0]
        projected_patches = self.projection(inputs)
        cls_token = tf.tile(self.cls_token, (n, 1, 1))
        dist_token = tf.tile(self.dist_token, (n, 1, 1))
        cls_token = tf.cast(cls_token, projected_patches.dtype)
        dist_token = tf.cast(dist_token, projected_patches.dtype)
        projected_patches = tf.concat([cls_token, dist_token, projected_patches], axis=1)
        encoded_patches = self.positional_embedding + projected_patches
        encoded_patches = self.dropout(encoded_patches)
        for transformer_module in self.transformer_blocks:
            encoded_patches = transformer_module(encoded_patches)
        representation = self.layer_norm(encoded_patches)
        (x, x_dist) = (self.head(representation[:, 0]), self.head_dist(representation[:, 1]))
        if not training or self.regular_training:
            return (x + x_dist) / 2
        elif training:
            return (x, x_dist)
"\nLet's verify if the `ViTDistilled` class can be initialized and called as expected.\n"
deit_tiny_distilled = ViTDistilled()
dummy_inputs = tf.ones((2, 224, 224, 3))
outputs = deit_tiny_distilled(dummy_inputs, training=False)
print(f'output_shape: {outputs.shape}')
'\n## Implementing the trainer\n\nUnlike what happens in standard knowledge distillation\n([Hinton et al.](https://arxiv.org/abs/1503.02531)),\nwhere a temperature-scaled softmax is used as well as KL divergence,\nDeiT authors use the following loss function:\n\n![](https://i.imgur.com/bXdxsBq.png)\n\n\nHere,\n\n* CE is cross-entropy\n* `psi` is the softmax function\n* Z_s denotes student predictions\n* y denotes true labels\n* y_t denotes teacher predictions\n'

class DeiT(keras.Model):

    def __init__(self, student, teacher, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher
        self.student_loss_tracker = keras.metrics.Mean(name='student_loss')
        self.distillation_loss_tracker = keras.metrics.Mean(name='distillation_loss')
        self.accuracy = keras.metrics.CategoricalAccuracy(name='accuracy')

    @property
    def metrics(self):
        if False:
            print('Hello World!')
        return [self.accuracy, self.student_loss_tracker, self.distillation_loss_tracker]

    def compile(self, optimizer, student_loss_fn, distillation_loss_fn, run_eagerly=False, jit_compile=False):
        if False:
            print('Hello World!')
        super().compile(optimizer=optimizer, run_eagerly=run_eagerly, jit_compile=jit_compile)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn

    def train_step(self, data):
        if False:
            i = 10
            return i + 15
        (x, y) = data
        teacher_predictions = self.teacher(x)['dense']
        teacher_predictions = tf.nn.softmax(teacher_predictions, axis=-1)
        with tf.GradientTape() as tape:
            (cls_predictions, dist_predictions) = self.student(x / 255.0, training=True)
            student_loss = self.student_loss_fn(y, cls_predictions)
            distillation_loss = self.distillation_loss_fn(teacher_predictions, dist_predictions)
            loss = (student_loss + distillation_loss) / 2
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        student_predictions = (cls_predictions + dist_predictions) / 2
        self.student_loss_tracker.update_state(student_loss)
        self.distillation_loss_tracker.update_state(distillation_loss)
        self.accuracy.update_state(y, student_predictions)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if False:
            return 10
        (x, y) = data
        y_prediction = self.student(x / 255.0)
        student_loss = self.student_loss_fn(y, y_prediction)
        self.student_loss_tracker.update_state(student_loss)
        self.accuracy.update_state(y, y_prediction)
        results = {m.name: m.result() for m in self.metrics}
        return results

    def call(self, inputs):
        if False:
            i = 10
            return i + 15
        return self.student(inputs / 255.0)
'\n## Load the teacher model\n\nThis model is based on the BiT family of ResNets\n([Kolesnikov et al.](https://arxiv.org/abs/1912.11370))\nfine-tuned on the `tf_flowers` dataset. You can refer to\n[this notebook](https://github.com/sayakpaul/deit-tf/blob/main/notebooks/bit-teacher.ipynb)\nto know how the training was performed. The teacher model has about 212 Million parameters\nwhich is about **40x more** than the student.\n'
'shell\nwget -q https://github.com/sayakpaul/deit-tf/releases/download/v0.1.0/bit_teacher_flowers.zip\nunzip -q bit_teacher_flowers.zip\n'
bit_teacher_flowers = keras.layers.TFSMLayer(filepath='bit_teacher_flowers', call_endpoint='serving_default')
'\n## Training through distillation\n'
deit_tiny = ViTDistilled()
deit_distiller = DeiT(student=deit_tiny, teacher=bit_teacher_flowers)
lr_scaled = BASE_LR / 512 * BATCH_SIZE
deit_distiller.compile(optimizer=keras.optimizers.AdamW(weight_decay=WEIGHT_DECAY, learning_rate=lr_scaled), student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1), distillation_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=True))
_ = deit_distiller.fit(train_dataset, validation_data=val_dataset, epochs=NUM_EPOCHS)
'\nIf we had trained the same model (the `ViTClassifier`) from scratch with the exact same\nhyperparameters, the model would have scored about 59% accuracy. You can adapt the following code\nto reproduce this result:\n\n```\nvit_tiny = ViTClassifier()\n\ninputs = keras.Input((RESOLUTION, RESOLUTION, 3))\nx = keras.layers.Rescaling(scale=1./255)(inputs)\noutputs = deit_tiny(x)\nmodel = keras.Model(inputs, outputs)\n\nmodel.compile(...)\nmodel.fit(...)\n```\n'
"\n## Notes\n\n* Through the use of distillation, we're effectively transferring the inductive biases of\na CNN-based teacher model.\n* Interestingly enough, this distillation strategy works better with a CNN as the teacher\nmodel rather than a Transformer as shown in the paper.\n* The use of regularization to train DeiT models is very important.\n* ViT models are initialized with a combination of different initializers including\ntruncated normal, random normal, Glorot uniform, etc. If you're looking for\nend-to-end reproduction of the original results, don't forget to initialize the ViTs well.\n* If you want to explore the pre-trained DeiT models in TensorFlow and Keras with code\nfor fine-tuning, [check out these models on TF-Hub](https://tfhub.dev/sayakpaul/collections/deit/1).\n\n## Acknowledgements\n\n* Ross Wightman for keeping\n[`timm`](https://github.com/rwightman/pytorch-image-models)\nupdated with readable implementations. I referred to the implementations of ViT and DeiT\na lot during implementing them in TensorFlow.\n* [Aritra Roy Gosthipaty](https://github.com/ariG23498)\nwho implemented some portions of the `ViTClassifier` in another project.\n* [Google Developers Experts](https://developers.google.com/programs/experts/)\nprogram for supporting me with GCP credits which were used to run experiments for this\nexample.\n"