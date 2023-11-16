"""
Title: A Vision Transformer without Attention
Author: [Aritra Roy Gosthipaty](https://twitter.com/ariG23498), [Ritwik Raha](https://twitter.com/ritwik_raha)
Converted to Keras 3: [Muhammad Anas Raza](https://anasrz.com)
Date created: 2022/02/24
Last modified: 2023/07/15
Description: A minimal implementation of ShiftViT.
Accelerator: GPU
"""
"\n## Introduction\n\n[Vision Transformers](https://arxiv.org/abs/2010.11929) (ViTs) have sparked a wave of\nresearch at the intersection of Transformers and Computer Vision (CV).\n\nViTs can simultaneously model long- and short-range dependencies, thanks to\nthe Multi-Head Self-Attention mechanism in the Transformer block. Many researchers believe\nthat the success of ViTs are purely due to the attention layer, and they seldom\nthink about other parts of the ViT model.\n\nIn the academic paper\n[When Shift Operation Meets Vision Transformer: An Extremely Simple Alternative to Attention Mechanism](https://arxiv.org/abs/2201.10801)\nthe authors propose to demystify the success of ViTs with the introduction of a **NO\nPARAMETER** operation in place of the attention operation. They swap the attention\noperation with a shifting operation.\n\nIn this example, we minimally implement the paper with close alignement to the author's\n[official implementation](https://github.com/microsoft/SPACH/blob/main/models/shiftvit.py).\n"
'\n## Setup and imports\n'
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
SEED = 42
keras.utils.set_random_seed(SEED)
'\n## Hyperparameters\n\nThese are the hyperparameters that we have chosen for the experiment.\nPlease feel free to tune them.\n'

class Config(object):
    batch_size = 256
    buffer_size = batch_size * 2
    input_shape = (32, 32, 3)
    num_classes = 10
    image_size = 48
    patch_size = 4
    projected_dim = 96
    num_shift_blocks_per_stages = [2, 4, 8, 2]
    epsilon = 1e-05
    stochastic_depth_rate = 0.2
    mlp_dropout_rate = 0.2
    num_div = 12
    shift_pixel = 1
    mlp_expand_ratio = 2
    lr_start = 1e-05
    lr_max = 0.001
    weight_decay = 0.0001
    epochs = 100
config = Config()
'\n## Load the CIFAR-10 dataset\n\nWe use the CIFAR-10 dataset for our experiments.\n'
((x_train, y_train), (x_test, y_test)) = keras.datasets.cifar10.load_data()
((x_train, y_train), (x_val, y_val)) = ((x_train[:40000], y_train[:40000]), (x_train[40000:], y_train[40000:]))
print(f'Training samples: {len(x_train)}')
print(f'Validation samples: {len(x_val)}')
print(f'Testing samples: {len(x_test)}')
AUTO = tf.data.AUTOTUNE
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(config.buffer_size).batch(config.batch_size).prefetch(AUTO)
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_ds = val_ds.batch(config.batch_size).prefetch(AUTO)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.batch(config.batch_size).prefetch(AUTO)
'\n## Data Augmentation\n\nThe augmentation pipeline consists of:\n\n- Rescaling\n- Resizing\n- Random cropping\n- Random horizontal flipping\n\n_Note_: The image data augmentation layers do not apply\ndata transformations at inference time. This means that\nwhen these layers are called with `training=False` they\nbehave differently. Refer to the\n[documentation](https://keras.io/api/layers/preprocessing_layers/image_augmentation/)\nfor more details.\n'

def get_augmentation_model():
    if False:
        for i in range(10):
            print('nop')
    'Build the data augmentation model.'
    data_augmentation = keras.Sequential([layers.Resizing(config.input_shape[0] + 20, config.input_shape[0] + 20), layers.RandomCrop(config.image_size, config.image_size), layers.RandomFlip('horizontal'), layers.Rescaling(1 / 255.0)])
    return data_augmentation
'\n## The ShiftViT architecture\n\nIn this section, we build the architecture proposed in\n[the ShiftViT paper](https://arxiv.org/abs/2201.10801).\n\n| ![ShiftViT Architecture](https://i.imgur.com/CHU40HX.png) |\n| :--: |\n| Figure 1: The entire architecutre of ShiftViT.\n[Source](https://arxiv.org/abs/2201.10801) |\n\nThe architecture as shown in Fig. 1, is inspired by\n[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030).\nHere the authors propose a modular architecture with 4 stages. Each stage works on its\nown spatial size, creating a hierarchical architecture.\n\nAn input image of size `HxWx3` is split into non-overlapping patches of size `4x4`.\nThis is done via the patchify layer which results in individual tokens of feature size `48`\n(`4x4x3`). Each stage comprises two parts:\n\n1. Embedding Generation\n2. Stacked Shift Blocks\n\nWe discuss the stages and the modules in detail in what follows.\n\n_Note_: Compared to the [official implementation](https://github.com/microsoft/SPACH/blob/main/models/shiftvit.py)\nwe restructure some key components to better fit the Keras API.\n'
'\n### The ShiftViT Block\n\n| ![ShiftViT block](https://i.imgur.com/IDe35vo.gif) |\n| :--: |\n| Figure 2: From the Model to a Shift Block. |\n\nEach stage in the ShiftViT architecture comprises of a Shift Block as shown in Fig 2.\n\n| ![Shift Vit Block](https://i.imgur.com/0q13pLu.png) |\n| :--: |\n| Figure 3: The Shift ViT Block. [Source](https://arxiv.org/abs/2201.10801) |\n\nThe Shift Block as shown in Fig. 3, comprises of the following:\n\n1. Shift Operation\n2. Linear Normalization\n3. MLP Layer\n'
'\n#### The MLP block\n\nThe MLP block is intended to be a stack of densely-connected layers.s\n'

class MLP(layers.Layer):
    """Get the MLP layer for each shift block.

    Args:
        mlp_expand_ratio (int): The ratio with which the first feature map is expanded.
        mlp_dropout_rate (float): The rate for dropout.
    """

    def __init__(self, mlp_expand_ratio, mlp_dropout_rate, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.mlp_expand_ratio = mlp_expand_ratio
        self.mlp_dropout_rate = mlp_dropout_rate

    def build(self, input_shape):
        if False:
            i = 10
            return i + 15
        input_channels = input_shape[-1]
        initial_filters = int(self.mlp_expand_ratio * input_channels)
        self.mlp = keras.Sequential([layers.Dense(units=initial_filters, activation=tf.nn.gelu), layers.Dropout(rate=self.mlp_dropout_rate), layers.Dense(units=input_channels), layers.Dropout(rate=self.mlp_dropout_rate)])

    def call(self, x):
        if False:
            i = 10
            return i + 15
        x = self.mlp(x)
        return x
'\n#### The DropPath layer\n\nStochastic depth is a regularization technique that randomly drops a set of\nlayers. During inference, the layers are kept as they are. It is very\nsimilar to Dropout, but it operates on a block of layers rather\nthan on individual nodes present inside a layer.\n'

class DropPath(layers.Layer):
    """Drop Path also known as the Stochastic Depth layer.

    Refernece:
        - https://keras.io/examples/vision/cct/#stochastic-depth-for-regularization
        - github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path_prob, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.drop_path_prob = drop_path_prob

    def call(self, x, training=False):
        if False:
            print('Hello World!')
        if training:
            keep_prob = 1 - self.drop_path_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return x / keep_prob * random_tensor
        return x
'\n#### Block\n\nThe most important operation in this paper is the **shift opperation**. In this section,\nwe describe the shift operation and compare it with its original implementation provided\nby the authors.\n\nA generic feature map is assumed to have the shape `[N, H, W, C]`. Here we choose a\n`num_div` parameter that decides the division size of the channels. The first 4 divisions\nare shifted (1 pixel) in the left, right, up, and down direction. The remaining splits\nare kept as is. After partial shifting the shifted channels are padded and the overflown\npixels are chopped off. This completes the partial shifting operation.\n\nIn the original implementation, the code is approximately:\n\n```python\nout[:, g * 0:g * 1, :, :-1] = x[:, g * 0:g * 1, :, 1:]  # shift left\nout[:, g * 1:g * 2, :, 1:] = x[:, g * 1:g * 2, :, :-1]  # shift right\nout[:, g * 2:g * 3, :-1, :] = x[:, g * 2:g * 3, 1:, :]  # shift up\nout[:, g * 3:g * 4, 1:, :] = x[:, g * 3:g * 4, :-1, :]  # shift down\n\nout[:, g * 4:, :, :] = x[:, g * 4:, :, :]  # no shift\n```\n\nIn TensorFlow it would be infeasible for us to assign shifted channels to a tensor in the\nmiddle of the training process. This is why we have resorted to the following procedure:\n\n1. Split the channels with the `num_div` parameter.\n2. Select each of the first four spilts and shift and pad them in the respective\ndirections.\n3. After shifting and padding, we concatenate the channel back.\n\n| ![Manim rendered animation for shift operation](https://i.imgur.com/PReeULP.gif) |\n| :--: |\n| Figure 4: The TensorFlow style shifting |\n\nThe entire procedure is explained in the Fig. 4.\n'

class ShiftViTBlock(layers.Layer):
    """A unit ShiftViT Block

    Args:
        shift_pixel (int): The number of pixels to shift. Defaults to `1`.
        mlp_expand_ratio (int): The ratio with which MLP features are
            expanded. Defaults to `2`.
        mlp_dropout_rate (float): The dropout rate used in MLP.
        num_div (int): The number of divisions of the feature map's channel.
            Totally, 4/num_div of channels will be shifted. Defaults to 12.
        epsilon (float): Epsilon constant.
        drop_path_prob (float): The drop probability for drop path.
    """

    def __init__(self, epsilon, drop_path_prob, mlp_dropout_rate, num_div=12, shift_pixel=1, mlp_expand_ratio=2, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.shift_pixel = shift_pixel
        self.mlp_expand_ratio = mlp_expand_ratio
        self.mlp_dropout_rate = mlp_dropout_rate
        self.num_div = num_div
        self.epsilon = epsilon
        self.drop_path_prob = drop_path_prob

    def build(self, input_shape):
        if False:
            i = 10
            return i + 15
        self.H = input_shape[1]
        self.W = input_shape[2]
        self.C = input_shape[3]
        self.layer_norm = layers.LayerNormalization(epsilon=self.epsilon)
        self.drop_path = DropPath(drop_path_prob=self.drop_path_prob) if self.drop_path_prob > 0.0 else layers.Activation('linear')
        self.mlp = MLP(mlp_expand_ratio=self.mlp_expand_ratio, mlp_dropout_rate=self.mlp_dropout_rate)

    def get_shift_pad(self, x, mode):
        if False:
            for i in range(10):
                print('nop')
        'Shifts the channels according to the mode chosen.'
        if mode == 'left':
            offset_height = 0
            offset_width = 0
            target_height = 0
            target_width = self.shift_pixel
        elif mode == 'right':
            offset_height = 0
            offset_width = self.shift_pixel
            target_height = 0
            target_width = self.shift_pixel
        elif mode == 'up':
            offset_height = 0
            offset_width = 0
            target_height = self.shift_pixel
            target_width = 0
        else:
            offset_height = self.shift_pixel
            offset_width = 0
            target_height = self.shift_pixel
            target_width = 0
        crop = tf.image.crop_to_bounding_box(x, offset_height=offset_height, offset_width=offset_width, target_height=self.H - target_height, target_width=self.W - target_width)
        shift_pad = tf.image.pad_to_bounding_box(crop, offset_height=offset_height, offset_width=offset_width, target_height=self.H, target_width=self.W)
        return shift_pad

    def call(self, x, training=False):
        if False:
            print('Hello World!')
        x_splits = tf.split(x, num_or_size_splits=self.C // self.num_div, axis=-1)
        x_splits[0] = self.get_shift_pad(x_splits[0], mode='left')
        x_splits[1] = self.get_shift_pad(x_splits[1], mode='right')
        x_splits[2] = self.get_shift_pad(x_splits[2], mode='up')
        x_splits[3] = self.get_shift_pad(x_splits[3], mode='down')
        x = tf.concat(x_splits, axis=-1)
        shortcut = x
        x = shortcut + self.drop_path(self.mlp(self.layer_norm(x)), training=training)
        return x
'\n### The ShiftViT blocks\n\n| ![Shift Blokcs](https://i.imgur.com/FKy5NnD.png) |\n| :--: |\n| Figure 5: Shift Blocks in the architecture. [Source](https://arxiv.org/abs/2201.10801) |\n\nEach stage of the architecture has shift blocks as shown in Fig.5. Each of these blocks\ncontain a variable number of stacked ShiftViT block (as built in the earlier section).\n\nShift blocks are followed by a PatchMerging layer that scales down feature inputs. The\nPatchMerging layer helps in the pyramidal structure of the model.\n'
'\n#### The PatchMerging layer\n\nThis layer merges the two adjacent tokens. This layer helps in scaling the features down\nspatially and increasing the features up channel wise. We use a Conv2D layer to merge the\npatches.\n'

class PatchMerging(layers.Layer):
    """The Patch Merging layer.

    Args:
        epsilon (float): The epsilon constant.
    """

    def __init__(self, epsilon, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        if False:
            return 10
        filters = 2 * input_shape[-1]
        self.reduction = layers.Conv2D(filters=filters, kernel_size=2, strides=2, padding='same', use_bias=False)
        self.layer_norm = layers.LayerNormalization(epsilon=self.epsilon)

    def call(self, x):
        if False:
            return 10
        x = self.layer_norm(x)
        x = self.reduction(x)
        return x
'\n#### Stacked Shift Blocks\n\nEach stage will have a variable number of stacked ShiftViT Blocks, as suggested in\nthe paper. This is a generic layer that will contain the stacked shift vit blocks\nwith the patch merging layer as well. Combining the two operations (shift ViT\nblock and patch merging) is a design choice we picked for better code reusability.\n'

class StackedShiftBlocks(layers.Layer):
    """The layer containing stacked ShiftViTBlocks.

    Args:
        epsilon (float): The epsilon constant.
        mlp_dropout_rate (float): The dropout rate used in the MLP block.
        num_shift_blocks (int): The number of shift vit blocks for this stage.
        stochastic_depth_rate (float): The maximum drop path rate chosen.
        is_merge (boolean): A flag that determines the use of the Patch Merge
            layer after the shift vit blocks.
        num_div (int): The division of channels of the feature map. Defaults to `12`.
        shift_pixel (int): The number of pixels to shift. Defaults to `1`.
        mlp_expand_ratio (int): The ratio with which the initial dense layer of
            the MLP is expanded Defaults to `2`.
    """

    def __init__(self, epsilon, mlp_dropout_rate, num_shift_blocks, stochastic_depth_rate, is_merge, num_div=12, shift_pixel=1, mlp_expand_ratio=2, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.mlp_dropout_rate = mlp_dropout_rate
        self.num_shift_blocks = num_shift_blocks
        self.stochastic_depth_rate = stochastic_depth_rate
        self.is_merge = is_merge
        self.num_div = num_div
        self.shift_pixel = shift_pixel
        self.mlp_expand_ratio = mlp_expand_ratio

    def build(self, input_shapes):
        if False:
            return 10
        dpr = [x for x in np.linspace(start=0, stop=self.stochastic_depth_rate, num=self.num_shift_blocks)]
        self.shift_blocks = list()
        for num in range(self.num_shift_blocks):
            self.shift_blocks.append(ShiftViTBlock(num_div=self.num_div, epsilon=self.epsilon, drop_path_prob=dpr[num], mlp_dropout_rate=self.mlp_dropout_rate, shift_pixel=self.shift_pixel, mlp_expand_ratio=self.mlp_expand_ratio))
        if self.is_merge:
            self.patch_merge = PatchMerging(epsilon=self.epsilon)

    def call(self, x, training=False):
        if False:
            for i in range(10):
                print('nop')
        for shift_block in self.shift_blocks:
            x = shift_block(x, training=training)
        if self.is_merge:
            x = self.patch_merge(x)
        return x
'\n## The ShiftViT model\n\nBuild the ShiftViT custom model.\n'

class ShiftViTModel(keras.Model):
    """The ShiftViT Model.

    Args:
        data_augmentation (keras.Model): A data augmentation model.
        projected_dim (int): The dimension to which the patches of the image are
            projected.
        patch_size (int): The patch size of the images.
        num_shift_blocks_per_stages (list[int]): A list of all the number of shit
            blocks per stage.
        epsilon (float): The epsilon constant.
        mlp_dropout_rate (float): The dropout rate used in the MLP block.
        stochastic_depth_rate (float): The maximum drop rate probability.
        num_div (int): The number of divisions of the channesl of the feature
            map. Defaults to `12`.
        shift_pixel (int): The number of pixel to shift. Defaults to `1`.
        mlp_expand_ratio (int): The ratio with which the initial mlp dense layer
            is expanded to. Defaults to `2`.
    """

    def __init__(self, data_augmentation, projected_dim, patch_size, num_shift_blocks_per_stages, epsilon, mlp_dropout_rate, stochastic_depth_rate, num_div=12, shift_pixel=1, mlp_expand_ratio=2, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.data_augmentation = data_augmentation
        self.patch_projection = layers.Conv2D(filters=projected_dim, kernel_size=patch_size, strides=patch_size, padding='same')
        self.stages = list()
        for (index, num_shift_blocks) in enumerate(num_shift_blocks_per_stages):
            if index == len(num_shift_blocks_per_stages) - 1:
                is_merge = False
            else:
                is_merge = True
            self.stages.append(StackedShiftBlocks(epsilon=epsilon, mlp_dropout_rate=mlp_dropout_rate, num_shift_blocks=num_shift_blocks, stochastic_depth_rate=stochastic_depth_rate, is_merge=is_merge, num_div=num_div, shift_pixel=shift_pixel, mlp_expand_ratio=mlp_expand_ratio))
        self.global_avg_pool = layers.GlobalAveragePooling2D()

    def get_config(self):
        if False:
            i = 10
            return i + 15
        config = super().get_config()
        config.update({'data_augmentation': self.data_augmentation, 'patch_projection': self.patch_projection, 'stages': self.stages, 'global_avg_pool': self.global_avg_pool})
        return config

    def _calculate_loss(self, data, training=False):
        if False:
            return 10
        (images, labels) = data
        augmented_images = self.data_augmentation(images, training=training)
        projected_patches = self.patch_projection(augmented_images)
        x = projected_patches
        for stage in self.stages:
            x = stage(x, training=training)
        logits = self.global_avg_pool(x)
        total_loss = self.compute_loss(data, labels, logits)
        return (total_loss, labels, logits)

    def train_step(self, inputs):
        if False:
            return 10
        with tf.GradientTape() as tape:
            (total_loss, labels, logits) = self._calculate_loss(data=inputs, training=True)
        train_vars = [self.data_augmentation.trainable_variables, self.patch_projection.trainable_variables, self.global_avg_pool.trainable_variables]
        train_vars = train_vars + [stage.trainable_variables for stage in self.stages]
        grads = tape.gradient(total_loss, train_vars)
        trainable_variable_list = []
        for (grad, var) in zip(grads, train_vars):
            for (g, v) in zip(grad, var):
                trainable_variable_list.append((g, v))
        self.optimizer.apply_gradients(trainable_variable_list)
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(total_loss)
            else:
                metric.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if False:
            print('Hello World!')
        (loss, labels, logits) = self._calculate_loss(data=data, training=False)
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
            else:
                metric.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}
'\n## Instantiate the model\n'
model = ShiftViTModel(data_augmentation=get_augmentation_model(), projected_dim=config.projected_dim, patch_size=config.patch_size, num_shift_blocks_per_stages=config.num_shift_blocks_per_stages, epsilon=config.epsilon, mlp_dropout_rate=config.mlp_dropout_rate, stochastic_depth_rate=config.stochastic_depth_rate, num_div=config.num_div, shift_pixel=config.shift_pixel, mlp_expand_ratio=config.mlp_expand_ratio)
'\n## Learning rate schedule\n\nIn many experiments, we want to warm up the model with a slowly increasing learning rate\nand then cool down the model with a slowly decaying learning rate. In the warmup cosine\ndecay, the learning rate linearly increases for the warmup steps and then decays with a\ncosine decay.\n'

class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses a warmup cosine decay schedule."""

    def __init__(self, lr_start, lr_max, warmup_steps, total_steps):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            lr_start: The initial learning rate\n            lr_max: The maximum learning rate to which lr should increase to in\n                the warmup steps\n            warmup_steps: The number of steps for which the model warms up\n            total_steps: The total number of steps for the model training\n        '
        super().__init__()
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if False:
            while True:
                i = 10
        if self.total_steps < self.warmup_steps:
            raise ValueError(f'Total number of steps {self.total_steps} must be' + f'larger or equal to warmup steps {self.warmup_steps}.')
        cos_annealed_lr = tf.cos(self.pi * (tf.cast(step, tf.float32) - self.warmup_steps) / tf.cast(self.total_steps - self.warmup_steps, tf.float32))
        learning_rate = 0.5 * self.lr_max * (1 + cos_annealed_lr)
        if self.warmup_steps > 0:
            if self.lr_max < self.lr_start:
                raise ValueError(f'lr_start {self.lr_start} must be smaller or' + f'equal to lr_max {self.lr_max}.')
            slope = (self.lr_max - self.lr_start) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.lr_start
            learning_rate = tf.where(step < self.warmup_steps, warmup_rate, learning_rate)
        return tf.where(step > self.total_steps, 0.0, learning_rate, name='learning_rate')
'\n## Compile and train the model\n'
total_steps = int(len(x_train) / config.batch_size * config.epochs)
warmup_epoch_percentage = 0.15
warmup_steps = int(total_steps * warmup_epoch_percentage)
scheduled_lrs = WarmUpCosine(lr_start=1e-05, lr_max=0.001, warmup_steps=warmup_steps, total_steps=total_steps)
optimizer = keras.optimizers.AdamW(learning_rate=scheduled_lrs, weight_decay=config.weight_decay)
model.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy'), keras.metrics.SparseTopKCategoricalAccuracy(5, name='top-5-accuracy')])
history = model.fit(train_ds, epochs=config.epochs, validation_data=val_ds, callbacks=[keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, mode='auto')])
print('TESTING')
(loss, acc_top1, acc_top5) = model.evaluate(test_ds)
print(f'Loss: {loss:0.2f}')
print(f'Top 1 test accuracy: {acc_top1 * 100:0.2f}%')
print(f'Top 5 test accuracy: {acc_top5 * 100:0.2f}%')
'\n## Conclusion\n\nThe most impactful contribution of the paper is not the novel architecture, but\nthe idea that hierarchical ViTs trained with no attention can perform quite well. This\nopens up the question of how essential attention is to the performance of ViTs.\n\nFor curious minds, we would suggest reading the\n[ConvNexT](https://arxiv.org/abs/2201.03545) paper which attends more to the training\nparadigms and architectural details of ViTs rather than providing a novel architecture\nbased on attention.\n\nAcknowledgements:\n\n- We would like to thank [PyImageSearch](https://pyimagesearch.com) for providing us with\nresources that helped in the completion of this project.\n- We would like to thank [JarvisLabs.ai](https://jarvislabs.ai/) for providing with the\nGPU credits.\n- We would like to thank [Manim Community](https://www.manim.community/) for the manim\nlibrary.\n- A personal note of thanks to [Puja Roychowdhury](https://twitter.com/pleb_talks) for\nhelping us with the Learning Rate Schedule.\n'