"""
Title: Image classification with modern MLP models
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Converted to Keras 3 by: [Guillaume Baquiast](https://www.linkedin.com/in/guillaume-baquiast-478965ba/), [divyasreepat](https://github.com/divyashreepathihalli)
Date created: 2021/05/30
Last modified: 2023/08/03
Description: Implementing the MLP-Mixer, FNet, and gMLP models for CIFAR-100 image classification.
Accelerator: GPU
"""
'\n## Introduction\n\nThis example implements three modern attention-free, multi-layer perceptron (MLP) based models for image\nclassification, demonstrated on the CIFAR-100 dataset:\n\n1. The [MLP-Mixer](https://arxiv.org/abs/2105.01601) model, by Ilya Tolstikhin et al., based on two types of MLPs.\n3. The [FNet](https://arxiv.org/abs/2105.03824) model, by James Lee-Thorp et al., based on unparameterized\nFourier Transform.\n2. The [gMLP](https://arxiv.org/abs/2105.08050) model, by Hanxiao Liu et al., based on MLP with gating.\n\nThe purpose of the example is not to compare between these models, as they might perform differently on\ndifferent datasets with well-tuned hyperparameters. Rather, it is to show simple implementations of their\nmain building blocks.\n'
'\n## Setup\n'
import numpy as np
import keras
from keras import layers
'\n## Prepare the data\n'
num_classes = 100
input_shape = (32, 32, 3)
((x_train, y_train), (x_test, y_test)) = keras.datasets.cifar100.load_data()
print(f'x_train shape: {x_train.shape} - y_train shape: {y_train.shape}')
print(f'x_test shape: {x_test.shape} - y_test shape: {y_test.shape}')
'\n## Configure the hyperparameters\n'
weight_decay = 0.0001
batch_size = 128
num_epochs = 50
dropout_rate = 0.2
image_size = 64
patch_size = 8
num_patches = (image_size // patch_size) ** 2
embedding_dim = 256
num_blocks = 4
print(f'Image size: {image_size} X {image_size} = {image_size ** 2}')
print(f'Patch size: {patch_size} X {patch_size} = {patch_size ** 2} ')
print(f'Patches per image: {num_patches}')
print(f'Elements per patch (3 channels): {patch_size ** 2 * 3}')
'\n## Build a classification model\n\nWe implement a method that builds a classifier given the processing blocks.\n'

def build_classifier(blocks, positional_encoding=False):
    if False:
        i = 10
        return i + 15
    inputs = layers.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    patches = Patches(patch_size)(augmented)
    x = layers.Dense(units=embedding_dim)(patches)
    if positional_encoding:
        x = x + PositionEmbedding(sequence_length=num_patches)(x)
    x = blocks(x)
    representation = layers.GlobalAveragePooling1D()(x)
    representation = layers.Dropout(rate=dropout_rate)(representation)
    logits = layers.Dense(num_classes)(representation)
    return keras.Model(inputs=inputs, outputs=logits)
'\n## Define an experiment\n\nWe implement a utility function to compile, train, and evaluate a given model.\n'

def run_experiment(model):
    if False:
        while True:
            i = 10
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[keras.metrics.SparseCategoricalAccuracy(name='acc'), keras.metrics.SparseTopKCategoricalAccuracy(5, name='top5-acc')])
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0.1, callbacks=[early_stopping, reduce_lr])
    (_, accuracy, top_5_accuracy) = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {round(accuracy * 100, 2)}%')
    print(f'Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%')
    return history
'\n## Use data augmentation\n'
data_augmentation = keras.Sequential([layers.Normalization(), layers.Resizing(image_size, image_size), layers.RandomFlip('horizontal'), layers.RandomZoom(height_factor=0.2, width_factor=0.2)], name='data_augmentation')
data_augmentation.layers[0].adapt(x_train)
'\n## Implement patch extraction as a layer\n'

class Patches(layers.Layer):

    def __init__(self, patch_size, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, x):
        if False:
            print('Hello World!')
        patches = keras.ops.image.extract_patches(x, self.patch_size)
        batch_size = keras.ops.shape(patches)[0]
        num_patches = keras.ops.shape(patches)[1] * keras.ops.shape(patches)[2]
        patch_dim = keras.ops.shape(patches)[3]
        out = keras.ops.reshape(patches, (batch_size, num_patches, patch_dim))
        return out
'\n## Implement position embedding as a layer\n'

class PositionEmbedding(keras.layers.Layer):

    def __init__(self, sequence_length, initializer='glorot_uniform', **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        if sequence_length is None:
            raise ValueError('`sequence_length` must be an Integer, received `None`.')
        self.sequence_length = int(sequence_length)
        self.initializer = keras.initializers.get(initializer)

    def get_config(self):
        if False:
            return 10
        config = super().get_config()
        config.update({'sequence_length': self.sequence_length, 'initializer': keras.initializers.serialize(self.initializer)})
        return config

    def build(self, input_shape):
        if False:
            while True:
                i = 10
        feature_size = input_shape[-1]
        self.position_embeddings = self.add_weight(name='embeddings', shape=[self.sequence_length, feature_size], initializer=self.initializer, trainable=True)
        super().build(input_shape)

    def call(self, inputs, start_index=0):
        if False:
            print('Hello World!')
        shape = keras.ops.shape(inputs)
        feature_length = shape[-1]
        sequence_length = shape[-2]
        position_embeddings = keras.ops.convert_to_tensor(self.position_embeddings)
        position_embeddings = keras.ops.slice(position_embeddings, (start_index, 0), (sequence_length, feature_length))
        return keras.ops.broadcast_to(position_embeddings, shape)

    def compute_output_shape(self, input_shape):
        if False:
            return 10
        return input_shape
'\n## The MLP-Mixer model\n\nThe MLP-Mixer is an architecture based exclusively on\nmulti-layer perceptrons (MLPs), that contains two types of MLP layers:\n\n1. One applied independently to image patches, which mixes the per-location features.\n2. The other applied across patches (along channels), which mixes spatial information.\n\nThis is similar to a [depthwise separable convolution based model](https://arxiv.org/pdf/1610.02357.pdf)\nsuch as the Xception model, but with two chained dense transforms, no max pooling, and layer normalization\ninstead of batch normalization.\n'
'\n### Implement the MLP-Mixer module\n'

class MLPMixerLayer(layers.Layer):

    def __init__(self, num_patches, hidden_units, dropout_rate, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self.mlp1 = keras.Sequential([layers.Dense(units=num_patches, activation='gelu'), layers.Dense(units=num_patches), layers.Dropout(rate=dropout_rate)])
        self.mlp2 = keras.Sequential([layers.Dense(units=num_patches, activation='gelu'), layers.Dense(units=hidden_units), layers.Dropout(rate=dropout_rate)])
        self.normalize = layers.LayerNormalization(epsilon=1e-06)

    def build(self, input_shape):
        if False:
            while True:
                i = 10
        return super().build(input_shape)

    def call(self, inputs):
        if False:
            while True:
                i = 10
        x = self.normalize(inputs)
        x_channels = keras.ops.transpose(x, axes=(0, 2, 1))
        mlp1_outputs = self.mlp1(x_channels)
        mlp1_outputs = keras.ops.transpose(mlp1_outputs, axes=(0, 2, 1))
        x = mlp1_outputs + inputs
        x_patches = self.normalize(x)
        mlp2_outputs = self.mlp2(x_patches)
        x = x + mlp2_outputs
        return x
'\n### Build, train, and evaluate the MLP-Mixer model\n\nNote that training the model with the current settings on a V100 GPUs\ntakes around 8 seconds per epoch.\n'
mlpmixer_blocks = keras.Sequential([MLPMixerLayer(num_patches, embedding_dim, dropout_rate) for _ in range(num_blocks)])
learning_rate = 0.005
mlpmixer_classifier = build_classifier(mlpmixer_blocks)
history = run_experiment(mlpmixer_classifier)
'\nThe MLP-Mixer model tends to have much less number of parameters compared\nto convolutional and transformer-based models, which leads to less training and\nserving computational cost.\n\nAs mentioned in the [MLP-Mixer](https://arxiv.org/abs/2105.01601) paper,\nwhen pre-trained on large datasets, or with modern regularization schemes,\nthe MLP-Mixer attains competitive scores to state-of-the-art models.\nYou can obtain better results by increasing the embedding dimensions,\nincreasing the number of mixer blocks, and training the model for longer.\nYou may also try to increase the size of the input images and use different patch sizes.\n'
'\n## The FNet model\n\nThe FNet uses a similar block to the Transformer block. However, FNet replaces the self-attention layer\nin the Transformer block with a parameter-free 2D Fourier transformation layer:\n\n1. One 1D Fourier Transform is applied along the patches.\n2. One 1D Fourier Transform is applied along the channels.\n'
'\n### Implement the FNet module\n'

class FNetLayer(layers.Layer):

    def __init__(self, embedding_dim, dropout_rate, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.ffn = keras.Sequential([layers.Dense(units=embedding_dim, activation='gelu'), layers.Dropout(rate=dropout_rate), layers.Dense(units=embedding_dim)])
        self.normalize1 = layers.LayerNormalization(epsilon=1e-06)
        self.normalize2 = layers.LayerNormalization(epsilon=1e-06)

    def call(self, inputs):
        if False:
            print('Hello World!')
        real_part = inputs
        im_part = keras.ops.zeros_like(inputs)
        x = keras.ops.fft2((real_part, im_part))[0]
        x = x + inputs
        x = self.normalize1(x)
        x_ffn = self.ffn(x)
        x = x + x_ffn
        return self.normalize2(x)
'\n### Build, train, and evaluate the FNet model\n\nNote that training the model with the current settings on a V100 GPUs\ntakes around 8 seconds per epoch.\n'
fnet_blocks = keras.Sequential([FNetLayer(embedding_dim, dropout_rate) for _ in range(num_blocks)])
learning_rate = 0.001
fnet_classifier = build_classifier(fnet_blocks, positional_encoding=True)
history = run_experiment(fnet_classifier)
'\nAs shown in the [FNet](https://arxiv.org/abs/2105.03824) paper,\nbetter results can be achieved by increasing the embedding dimensions,\nincreasing the number of FNet blocks, and training the model for longer.\nYou may also try to increase the size of the input images and use different patch sizes.\nThe FNet scales very efficiently to long inputs, runs much faster than attention-based\nTransformer models, and produces competitive accuracy results.\n'
'\n## The gMLP model\n\nThe gMLP is a MLP architecture that features a Spatial Gating Unit (SGU).\nThe SGU enables cross-patch interactions across the spatial (channel) dimension, by:\n\n1. Transforming the input spatially by applying linear projection across patches (along channels).\n2. Applying element-wise multiplication of the input and its spatial transformation.\n'
'\n### Implement the gMLP module\n'

class gMLPLayer(layers.Layer):

    def __init__(self, num_patches, embedding_dim, dropout_rate, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self.channel_projection1 = keras.Sequential([layers.Dense(units=embedding_dim * 2, activation='gelu'), layers.Dropout(rate=dropout_rate)])
        self.channel_projection2 = layers.Dense(units=embedding_dim)
        self.spatial_projection = layers.Dense(units=num_patches, bias_initializer='Ones')
        self.normalize1 = layers.LayerNormalization(epsilon=1e-06)
        self.normalize2 = layers.LayerNormalization(epsilon=1e-06)

    def spatial_gating_unit(self, x):
        if False:
            while True:
                i = 10
        (u, v) = keras.ops.split(x, indices_or_sections=2, axis=2)
        v = self.normalize2(v)
        v_channels = keras.ops.transpose(v, axes=(0, 2, 1))
        v_projected = self.spatial_projection(v_channels)
        v_projected = keras.ops.transpose(v_projected, axes=(0, 2, 1))
        return u * v_projected

    def call(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        x = self.normalize1(inputs)
        x_projected = self.channel_projection1(x)
        x_spatial = self.spatial_gating_unit(x_projected)
        x_projected = self.channel_projection2(x_spatial)
        return x + x_projected
'\n### Build, train, and evaluate the gMLP model\n\nNote that training the model with the current settings on a V100 GPUs\ntakes around 9 seconds per epoch.\n'
gmlp_blocks = keras.Sequential([gMLPLayer(num_patches, embedding_dim, dropout_rate) for _ in range(num_blocks)])
learning_rate = 0.003
gmlp_classifier = build_classifier(gmlp_blocks)
history = run_experiment(gmlp_classifier)
'\nAs shown in the [gMLP](https://arxiv.org/abs/2105.08050) paper,\nbetter results can be achieved by increasing the embedding dimensions,\nincreasing the number of gMLP blocks, and training the model for longer.\nYou may also try to increase the size of the input images and use different patch sizes.\nNote that, the paper used advanced regularization strategies, such as MixUp and CutMix,\nas well as AutoAugment.\n'