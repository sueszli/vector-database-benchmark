"""
Title: Image classification with ConvMixer
Author: [Sayak Paul](https://twitter.com/RisingSayak)
Date created: 2021/10/12
Last modified: 2021/10/12
Description: An all-convolutional network applied to patches of images.
Accelerator: GPU
Converted to Keras 3 by: [Md Awsafur Rahman](https://awsaf49.github.io)
"""
'\n## Introduction\n\nVision Transformers (ViT; [Dosovitskiy et al.](https://arxiv.org/abs/1612.00593)) extract\nsmall patches from the input images, linearly project them, and then apply the\nTransformer ([Vaswani et al.](https://arxiv.org/abs/1706.03762)) blocks. The application\nof ViTs to image recognition tasks is quickly becoming a promising area of research,\nbecause ViTs eliminate the need to have strong inductive biases (such as convolutions) for\nmodeling locality. This presents them as a general computation primititive capable of\nlearning just from the training data with as minimal inductive priors as possible. ViTs\nyield great downstream performance when trained with proper regularization, data\naugmentation, and relatively large datasets.\n\nIn the [Patches Are All You Need](https://openreview.net/pdf?id=TVHS5Y4dNvM) paper (note:\nat\nthe time of writing, it is a submission to the ICLR 2022 conference), the authors extend\nthe idea of using patches to train an all-convolutional network and demonstrate\ncompetitive results. Their architecture namely **ConvMixer** uses recipes from the recent\nisotrophic architectures like ViT, MLP-Mixer\n([Tolstikhin et al.](https://arxiv.org/abs/2105.01601)), such as using the same\ndepth and resolution across different layers in the network, residual connections,\nand so on.\n\nIn this example, we will implement the ConvMixer model and demonstrate its performance on\nthe CIFAR-10 dataset.\n'
'\n## Imports\n'
import keras
from keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
'\n## Hyperparameters\n\nTo keep run time short, we will train the model for only 10 epochs. To focus on\nthe core ideas of ConvMixer, we will not use other training-specific elements like\nRandAugment ([Cubuk et al.](https://arxiv.org/abs/1909.13719)). If you are interested in\nlearning more about those details, please refer to the\n[original paper](https://openreview.net/pdf?id=TVHS5Y4dNvM).\n'
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 128
num_epochs = 10
'\n## Load the CIFAR-10 dataset\n'
((x_train, y_train), (x_test, y_test)) = keras.datasets.cifar10.load_data()
val_split = 0.1
val_indices = int(len(x_train) * val_split)
(new_x_train, new_y_train) = (x_train[val_indices:], y_train[val_indices:])
(x_val, y_val) = (x_train[:val_indices], y_train[:val_indices])
print(f'Training data samples: {len(new_x_train)}')
print(f'Validation data samples: {len(x_val)}')
print(f'Test data samples: {len(x_test)}')
"\n## Prepare `tf.data.Dataset` objects\n\nOur data augmentation pipeline is different from what the authors used for the CIFAR-10\ndataset, which is fine for the purpose of the example.\nNote that, it's ok to use **TF APIs for data I/O and preprocessing** with other backends\n(jax, torch) as it is feature-complete framework when it comes to data preprocessing.\n"
image_size = 32
auto = tf.data.AUTOTUNE
augmentation_layers = [keras.layers.RandomCrop(image_size, image_size), keras.layers.RandomFlip('horizontal')]

def augment_images(images):
    if False:
        i = 10
        return i + 15
    for layer in augmentation_layers:
        images = layer(images, training=True)
    return images

def make_datasets(images, labels, is_train=False):
    if False:
        return 10
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_train:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.batch(batch_size)
    if is_train:
        dataset = dataset.map(lambda x, y: (augment_images(x), y), num_parallel_calls=auto)
    return dataset.prefetch(auto)
train_dataset = make_datasets(new_x_train, new_y_train, is_train=True)
val_dataset = make_datasets(x_val, y_val)
test_dataset = make_datasets(x_test, y_test)
'\n## ConvMixer utilities\n\nThe following figure (taken from the original paper) depicts the ConvMixer model:\n\n![](https://i.imgur.com/yF8actg.png)\n\nConvMixer is very similar to the MLP-Mixer, model with the following key\ndifferences:\n\n* Instead of using fully-connected layers, it uses standard convolution layers.\n* Instead of LayerNorm (which is typical for ViTs and MLP-Mixers), it uses BatchNorm.\n\nTwo types of convolution layers are used in ConvMixer. **(1)**: Depthwise convolutions,\nfor mixing spatial locations of the images, **(2)**: Pointwise convolutions (which follow\nthe depthwise convolutions), for mixing channel-wise information across the patches.\nAnother keypoint is the use of *larger kernel sizes* to allow a larger receptive field.\n'

def activation_block(x):
    if False:
        while True:
            i = 10
    x = layers.Activation('gelu')(x)
    return layers.BatchNormalization()(x)

def conv_stem(x, filters: int, patch_size: int):
    if False:
        i = 10
        return i + 15
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block(x)

def conv_mixer_block(x, filters: int, kernel_size: int):
    if False:
        print('Hello World!')
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding='same')(x)
    x = layers.Add()([activation_block(x), x0])
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)
    return x

def get_conv_mixer_256_8(image_size=32, filters=256, depth=8, kernel_size=5, patch_size=2, num_classes=10):
    if False:
        i = 10
        return i + 15
    'ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.\n    The hyperparameter values are taken from the paper.\n    '
    inputs = keras.Input((image_size, image_size, 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = conv_stem(x, filters, patch_size)
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)
'\nThe model used in this experiment is termed as **ConvMixer-256/8** where 256 denotes the\nnumber of channels and 8 denotes the depth. The resulting model only has 0.8 million\nparameters.\n'
'\n## Model training and evaluation utility\n'

def run_experiment(model):
    if False:
        print('Hello World!')
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    checkpoint_filepath = '/tmp/checkpoint.keras'
    checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_accuracy', save_best_only=True, save_weights_only=False)
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=num_epochs, callbacks=[checkpoint_callback])
    model.load_weights(checkpoint_filepath)
    (_, accuracy) = model.evaluate(test_dataset)
    print(f'Test accuracy: {round(accuracy * 100, 2)}%')
    return (history, model)
'\n## Train and evaluate model\n'
conv_mixer_model = get_conv_mixer_256_8()
(history, conv_mixer_model) = run_experiment(conv_mixer_model)
'\nThe gap in training and validation performance can be mitigated by using additional\nregularization techniques. Nevertheless, being able to get to ~83% accuracy within 10\nepochs with 0.8 million parameters is a strong result.\n'
'\n## Visualizing the internals of ConvMixer\n\nWe can visualize the patch embeddings and the learned convolution filters. Recall\nthat each patch embedding and intermediate feature map have the same number of channels\n(256 in this case). This will make our visualization utility easier to implement.\n'

def visualization_plot(weights, idx=1):
    if False:
        while True:
            i = 10
    (p_min, p_max) = (weights.min(), weights.max())
    weights = (weights - p_min) / (p_max - p_min)
    num_filters = 256
    plt.figure(figsize=(8, 8))
    for i in range(num_filters):
        current_weight = weights[:, :, :, i]
        if current_weight.shape[-1] == 1:
            current_weight = current_weight.squeeze()
        ax = plt.subplot(16, 16, idx)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(current_weight)
        idx += 1
patch_embeddings = conv_mixer_model.layers[2].get_weights()[0]
visualization_plot(patch_embeddings)
'\nEven though we did not train the network to convergence, we can notice that different\npatches show different patterns. Some share similarity with others while some are very\ndifferent. These visualizations are more salient with larger image sizes.\n\nSimilarly, we can visualize the raw convolution kernels. This can help us understand\nthe patterns to which a given kernel is receptive.\n'
for (i, layer) in enumerate(conv_mixer_model.layers):
    if isinstance(layer, layers.DepthwiseConv2D):
        if layer.get_config()['kernel_size'] == (5, 5):
            print(i, layer)
idx = 26
kernel = conv_mixer_model.layers[idx].get_weights()[0]
kernel = np.expand_dims(kernel.squeeze(), axis=2)
visualization_plot(kernel)
'\nWe see that different filters in the kernel have different locality spans, and this\npattern\nis likely to evolve with more training.\n'
"\n## Final notes\n\nThere's been a recent trend on fusing convolutions with other data-agnostic operations\nlike self-attention. Following works are along this line of research:\n\n* ConViT ([d'Ascoli et al.](https://arxiv.org/abs/2103.10697))\n* CCT ([Hassani et al.](https://arxiv.org/abs/2104.05704))\n* CoAtNet ([Dai et al.](https://arxiv.org/abs/2106.04803))\n"