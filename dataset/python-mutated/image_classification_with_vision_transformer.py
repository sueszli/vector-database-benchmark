"""
Title: Image classification with Vision Transformer
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Converted to Keras 3 by: [divyasreepat](https://github.com/divyashreepathihalli), [Soumik Rakshit](http://github.com/soumik12345)
Date created: 2021/01/18
Last modified: 2021/01/18
Description: Implementing the Vision Transformer (ViT) model for image classification.
Accelerator: GPU
"""
'\n## Introduction\n\nThis example implements the [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)\nmodel by Alexey Dosovitskiy et al. for image classification,\nand demonstrates it on the CIFAR-100 dataset.\nThe ViT model applies the Transformer architecture with self-attention to sequences of\nimage patches, without using convolution layers.\n\n'
'\n## Setup\n'
import os
os.environ['KERAS_BACKEND'] = 'jax'
import keras
from keras import layers
from keras import ops
import numpy as np
import matplotlib.pyplot as plt
'\n## Prepare the data\n'
num_classes = 100
input_shape = (32, 32, 3)
((x_train, y_train), (x_test, y_test)) = keras.datasets.cifar100.load_data()
print(f'x_train shape: {x_train.shape} - y_train shape: {y_train.shape}')
print(f'x_test shape: {x_test.shape} - y_test shape: {y_test.shape}')
'\n## Configure the hyperparameters\n'
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100
image_size = 72
patch_size = 6
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [projection_dim * 2, projection_dim]
transformer_layers = 8
mlp_head_units = [2048, 1024]
'\n## Use data augmentation\n'
data_augmentation = keras.Sequential([layers.Normalization(), layers.Resizing(image_size, image_size), layers.RandomFlip('horizontal'), layers.RandomRotation(factor=0.02), layers.RandomZoom(height_factor=0.2, width_factor=0.2)], name='data_augmentation')
data_augmentation.layers[0].adapt(x_train)
'\n## Implement multilayer perceptron (MLP)\n'

def mlp(x, hidden_units, dropout_rate):
    if False:
        i = 10
        return i + 15
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
'\n## Implement patch creation as a layer\n'

class Patches(layers.Layer):

    def __init__(self, patch_size):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        if False:
            i = 10
            return i + 15
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(patches, (batch_size, num_patches_h * num_patches_w, self.patch_size * self.patch_size * channels))
        return patches

    def get_config(self):
        if False:
            return 10
        config = super().get_config()
        config.update({'patch_size': self.patch_size})
        return config
"\nLet's display patches for a sample image\n"
plt.figure(figsize=(4, 4))
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image.astype('uint8'))
plt.axis('off')
resized_image = ops.image.resize(ops.convert_to_tensor([image]), size=(image_size, image_size))
patches = Patches(patch_size)(resized_image)
print(f'Image size: {image_size} X {image_size}')
print(f'Patch size: {patch_size} X {patch_size}')
print(f'Patches per image: {patches.shape[1]}')
print(f'Elements per patch: {patches.shape[-1]}')
n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for (i, patch) in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = ops.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(ops.convert_to_numpy(patch_img).astype('uint8'))
    plt.axis('off')
'\n## Implement the patch encoding layer\n\nThe `PatchEncoder` layer will linearly transform a patch by projecting it into a\nvector of size `projection_dim`. In addition, it adds a learnable position\nembedding to the projected vector.\n'

class PatchEncoder(layers.Layer):

    def __init__(self, num_patches, projection_dim):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        if False:
            return 10
        positions = ops.expand_dims(ops.arange(start=0, stop=self.num_patches, step=1), axis=0)
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        config = super().get_config()
        config.update({'num_patches': self.num_patches})
        return config
'\n## Build the ViT model\n\nThe ViT model consists of multiple Transformer blocks,\nwhich use the `layers.MultiHeadAttention` layer as a self-attention mechanism\napplied to the sequence of patches. The Transformer blocks produce a\n`[batch_size, num_patches, projection_dim]` tensor, which is processed via an\nclassifier head with softmax to produce the final class probabilities output.\n\nUnlike the technique described in the [paper](https://arxiv.org/abs/2010.11929),\nwhich prepends a learnable embedding to the sequence of encoded patches to serve\nas the image representation, all the outputs of the final Transformer block are\nreshaped with `layers.Flatten()` and used as the image\nrepresentation input to the classifier head.\nNote that the `layers.GlobalAveragePooling1D` layer\ncould also be used instead to aggregate the outputs of the Transformer block,\nespecially when the number of patches and the projection dimensions are large.\n'

def create_vit_classifier():
    if False:
        for i in range(10):
            print('nop')
    inputs = keras.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    patches = Patches(patch_size)(augmented)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-06)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-06)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])
    representation = layers.LayerNormalization(epsilon=1e-06)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes)(features)
    model = keras.Model(inputs=inputs, outputs=logits)
    return model
'\n## Compile, train, and evaluate the mode\n'

def run_experiment(model):
    if False:
        while True:
            i = 10
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy'), keras.metrics.SparseTopKCategoricalAccuracy(5, name='top-5-accuracy')])
    checkpoint_filepath = '/tmp/checkpoint.weights.h5'
    checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_accuracy', save_best_only=True, save_weights_only=True)
    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0.1, callbacks=[checkpoint_callback])
    model.load_weights(checkpoint_filepath)
    (_, accuracy, top_5_accuracy) = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {round(accuracy * 100, 2)}%')
    print(f'Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%')
    return history
vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)

def plot_history(item):
    if False:
        for i in range(10):
            print('nop')
    plt.plot(history.history[item], label=item)
    plt.plot(history.history['val_' + item], label='val_' + item)
    plt.xlabel('Epochs')
    plt.ylabel(item)
    plt.title('Train and Validation {} Over Epochs'.format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()
plot_history('loss')
plot_history('top-5-accuracy')
"\nAfter 100 epochs, the ViT model achieves around 55% accuracy and\n82% top-5 accuracy on the test data. These are not competitive results on the CIFAR-100 dataset,\nas a ResNet50V2 trained from scratch on the same data can achieve 67% accuracy.\n\nNote that the state of the art results reported in the\n[paper](https://arxiv.org/abs/2010.11929) are achieved by pre-training the ViT model using\nthe JFT-300M dataset, then fine-tuning it on the target dataset. To improve the model quality\nwithout pre-training, you can try to train the model for more epochs, use a larger number of\nTransformer layers, resize the input images, change the patch size, or increase the projection dimensions. \nBesides, as mentioned in the paper, the quality of the model is affected not only by architecture choices, \nbut also by parameters such as the learning rate schedule, optimizer, weight decay, etc.\nIn practice, it's recommended to fine-tune a ViT model\nthat was pre-trained using a large, high-resolution dataset.\n"