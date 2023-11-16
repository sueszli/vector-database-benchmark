"""
Title: Image classification with Perceiver
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Date created: 2021/04/30
Last modified: 2021/01/30
Description: Implementing the Perceiver model for image classification.
Accelerator: GPU
"""
"\n## Introduction\n\nThis example implements the\n[Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206)\nmodel by Andrew Jaegle et al. for image classification,\nand demonstrates it on the CIFAR-100 dataset.\n\nThe Perceiver model leverages an asymmetric attention mechanism to iteratively\ndistill inputs into a tight latent bottleneck,\nallowing it to scale to handle very large inputs.\n\nIn other words: let's assume that your input data array (e.g. image) has `M` elements (i.e. patches), where `M` is large.\nIn a standard Transformer model, a self-attention operation is performed for the `M` elements.\nThe complexity of this operation is `O(M^2)`.\nHowever, the Perceiver model creates a latent array of size `N` elements, where `N << M`,\nand performs two operations iteratively:\n\n1. Cross-attention Transformer between the latent array and the data array - The complexity of this operation is `O(M.N)`.\n2. Self-attention Transformer on the latent array -  The complexity of this operation is `O(N^2)`.\n\nThis example requires TensorFlow 2.4 or higher, as well as\n[TensorFlow Addons](https://www.tensorflow.org/addons/overview),\nwhich can be installed using the following command:\n\n```python\npip install -U tensorflow-addons\n```\n"
'\n## Setup\n'
import numpy as np
import tensorflow as tf
import keras
from keras import layers
'\n## Prepare the data\n'
num_classes = 100
input_shape = (32, 32, 3)
((x_train, y_train), (x_test, y_test)) = keras.datasets.cifar100.load_data()
print(f'x_train shape: {x_train.shape} - y_train shape: {y_train.shape}')
print(f'x_test shape: {x_test.shape} - y_test shape: {y_test.shape}')
'\n## Configure the hyperparameters\n'
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 64
num_epochs = 50
dropout_rate = 0.2
image_size = 64
patch_size = 2
num_patches = (image_size // patch_size) ** 2
latent_dim = 256
projection_dim = 256
num_heads = 8
ffn_units = [projection_dim, projection_dim]
num_transformer_blocks = 4
num_iterations = 2
classifier_units = [projection_dim, num_classes]
print(f'Image size: {image_size} X {image_size} = {image_size ** 2}')
print(f'Patch size: {patch_size} X {patch_size} = {patch_size ** 2} ')
print(f'Patches per image: {num_patches}')
print(f'Elements per patch (3 channels): {patch_size ** 2 * 3}')
print(f'Latent array shape: {latent_dim} X {projection_dim}')
print(f'Data array shape: {num_patches} X {projection_dim}')
'\nNote that, in order to use each pixel as an individual input in the data array,\nset `patch_size` to 1.\n'
'\n## Use data augmentation\n'
data_augmentation = keras.Sequential([layers.Normalization(), layers.Resizing(image_size, image_size), layers.RandomFlip('horizontal'), layers.RandomZoom(height_factor=0.2, width_factor=0.2)], name='data_augmentation')
data_augmentation.layers[0].adapt(x_train)
'\n## Implement Feedforward network (FFN)\n'

def create_ffn(hidden_units, dropout_rate):
    if False:
        for i in range(10):
            print('nop')
    ffn_layers = []
    for units in hidden_units[:-1]:
        ffn_layers.append(layers.Dense(units, activation=tf.nn.gelu))
    ffn_layers.append(layers.Dense(units=hidden_units[-1]))
    ffn_layers.append(layers.Dropout(dropout_rate))
    ffn = keras.Sequential(ffn_layers)
    return ffn
'\n## Implement patch creation as a layer\n'

class Patches(layers.Layer):

    def __init__(self, patch_size):
        if False:
            while True:
                i = 10
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        if False:
            return 10
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(images=images, sizes=[1, self.patch_size, self.patch_size, 1], strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding='VALID')
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
'\n## Implement the patch encoding layer\n\nThe `PatchEncoder` layer will linearly transform a patch by projecting it into\na vector of size `latent_dim`. In addition, it adds a learnable position embedding\nto the projected vector.\n\nNote that the orginal Perceiver paper uses the Fourier feature positional encodings.\n'

class PatchEncoder(layers.Layer):

    def __init__(self, num_patches, projection_dim):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patches):
        if False:
            return 10
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded
'\n## Build the Perceiver model\n\nThe Perceiver consists of two modules: a cross-attention\nmodule and a standard Transformer with self-attention.\n'
'\n### Cross-attention module\n\nThe cross-attention expects a `(latent_dim, projection_dim)` latent array,\nand the `(data_dim,  projection_dim)` data array as inputs,\nto produce a `(latent_dim, projection_dim)` latent array as an output.\nTo apply cross-attention, the `query` vectors are generated from the latent array,\nwhile the `key` and `value` vectors are generated from the encoded image.\n\nNote that the data array in this example is the image,\nwhere the `data_dim` is set to the `num_patches`.\n'

def create_cross_attention_module(latent_dim, data_dim, projection_dim, ffn_units, dropout_rate):
    if False:
        return 10
    inputs = {'latent_array': layers.Input(shape=(latent_dim, projection_dim), name='latent_array'), 'data_array': layers.Input(shape=(data_dim, projection_dim), name='data_array')}
    latent_array = layers.LayerNormalization(epsilon=1e-06)(inputs['latent_array'])
    data_array = layers.LayerNormalization(epsilon=1e-06)(inputs['data_array'])
    query = layers.Dense(units=projection_dim)(latent_array)
    key = layers.Dense(units=projection_dim)(data_array)
    value = layers.Dense(units=projection_dim)(data_array)
    attention_output = layers.Attention(use_scale=True, dropout=0.1)([query, key, value], return_attention_scores=False)
    attention_output = layers.Add()([attention_output, latent_array])
    attention_output = layers.LayerNormalization(epsilon=1e-06)(attention_output)
    ffn = create_ffn(hidden_units=ffn_units, dropout_rate=dropout_rate)
    outputs = ffn(attention_output)
    outputs = layers.Add()([outputs, attention_output])
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
'\n### Transformer module\n\nThe Transformer expects the output latent vector from the cross-attention module\nas an input, applies multi-head self-attention to its `latent_dim` elements,\nfollowed by feedforward network, to produce another `(latent_dim, projection_dim)` latent array.\n'

def create_transformer_module(latent_dim, projection_dim, num_heads, num_transformer_blocks, ffn_units, dropout_rate):
    if False:
        i = 10
        return i + 15
    inputs = layers.Input(shape=(latent_dim, projection_dim))
    x0 = inputs
    for _ in range(num_transformer_blocks):
        x1 = layers.LayerNormalization(epsilon=1e-06)(x0)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        x2 = layers.Add()([attention_output, x0])
        x3 = layers.LayerNormalization(epsilon=1e-06)(x2)
        ffn = create_ffn(hidden_units=ffn_units, dropout_rate=dropout_rate)
        x3 = ffn(x3)
        x0 = layers.Add()([x3, x2])
    model = keras.Model(inputs=inputs, outputs=x0)
    return model
'\n### Perceiver model\n\nThe Perceiver model repeats the cross-attention and Transformer modules\n`num_iterations` times—with shared weights and skip connections—to allow\nthe latent array to iteratively extract information from the input image as it is needed.\n'

class Perceiver(keras.Model):

    def __init__(self, patch_size, data_dim, latent_dim, projection_dim, num_heads, num_transformer_blocks, ffn_units, dropout_rate, num_iterations, classifier_units):
        if False:
            return 10
        super().__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
        self.ffn_units = ffn_units
        self.dropout_rate = dropout_rate
        self.num_iterations = num_iterations
        self.classifier_units = classifier_units

    def build(self, input_shape):
        if False:
            print('Hello World!')
        self.latent_array = self.add_weight(shape=(self.latent_dim, self.projection_dim), initializer='random_normal', trainable=True)
        self.patcher = Patches(self.patch_size)
        self.patch_encoder = PatchEncoder(self.data_dim, self.projection_dim)
        self.cross_attention = create_cross_attention_module(self.latent_dim, self.data_dim, self.projection_dim, self.ffn_units, self.dropout_rate)
        self.transformer = create_transformer_module(self.latent_dim, self.projection_dim, self.num_heads, self.num_transformer_blocks, self.ffn_units, self.dropout_rate)
        self.global_average_pooling = layers.GlobalAveragePooling1D()
        self.classification_head = create_ffn(hidden_units=self.classifier_units, dropout_rate=self.dropout_rate)
        super().build(input_shape)

    def call(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        augmented = data_augmentation(inputs)
        patches = self.patcher(augmented)
        encoded_patches = self.patch_encoder(patches)
        cross_attention_inputs = {'latent_array': tf.expand_dims(self.latent_array, 0), 'data_array': encoded_patches}
        for _ in range(self.num_iterations):
            latent_array = self.cross_attention(cross_attention_inputs)
            latent_array = self.transformer(latent_array)
            cross_attention_inputs['latent_array'] = latent_array
        representation = self.global_average_pooling(latent_array)
        logits = self.classification_head(representation)
        return logits
'\n## Compile, train, and evaluate the mode\n'

def run_experiment(model):
    if False:
        for i in range(10):
            print('nop')
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[keras.metrics.SparseCategoricalAccuracy(name='acc'), keras.metrics.SparseTopKCategoricalAccuracy(5, name='top5-acc')])
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0.1, callbacks=[early_stopping, reduce_lr])
    (_, accuracy, top_5_accuracy) = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {round(accuracy * 100, 2)}%')
    print(f'Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%')
    return history
'\nNote that training the perceiver model with the current settings on a V100 GPUs takes\naround 200 seconds.\n'
perceiver_classifier = Perceiver(patch_size, num_patches, latent_dim, projection_dim, num_heads, num_transformer_blocks, ffn_units, dropout_rate, num_iterations, classifier_units)
history = run_experiment(perceiver_classifier)
'\nAfter 40 epochs, the Perceiver model achieves around 53% accuracy and 81% top-5 accuracy on the test data.\n\nAs mentioned in the ablations of the [Perceiver](https://arxiv.org/abs/2103.03206) paper,\nyou can obtain better results by increasing the latent array size,\nincreasing the (projection) dimensions of the latent array and data array elements,\nincreasing the number of blocks in the Transformer module, and increasing the number of iterations of applying\nthe cross-attention and the latent Transformer modules. You may also try to increase the size the input images\nand use different patch sizes.\n\nThe Perceiver benefits from inceasing the model size. However, larger models needs bigger accelerators\nto fit in and train efficiently. This is why in the Perceiver paper they used 32 TPU cores to run the experiments.\n'