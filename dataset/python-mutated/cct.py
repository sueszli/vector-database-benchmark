"""
Title: Compact Convolutional Transformers
Author: [Sayak Paul](https://twitter.com/RisingSayak)
Date created: 2021/06/30
Last modified: 2023/08/07
Description: Compact Convolutional Transformers for efficient image classification.
Accelerator: GPU
Converted to Keras 3 by: [Muhammad Anas Raza](https://anasrz.com), [Guillaume Baquiast](https://www.linkedin.com/in/guillaume-baquiast-478965ba/)
"""
"\nAs discussed in the [Vision Transformers (ViT)](https://arxiv.org/abs/2010.11929) paper,\na Transformer-based architecture for vision typically requires a larger dataset than\nusual, as well as a longer pre-training schedule. [ImageNet-1k](http://imagenet.org/)\n(which has about a million images) is considered to fall under the medium-sized data regime with\nrespect to ViTs. This is primarily because, unlike CNNs, ViTs (or a typical\nTransformer-based architecture) do not have well-informed inductive biases (such as\nconvolutions for processing images). This begs the question: can't we combine the\nbenefits of convolution and the benefits of Transformers\nin a single network architecture? These benefits include parameter-efficiency, and\nself-attention to process long-range and global dependencies (interactions between\ndifferent regions in an image).\n\nIn [Escaping the Big Data Paradigm with Compact Transformers](https://arxiv.org/abs/2104.05704),\nHassani et al. present an approach for doing exactly this. They proposed the\n**Compact Convolutional Transformer** (CCT) architecture. In this example, we will work on an\nimplementation of CCT and we will see how well it performs on the CIFAR-10 dataset.\n\nIf you are unfamiliar with the concept of self-attention or Transformers, you can read\n[this chapter](https://livebook.manning.com/book/deep-learning-with-python-second-edition/chapter-11/r-3/312)\nfrom  François Chollet's book *Deep Learning with Python*. This example uses\ncode snippets from another example,\n[Image classification with Vision Transformer](https://keras.io/examples/vision/image_classification_with_vision_transformer/).\n"
'\n## Imports\n'
from keras import layers
import keras
import matplotlib.pyplot as plt
import numpy as np
'\n## Hyperparameters and constants\n'
positional_emb = True
conv_layers = 2
projection_dim = 128
num_heads = 2
transformer_units = [projection_dim, projection_dim]
transformer_layers = 2
stochastic_depth_rate = 0.1
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 128
num_epochs = 30
image_size = 32
'\n## Load CIFAR-10 dataset\n'
num_classes = 10
input_shape = (32, 32, 3)
((x_train, y_train), (x_test, y_test)) = keras.datasets.cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(f'x_train shape: {x_train.shape} - y_train shape: {y_train.shape}')
print(f'x_test shape: {x_test.shape} - y_test shape: {y_test.shape}')
'\n## The CCT tokenizer\n\nThe first recipe introduced by the CCT authors is the tokenizer for processing the\nimages. In a standard ViT, images are organized into uniform *non-overlapping* patches.\nThis eliminates the boundary-level information present in between different patches. This\nis important for a neural network to effectively exploit the locality information. The\nfigure below presents an illustration of how images are organized into patches.\n\n![](https://i.imgur.com/IkBK9oY.png)\n\nWe already know that convolutions are quite good at exploiting locality information. So,\nbased on this, the authors introduce an all-convolution mini-network to produce image\npatches.\n'

class CCTTokenizer(layers.Layer):

    def __init__(self, kernel_size=3, stride=1, padding=1, pooling_kernel_size=3, pooling_stride=2, num_conv_layers=conv_layers, num_output_channels=[64, 128], positional_emb=positional_emb, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.conv_model = keras.Sequential()
        for i in range(num_conv_layers):
            self.conv_model.add(layers.Conv2D(num_output_channels[i], kernel_size, stride, padding='valid', use_bias=False, activation='relu', kernel_initializer='he_normal'))
            self.conv_model.add(layers.ZeroPadding2D(padding))
            self.conv_model.add(layers.MaxPooling2D(pooling_kernel_size, pooling_stride, 'same'))
        self.positional_emb = positional_emb

    def call(self, images):
        if False:
            for i in range(10):
                print('nop')
        outputs = self.conv_model(images)
        reshaped = keras.ops.reshape(outputs, (-1, keras.ops.shape(outputs)[1] * keras.ops.shape(outputs)[2], keras.ops.shape(outputs)[-1]))
        return reshaped
'\nPositional embeddings are optional in CCT. If we want to use them, we can use\nthe Layer defined below.\n'

class PositionEmbedding(keras.layers.Layer):

    def __init__(self, sequence_length, initializer='glorot_uniform', **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        if sequence_length is None:
            raise ValueError('`sequence_length` must be an Integer, received `None`.')
        self.sequence_length = int(sequence_length)
        self.initializer = keras.initializers.get(initializer)

    def get_config(self):
        if False:
            while True:
                i = 10
        config = super().get_config()
        config.update({'sequence_length': self.sequence_length, 'initializer': keras.initializers.serialize(self.initializer)})
        return config

    def build(self, input_shape):
        if False:
            for i in range(10):
                print('nop')
        feature_size = input_shape[-1]
        self.position_embeddings = self.add_weight(name='embeddings', shape=[self.sequence_length, feature_size], initializer=self.initializer, trainable=True)
        super().build(input_shape)

    def call(self, inputs, start_index=0):
        if False:
            i = 10
            return i + 15
        shape = keras.ops.shape(inputs)
        feature_length = shape[-1]
        sequence_length = shape[-2]
        position_embeddings = keras.ops.convert_to_tensor(self.position_embeddings)
        position_embeddings = keras.ops.slice(position_embeddings, (start_index, 0), (sequence_length, feature_length))
        return keras.ops.broadcast_to(position_embeddings, shape)

    def compute_output_shape(self, input_shape):
        if False:
            for i in range(10):
                print('nop')
        return input_shape
'\n## Sequence Pooling\nAnother recipe introduced in CCT is attention pooling or sequence pooling. In ViT, only\nthe feature map corresponding to the class token is pooled and is then used for the\nsubsequent classification task (or any other downstream task).\n'

class SequencePooling(layers.Layer):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.attention = layers.Dense(1)

    def call(self, x):
        if False:
            return 10
        attention_weights = keras.ops.softmax(self.attention(x), axis=1)
        attention_weights = keras.ops.transpose(attention_weights, axes=(0, 2, 1))
        weighted_representation = keras.ops.matmul(attention_weights, x)
        return keras.ops.squeeze(weighted_representation, -2)
'\n## Stochastic depth for regularization\n\n[Stochastic depth](https://arxiv.org/abs/1603.09382) is a regularization technique that\nrandomly drops a set of layers. During inference, the layers are kept as they are. It is\nvery much similar to [Dropout](https://jmlr.org/papers/v15/srivastava14a.html) but only\nthat it operates on a block of layers rather than individual nodes present inside a\nlayer. In CCT, stochastic depth is used just before the residual blocks of a Transformers\nencoder.\n'

class StochasticDepth(layers.Layer):

    def __init__(self, drop_prop, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.drop_prob = drop_prop
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, x, training=None):
        if False:
            while True:
                i = 10
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (keras.ops.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            random_tensor = keep_prob + keras.random.uniform(shape, 0, 1, seed=self.seed_generator)
            random_tensor = keras.ops.floor(random_tensor)
            return x / keep_prob * random_tensor
        return x
'\n## MLP for the Transformers encoder\n'

def mlp(x, hidden_units, dropout_rate):
    if False:
        print('Hello World!')
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.ops.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
'\n## Data augmentation\n\nIn the [original paper](https://arxiv.org/abs/2104.05704), the authors use\n[AutoAugment](https://arxiv.org/abs/1805.09501) to induce stronger regularization. For\nthis example, we will be using the standard geometric augmentations like random cropping\nand flipping.\n'
data_augmentation = keras.Sequential([layers.Rescaling(scale=1.0 / 255), layers.RandomCrop(image_size, image_size), layers.RandomFlip('horizontal')], name='data_augmentation')
'\n## The final CCT model\n\nIn CCT, outputs from the Transformers encoder are weighted and then passed on to the final task-specific layer (in\nthis example, we do classification).\n'

def create_cct_model(image_size=image_size, input_shape=input_shape, num_heads=num_heads, projection_dim=projection_dim, transformer_units=transformer_units):
    if False:
        print('Hello World!')
    inputs = layers.Input(input_shape)
    augmented = data_augmentation(inputs)
    cct_tokenizer = CCTTokenizer()
    encoded_patches = cct_tokenizer(augmented)
    if positional_emb:
        sequence_length = encoded_patches.shape[1]
        encoded_patches += PositionEmbedding(sequence_length=sequence_length)(encoded_patches)
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]
    for i in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-05)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-05)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = layers.Add()([x3, x2])
    representation = layers.LayerNormalization(epsilon=1e-05)(encoded_patches)
    weighted_representation = SequencePooling()(representation)
    logits = layers.Dense(num_classes)(weighted_representation)
    model = keras.Model(inputs=inputs, outputs=logits)
    return model
'\n## Model training and evaluation\n'

def run_experiment(model):
    if False:
        for i in range(10):
            print('nop')
    optimizer = keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)
    model.compile(optimizer=optimizer, loss=keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1), metrics=[keras.metrics.CategoricalAccuracy(name='accuracy'), keras.metrics.TopKCategoricalAccuracy(5, name='top-5-accuracy')])
    checkpoint_filepath = '/tmp/checkpoint.weights.h5'
    checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_accuracy', save_best_only=True, save_weights_only=True)
    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0.1, callbacks=[checkpoint_callback])
    model.load_weights(checkpoint_filepath)
    (_, accuracy, top_5_accuracy) = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {round(accuracy * 100, 2)}%')
    print(f'Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%')
    return history
cct_model = create_cct_model()
history = run_experiment(cct_model)
"\nLet's now visualize the training progress of the model.\n"
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation Losses Over Epochs', fontsize=14)
plt.legend()
plt.grid()
plt.show()
'\nThe CCT model we just trained has just **0.4 million** parameters, and it gets us to\n~79% top-1 accuracy within 30 epochs. The plot above shows no signs of overfitting as\nwell. This means we can train this network for longer (perhaps with a bit more\nregularization) and may obtain even better performance. This performance can further be\nimproved by additional recipes like cosine decay learning rate schedule, other data augmentation\ntechniques like [AutoAugment](https://arxiv.org/abs/1805.09501),\n[MixUp](https://arxiv.org/abs/1710.09412) or\n[Cutmix](https://arxiv.org/abs/1905.04899). With these modifications, the authors present\n95.1% top-1 accuracy on the CIFAR-10 dataset. The authors also present a number of\nexperiments to study how the number of convolution blocks, Transformers layers, etc.\naffect the final performance of CCTs.\n\nFor a comparison, a ViT model takes about **4.7 million** parameters and **100\nepochs** of training to reach a top-1 accuracy of 78.22% on the CIFAR-10 dataset. You can\nrefer to\n[this notebook](https://colab.research.google.com/gist/sayakpaul/1a80d9f582b044354a1a26c5cb3d69e5/image_classification_with_vision_transformer.ipynb)\nto know about the experimental setup.\n\nThe authors also demonstrate the performance of Compact Convolutional Transformers on\nNLP tasks and they report competitive results there.\n'