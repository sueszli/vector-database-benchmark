"""
Title: Image classification with EANet (External Attention Transformer)
Author: [ZhiYong Chang](https://github.com/czy00000)
Converted to Keras 3: [Muhammad Anas Raza](https://anasrz.com)
Date created: 2021/10/19
Last modified: 2023/07/18
Description: Image classification with a Transformer that leverages external attention.
Accelerator: GPU
"""
'\n## Introduction\n\nThis example implements the [EANet](https://arxiv.org/abs/2105.02358)\nmodel for image classification, and demonstrates it on the CIFAR-100 dataset.\nEANet introduces a novel attention mechanism\nnamed ***external attention***, based on two external, small, learnable, and\nshared memories, which can be implemented easily by simply using two cascaded\nlinear layers and two normalization layers. It conveniently replaces self-attention\nas used in existing architectures. External attention has linear complexity, as it only\nimplicitly considers the correlations between all samples.\n'
'\n## Setup\n'
import keras
from keras import layers
from keras import ops
import matplotlib.pyplot as plt
'\n## Prepare the data\n'
num_classes = 100
input_shape = (32, 32, 3)
((x_train, y_train), (x_test, y_test)) = keras.datasets.cifar100.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(f'x_train shape: {x_train.shape} - y_train shape: {y_train.shape}')
print(f'x_test shape: {x_test.shape} - y_test shape: {y_test.shape}')
'\n## Configure the hyperparameters\n'
weight_decay = 0.0001
learning_rate = 0.001
label_smoothing = 0.1
validation_split = 0.2
batch_size = 128
num_epochs = 50
patch_size = 2
num_patches = (input_shape[0] // patch_size) ** 2
embedding_dim = 64
mlp_dim = 64
dim_coefficient = 4
num_heads = 4
attention_dropout = 0.2
projection_dropout = 0.2
num_transformer_blocks = 8
print(f'Patch size: {patch_size} X {patch_size} = {patch_size ** 2} ')
print(f'Patches per image: {num_patches}')
'\n## Use data augmentation\n'
data_augmentation = keras.Sequential([layers.Normalization(), layers.RandomFlip('horizontal'), layers.RandomRotation(factor=0.1), layers.RandomContrast(factor=0.1), layers.RandomZoom(height_factor=0.2, width_factor=0.2)], name='data_augmentation')
data_augmentation.layers[0].adapt(x_train)
'\n## Implement the patch extraction and encoding layer\n'

class PatchExtract(layers.Layer):

    def __init__(self, patch_size, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, x):
        if False:
            while True:
                i = 10
        (B, C) = (ops.shape(x)[0], ops.shape(x)[-1])
        x = ops.image.extract_patches(x, self.patch_size)
        x = ops.reshape(x, (B, -1, self.patch_size * self.patch_size * C))
        return x

class PatchEmbedding(layers.Layer):

    def __init__(self, num_patch, embed_dim, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        if False:
            return 10
        pos = ops.arange(start=0, stop=self.num_patch, step=1)
        return self.proj(patch) + self.pos_embed(pos)
'\n## Implement the external attention block\n'

def external_attention(x, dim, num_heads, dim_coefficient=4, attention_dropout=0, projection_dropout=0):
    if False:
        i = 10
        return i + 15
    (_, num_patch, channel) = x.shape
    assert dim % num_heads == 0
    num_heads = num_heads * dim_coefficient
    x = layers.Dense(dim * dim_coefficient)(x)
    x = ops.reshape(x, (-1, num_patch, num_heads, dim * dim_coefficient // num_heads))
    x = ops.transpose(x, axes=[0, 2, 1, 3])
    attn = layers.Dense(dim // dim_coefficient)(x)
    attn = layers.Softmax(axis=2)(attn)
    attn = layers.Lambda(lambda attn: ops.divide(attn, ops.convert_to_tensor(1e-09) + ops.sum(attn, axis=-1, keepdims=True)))(attn)
    attn = layers.Dropout(attention_dropout)(attn)
    x = layers.Dense(dim * dim_coefficient // num_heads)(attn)
    x = ops.transpose(x, axes=[0, 2, 1, 3])
    x = ops.reshape(x, [-1, num_patch, dim * dim_coefficient])
    x = layers.Dense(dim)(x)
    x = layers.Dropout(projection_dropout)(x)
    return x
'\n## Implement the MLP block\n'

def mlp(x, embedding_dim, mlp_dim, drop_rate=0.2):
    if False:
        while True:
            i = 10
    x = layers.Dense(mlp_dim, activation=ops.gelu)(x)
    x = layers.Dropout(drop_rate)(x)
    x = layers.Dense(embedding_dim)(x)
    x = layers.Dropout(drop_rate)(x)
    return x
'\n## Implement the Transformer block\n'

def transformer_encoder(x, embedding_dim, mlp_dim, num_heads, dim_coefficient, attention_dropout, projection_dropout, attention_type='external_attention'):
    if False:
        for i in range(10):
            print('nop')
    residual_1 = x
    x = layers.LayerNormalization(epsilon=1e-05)(x)
    if attention_type == 'external_attention':
        x = external_attention(x, embedding_dim, num_heads, dim_coefficient, attention_dropout, projection_dropout)
    elif attention_type == 'self_attention':
        x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim, dropout=attention_dropout)(x, x)
    x = layers.add([x, residual_1])
    residual_2 = x
    x = layers.LayerNormalization(epsilon=1e-05)(x)
    x = mlp(x, embedding_dim, mlp_dim)
    x = layers.add([x, residual_2])
    return x
'\n## Implement the EANet model\n'
'\nThe EANet model leverages external attention.\nThe computational complexity of traditional self attention is `O(d * N ** 2)`,\nwhere `d` is the embedding size, and `N` is the number of patch.\nthe authors find that most pixels are closely related to just a few other\npixels, and an `N`-to-`N` attention matrix may be redundant.\nSo, they propose as an alternative an external\nattention module where the computational complexity of external attention is `O(d * S * N)`.\nAs `d` and `S` are hyper-parameters,\nthe proposed algorithm is linear in the number of pixels. In fact, this is equivalent\nto a drop patch operation, because a lot of information contained in a patch\nin an image is redundant and unimportant.\n'

def get_model(attention_type='external_attention'):
    if False:
        i = 10
        return i + 15
    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = PatchExtract(patch_size)(x)
    x = PatchEmbedding(num_patches, embedding_dim)(x)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, embedding_dim, mlp_dim, num_heads, dim_coefficient, attention_dropout, projection_dropout, attention_type)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
'\n## Train on CIFAR-100\n\n'
model = get_model(attention_type='external_attention')
model.compile(loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing), optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay), metrics=[keras.metrics.CategoricalAccuracy(name='accuracy'), keras.metrics.TopKCategoricalAccuracy(5, name='top-5-accuracy')])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_split=validation_split)
"\n### Let's visualize the training progress of the model.\n\n"
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation Losses Over Epochs', fontsize=14)
plt.legend()
plt.grid()
plt.show()
"\n### Let's display the final results of the test on CIFAR-100.\n\n"
(loss, accuracy, top_5_accuracy) = model.evaluate(x_test, y_test)
print(f'Test loss: {round(loss, 2)}')
print(f'Test accuracy: {round(accuracy * 100, 2)}%')
print(f'Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%')
'\nEANet just replaces self attention in Vit with external attention.\nThe traditional Vit achieved a ~73% test top-5 accuracy and ~41 top-1 accuracy after\ntraining 50 epochs, but with 0.6M parameters. Under the same experimental environment\nand the same hyperparameters, The EANet model we just trained has just 0.3M parameters,\nand it gets us to ~73% test top-5 accuracy and ~43% top-1 accuracy. This fully demonstrates the\neffectiveness of external attention.\n\nWe only show the training\nprocess of EANet, you can train Vit under the same experimental conditions and observe\nthe test results.\n'