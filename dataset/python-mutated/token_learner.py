"""
Title: Learning to tokenize in Vision Transformers
Authors: [Aritra Roy Gosthipaty](https://twitter.com/ariG23498), [Sayak Paul](https://twitter.com/RisingSayak) (equal contribution)
Converted to Keras 3 by: [Muhammad Anas Raza](https://anasrz.com)
Date created: 2021/12/10
Last modified: 2023/08/14
Description: Adaptively generating a smaller number of tokens for Vision Transformers.
Accelerator: GPU
"""
'\n## Introduction\n\nVision Transformers ([Dosovitskiy et al.](https://arxiv.org/abs/2010.11929)) and many\nother Transformer-based architectures ([Liu et al.](https://arxiv.org/abs/2103.14030),\n[Yuan et al.](https://arxiv.org/abs/2101.11986), etc.) have shown strong results in\nimage recognition. The following provides a brief overview of the components involved in the\nVision Transformer architecture for image classification:\n\n* Extract small patches from input images.\n* Linearly project those patches.\n* Add positional embeddings to these linear projections.\n* Run these projections through a series of Transformer ([Vaswani et al.](https://arxiv.org/abs/1706.03762))\nblocks.\n* Finally, take the representation from the final Transformer block and add a\nclassification head.\n\nIf we take 224x224 images and extract 16x16 patches, we get a total of 196 patches (also\ncalled tokens) for each image. The number of patches increases as we increase the\nresolution, leading to higher memory footprint. Could we use a reduced\nnumber of patches without having to compromise performance?\nRyoo et al. investigate this question in\n[TokenLearner: Adaptive Space-Time Tokenization for Videos](https://openreview.net/forum?id=z-l1kpDXs88).\nThey introduce a novel module called **TokenLearner** that can help reduce the number\nof patches used by a Vision Transformer (ViT) in an adaptive manner. With TokenLearner\nincorporated in the standard ViT architecture, they are able to reduce the amount of\ncompute (measured in FLOPS) used by the model.\n\nIn this example, we implement the TokenLearner module and demonstrate its\nperformance with a mini ViT and the CIFAR-10 dataset. We make use of the following\nreferences:\n\n* [Official TokenLearner code](https://github.com/google-research/scenic/blob/main/scenic/projects/token_learner/model.py)\n* [Image Classification with ViTs on keras.io](https://keras.io/examples/vision/image_classification_with_vision_transformer/)\n* [TokenLearner slides from NeurIPS 2021](https://nips.cc/media/neurips-2021/Slides/26578.pdf)\n'
'\n## Imports\n'
import keras
from keras import layers
from keras import ops
from tensorflow import data as tf_data
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import math
'\n## Hyperparameters\n\nPlease feel free to change the hyperparameters and check your results. The best way to\ndevelop intuition about the architecture is to experiment with it.\n'
BATCH_SIZE = 256
AUTO = tf_data.AUTOTUNE
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EPOCHS = 20
IMAGE_SIZE = 48
PATCH_SIZE = 6
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
LAYER_NORM_EPS = 1e-06
PROJECTION_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
MLP_UNITS = [PROJECTION_DIM * 2, PROJECTION_DIM]
NUM_TOKENS = 4
'\n## Load and prepare the CIFAR-10 dataset\n'
((x_train, y_train), (x_test, y_test)) = keras.datasets.cifar10.load_data()
((x_train, y_train), (x_val, y_val)) = ((x_train[:40000], y_train[:40000]), (x_train[40000:], y_train[40000:]))
print(f'Training samples: {len(x_train)}')
print(f'Validation samples: {len(x_val)}')
print(f'Testing samples: {len(x_test)}')
train_ds = tf_data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(BATCH_SIZE * 100).batch(BATCH_SIZE).prefetch(AUTO)
val_ds = tf_data.Dataset.from_tensor_slices((x_val, y_val))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTO)
test_ds = tf_data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTO)
'\n## Data augmentation\n\nThe augmentation pipeline consists of:\n\n- Rescaling\n- Resizing\n- Random cropping (fixed-sized or random sized)\n- Random horizontal flipping\n'
data_augmentation = keras.Sequential([layers.Rescaling(1 / 255.0), layers.Resizing(INPUT_SHAPE[0] + 20, INPUT_SHAPE[0] + 20), layers.RandomCrop(IMAGE_SIZE, IMAGE_SIZE), layers.RandomFlip('horizontal')], name='data_augmentation')
'\nNote that image data augmentation layers do not apply data transformations at inference time.\nThis means that when these layers are called with `training=False` they behave differently. Refer\n[to the documentation](https://keras.io/api/layers/preprocessing_layers/image_augmentation/) for more\ndetails.\n'
"\n## Positional embedding module\n\nA [Transformer](https://arxiv.org/abs/1706.03762) architecture consists of **multi-head\nself attention** layers and **fully-connected feed forward** networks (MLP) as the main\ncomponents. Both these components are _permutation invariant_: they're not aware of\nfeature order.\n\nTo overcome this problem we inject tokens with positional information. The\n`position_embedding` function adds this positional information to the linearly projected\ntokens.\n"

class PatchEncoder(layers.Layer):

    def __init__(self, num_patches, projection_dim):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.num_patches = num_patches
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        if False:
            return 10
        positions = ops.expand_dims(ops.arange(start=0, stop=self.num_patches, step=1), axis=0)
        encoded = patch + self.position_embedding(positions)
        return encoded

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        config = super().get_config()
        config.update({'num_patches': self.num_patches})
        return config
'\n## MLP block for Transformer\n\nThis serves as the Fully Connected Feed Forward block for our Transformer.\n'

def mlp(x, dropout_rate, hidden_units):
    if False:
        for i in range(10):
            print('nop')
    for units in hidden_units:
        x = layers.Dense(units, activation=ops.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
'\n## TokenLearner module\n\nThe following figure presents a pictorial overview of the module\n([source](https://ai.googleblog.com/2021/12/improving-vision-transformer-efficiency.html)).\n\n![TokenLearner module GIF](https://blogger.googleusercontent.com/img/a/AVvXsEiylT3_nmd9-tzTnz3g3Vb4eTn-L5sOwtGJOad6t2we7FsjXSpbLDpuPrlInAhtE5hGCA_PfYTJtrIOKfLYLYGcYXVh1Ksfh_C1ZC-C8gw6GKtvrQesKoMrEA_LU_Gd5srl5-3iZDgJc1iyCELoXtfuIXKJ2ADDHOBaUjhU8lXTVdr2E7bCVaFgVHHkmA=w640-h208)\n\nThe TokenLearner module takes as input an image-shaped tensor. It then passes it through\nmultiple single-channel convolutional layers extracting different spatial attention maps\nfocusing on different parts of the input. These attention maps are then element-wise\nmultiplied to the input and result is aggregated with pooling. This pooled output can be\ntrated as a summary of the input and has much lesser number of patches (8, for example)\nthan the original one (196, for example).\n\nUsing multiple convolution layers helps with expressivity. Imposing a form of spatial\nattention helps retain relevant information from the inputs. Both of these components are\ncrucial to make TokenLearner work, especially when we are significantly reducing the number of patches.\n'

def token_learner(inputs, number_of_tokens=NUM_TOKENS):
    if False:
        print('Hello World!')
    x = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(inputs)
    attention_maps = keras.Sequential([layers.Conv2D(filters=number_of_tokens, kernel_size=(3, 3), activation=ops.gelu, padding='same', use_bias=False), layers.Conv2D(filters=number_of_tokens, kernel_size=(3, 3), activation=ops.gelu, padding='same', use_bias=False), layers.Conv2D(filters=number_of_tokens, kernel_size=(3, 3), activation=ops.gelu, padding='same', use_bias=False), layers.Conv2D(filters=number_of_tokens, kernel_size=(3, 3), activation='sigmoid', padding='same', use_bias=False), layers.Reshape((-1, number_of_tokens)), layers.Permute((2, 1))])(x)
    num_filters = inputs.shape[-1]
    inputs = layers.Reshape((1, -1, num_filters))(inputs)
    attended_inputs = ops.expand_dims(attention_maps, axis=-1) * inputs
    outputs = ops.mean(attended_inputs, axis=2)
    return outputs
'\n## Transformer block\n'

def transformer(encoded_patches):
    if False:
        print('Hello World!')
    x1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(encoded_patches)
    attention_output = layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1)(x1, x1)
    x2 = layers.Add()([attention_output, encoded_patches])
    x3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x2)
    x4 = mlp(x3, hidden_units=MLP_UNITS, dropout_rate=0.1)
    encoded_patches = layers.Add()([x4, x2])
    return encoded_patches
'\n## ViT model with the TokenLearner module\n'

def create_vit_classifier(use_token_learner=True, token_learner_units=NUM_TOKENS):
    if False:
        for i in range(10):
            print('nop')
    inputs = layers.Input(shape=INPUT_SHAPE)
    augmented = data_augmentation(inputs)
    projected_patches = layers.Conv2D(filters=PROJECTION_DIM, kernel_size=(PATCH_SIZE, PATCH_SIZE), strides=(PATCH_SIZE, PATCH_SIZE), padding='VALID')(augmented)
    (_, h, w, c) = projected_patches.shape
    projected_patches = layers.Reshape((h * w, c))(projected_patches)
    encoded_patches = PatchEncoder(num_patches=NUM_PATCHES, projection_dim=PROJECTION_DIM)(projected_patches)
    encoded_patches = layers.Dropout(0.1)(encoded_patches)
    for i in range(NUM_LAYERS):
        encoded_patches = transformer(encoded_patches)
        if use_token_learner and i == NUM_LAYERS // 2:
            (_, hh, c) = encoded_patches.shape
            h = int(math.sqrt(hh))
            encoded_patches = layers.Reshape((h, h, c))(encoded_patches)
            encoded_patches = token_learner(encoded_patches, token_learner_units)
    representation = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(representation)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
'\nAs shown in the [TokenLearner paper](https://openreview.net/forum?id=z-l1kpDXs88), it is\nalmost always advantageous to include the TokenLearner module in the middle of the\nnetwork.\n'
'\n## Training utility\n'

def run_experiment(model):
    if False:
        while True:
            i = 10
    optimizer = keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy'), keras.metrics.SparseTopKCategoricalAccuracy(5, name='top-5-accuracy')])
    checkpoint_filepath = '/tmp/checkpoint.weights.h5'
    checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_accuracy', save_best_only=True, save_weights_only=True)
    _ = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[checkpoint_callback])
    model.load_weights(checkpoint_filepath)
    (_, accuracy, top_5_accuracy) = model.evaluate(test_ds)
    print(f'Test accuracy: {round(accuracy * 100, 2)}%')
    print(f'Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%')
'\n## Train and evaluate a ViT with TokenLearner\n'
vit_token_learner = create_vit_classifier()
run_experiment(vit_token_learner)
"\n## Results\n\nWe experimented with and without the TokenLearner inside the mini ViT we implemented\n(with the same hyperparameters presented in this example). Here are our results:\n\n| **TokenLearner** | **# tokens in<br> TokenLearner** | **Top-1 Acc<br>(Averaged across 5 runs)** | **GFLOPs** | **TensorBoard** |\n|:---:|:---:|:---:|:---:|:---:|\n| N | - | 56.112% | 0.0184 | [Link](https://tensorboard.dev/experiment/vkCwM49dQZ2RiK0ZT4mj7w/) |\n| Y | 8 | **56.55%** | **0.0153** | [Link](https://tensorboard.dev/experiment/vkCwM49dQZ2RiK0ZT4mj7w/) |\n| N | - | 56.37% | 0.0184 | [Link](https://tensorboard.dev/experiment/hdyJ4wznQROwqZTgbtmztQ/) |\n| Y | 4 | **56.4980%** | **0.0147** | [Link](https://tensorboard.dev/experiment/hdyJ4wznQROwqZTgbtmztQ/) |\n| N | - (# Transformer layers: 8) | 55.36% | 0.0359 | [Link](https://tensorboard.dev/experiment/sepBK5zNSaOtdCeEG6SV9w/) |\n\nTokenLearner is able to consistently outperform our mini ViT without the module. It is\nalso interesting to notice that it was also able to outperform a deeper version of our\nmini ViT (with 8 layers). The authors also report similar observations in the paper and\nthey attribute this to the adaptiveness of TokenLearner.\n\nOne should also note that the FLOPs count **decreases** considerably with the addition of\nthe TokenLearner module. With less FLOPs count the TokenLearner module is able to\ndeliver better results. This aligns very well with the authors' findings.\n\nAdditionally, the authors [introduced](https://github.com/google-research/scenic/blob/main/scenic/projects/token_learner/model.py#L104)\na newer version of the TokenLearner for smaller training data regimes. Quoting the authors:\n\n> Instead of using 4 conv. layers with small channels to implement spatial attention,\n  this version uses 2 grouped conv. layers with more channels. It also uses softmax\n  instead of sigmoid. We confirmed that this version works better when having limited\n  training data, such as training with ImageNet1K from scratch.\n\nWe experimented with this module and in the following table we summarize the results:\n\n| **# Groups** | **# Tokens** | **Top-1 Acc** | **GFLOPs** | **TensorBoard** |\n|:---:|:---:|:---:|:---:|:---:|\n| 4 | 4 | 54.638% | 0.0149 | [Link](https://tensorboard.dev/experiment/KmfkGqAGQjikEw85phySmw/) |\n| 8 | 8 | 54.898% | 0.0146 | [Link](https://tensorboard.dev/experiment/0PpgYOq9RFWV9njX6NJQ2w/) |\n| 4 | 8 | 55.196% | 0.0149 | [Link](https://tensorboard.dev/experiment/WUkrHbZASdu3zrfmY4ETZg/) |\n\nPlease note that we used the same hyperparameters presented in this example. Our\nimplementation is available\n[in this notebook](https://github.com/ariG23498/TokenLearner/blob/master/TokenLearner-V1.1.ipynb).\nWe acknowledge that the results with this new TokenLearner module are slightly off\nthan expected and this might mitigate with hyperparameter tuning.\n\n*Note*: To compute the FLOPs of our models we used\n[this utility](https://github.com/AdityaKane2001/regnety/blob/main/regnety/utils/model_utils.py#L27)\nfrom [this repository](https://github.com/AdityaKane2001/regnety).\n"
'\n## Number of parameters\n\nYou may have noticed that adding the TokenLearner module increases the number of\nparameters of the base network. But that does not mean it is less efficient as shown by\n[Dehghani et al.](https://arxiv.org/abs/2110.12894). Similar findings were reported\nby [Bello et al.](https://arxiv.org/abs/2103.07579) as well. The TokenLearner module\nhelps reducing the FLOPS in the overall network thereby helping to reduce the memory\nfootprint.\n'
'\n## Final notes\n\n* TokenFuser: The authors of the paper also propose another module named TokenFuser. This\nmodule helps in remapping the representation of the TokenLearner output back to its\noriginal spatial resolution. To reuse the TokenLearner in the ViT architecture, the\nTokenFuser is a must. We first learn the tokens from the TokenLearner, build a\nrepresentation of the tokens from a Transformer layer and then remap the representation\ninto the original spatial resolution, so that it can again be consumed by a TokenLearner.\nNote here that you can only use the TokenLearner module once in entire ViT model if not\npaired with the TokenFuser.\n* Use of these modules for video: The authors also suggest that TokenFuser goes really\nwell with Vision Transformers for Videos ([Arnab et al.](https://arxiv.org/abs/2103.15691)).\n\nWe are grateful to [JarvisLabs](https://jarvislabs.ai/) and\n[Google Developers Experts](https://developers.google.com/programs/experts/)\nprogram for helping with GPU credits. Also, we are thankful to Michael Ryoo (first\nauthor of TokenLearner) for fruitful discussions.\n\n| Trained Model | Demo |\n| :--: | :--: |\n| [![Generic badge](https://img.shields.io/badge/🤗%20Model-TokenLearner-black.svg)](https://huggingface.co/keras-io/learning_to_tokenize_in_ViT) | [![Generic badge](https://img.shields.io/badge/🤗%20Spaces-TokenLearner-black.svg)](https://huggingface.co/spaces/keras-io/token_learner) |\n'