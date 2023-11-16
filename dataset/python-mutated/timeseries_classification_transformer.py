"""
Title: Timeseries classification with a Transformer model
Author: [Theodoros Ntakouris](https://github.com/ntakouris)
Date created: 2021/06/25
Last modified: 2021/08/05
Description: This notebook demonstrates how to do timeseries classification using a Transformer model.
Accelerator: GPU
"""
'\n## Introduction\n\nThis is the Transformer architecture from\n[Attention Is All You Need](https://arxiv.org/abs/1706.03762),\napplied to timeseries instead of natural language.\n\nThis example requires TensorFlow 2.4 or higher.\n\n## Load the dataset\n\nWe are going to use the same dataset and preprocessing as the\n[TimeSeries Classification from Scratch](https://keras.io/examples/timeseries/timeseries_classification_from_scratch)\nexample.\n'
import numpy as np

def readucr(filename):
    if False:
        print('Hello World!')
    data = np.loadtxt(filename, delimiter='\t')
    y = data[:, 0]
    x = data[:, 1:]
    return (x, y.astype(int))
root_url = 'https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/'
(x_train, y_train) = readucr(root_url + 'FordA_TRAIN.tsv')
(x_test, y_test) = readucr(root_url + 'FordA_TEST.tsv')
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
n_classes = len(np.unique(y_train))
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]
y_train[y_train == -1] = 0
y_test[y_test == -1] = 0
'\n## Build the model\n\nOur model processes a tensor of shape `(batch size, sequence length, features)`,\nwhere `sequence length` is the number of time steps and `features` is each input\ntimeseries.\n\nYou can replace your classification RNN layers with this one: the\ninputs are fully compatible!\n'
import keras
from keras import layers
'\nWe include residual connections, layer normalization, and dropout.\nThe resulting layer can be stacked multiple times.\n\nThe projection layers are implemented through `keras.layers.Conv1D`.\n'

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    if False:
        print('Hello World!')
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-06)(x)
    res = x + inputs
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-06)(x)
    return x + res
'\nThe main part of our model is now complete. We can stack multiple of those\n`transformer_encoder` blocks and we can also proceed to add the final\nMulti-Layer Perceptron classification head. Apart from a stack of `Dense`\nlayers, we need to reduce the output tensor of the `TransformerEncoder` part of\nour model down to a vector of features for each data point in the current\nbatch. A common way to achieve this is to use a pooling layer. For\nthis example, a `GlobalAveragePooling1D` layer is sufficient.\n'

def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    if False:
        return 10
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = layers.GlobalAveragePooling1D(data_format='channels_first')(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation='relu')(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)
'\n## Train and evaluate\n'
input_shape = x_train.shape[1:]
model = build_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], mlp_dropout=0.4, dropout=0.25)
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=['sparse_categorical_accuracy'])
model.summary()
callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
model.fit(x_train, y_train, validation_split=0.2, epochs=2, batch_size=64, callbacks=callbacks)
model.evaluate(x_test, y_test, verbose=1)
'\n## Conclusions\n\nIn about 110-120 epochs (25s each on Colab), the model reaches a training\naccuracy of ~0.95, validation accuracy of ~84 and a testing\naccuracy of ~85, without hyperparameter tuning. And that is for a model\nwith less than 100k parameters. Of course, parameter count and accuracy could be\nimproved by a hyperparameter search and a more sophisticated learning rate\nschedule, or a different optimizer.\n\nYou can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/timeseries_transformer_classification)\nand try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/timeseries_transformer_classification).\n'