"""
Title: Structured data learning with TabTransformer
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Date created: 2022/01/18
Last modified: 2022/01/18
Description: Using contextual embeddings for structured data classification.
Accelerator: GPU
"""
'\n## Introduction\n\nThis example demonstrates how to do structured data classification using\n[TabTransformer](https://arxiv.org/abs/2012.06678), a deep tabular data modeling\narchitecture for supervised and semi-supervised learning.\nThe TabTransformer is built upon self-attention based Transformers.\nThe Transformer layers transform the embeddings of categorical features\ninto robust contextual embeddings to achieve higher predictive accuracy.\n\n\n\n## Setup\n'
import keras
from keras import layers
from keras import ops
import math
import numpy as np
import pandas as pd
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
from functools import partial
"\n## Prepare the data\n\nThis example uses the\n[United States Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/census+income)\nprovided by the\n[UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).\nThe task is binary classification\nto predict whether a person is likely to be making over USD 50,000 a year.\n\nThe dataset includes 48,842 instances with 14 input features: 5 numerical features and 9 categorical features.\n\nFirst, let's load the dataset from the UCI Machine Learning Repository into a Pandas\nDataFrame:\n"
CSV_HEADER = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'gender', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income_bracket']
train_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
train_data = pd.read_csv(train_data_url, header=None, names=CSV_HEADER)
test_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
test_data = pd.read_csv(test_data_url, header=None, names=CSV_HEADER)
print(f'Train dataset shape: {train_data.shape}')
print(f'Test dataset shape: {test_data.shape}')
"\nRemove the first record (because it is not a valid data example) and a trailing 'dot' in the class labels.\n"
test_data = test_data[1:]
test_data.income_bracket = test_data.income_bracket.apply(lambda value: value.replace('.', ''))
'\nNow we store the training and test data in separate CSV files.\n'
train_data_file = 'train_data.csv'
test_data_file = 'test_data.csv'
train_data.to_csv(train_data_file, index=False, header=False)
test_data.to_csv(test_data_file, index=False, header=False)
'\n## Define dataset metadata\n\nHere, we define the metadata of the dataset that will be useful for reading and parsing\nthe data into input features, and encoding the input features with respect to their types.\n'
NUMERIC_FEATURE_NAMES = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
CATEGORICAL_FEATURES_WITH_VOCABULARY = {'workclass': sorted(list(train_data['workclass'].unique())), 'education': sorted(list(train_data['education'].unique())), 'marital_status': sorted(list(train_data['marital_status'].unique())), 'occupation': sorted(list(train_data['occupation'].unique())), 'relationship': sorted(list(train_data['relationship'].unique())), 'race': sorted(list(train_data['race'].unique())), 'gender': sorted(list(train_data['gender'].unique())), 'native_country': sorted(list(train_data['native_country'].unique()))}
WEIGHT_COLUMN_NAME = 'fnlwgt'
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
COLUMN_DEFAULTS = [[0.0] if feature_name in NUMERIC_FEATURE_NAMES + [WEIGHT_COLUMN_NAME] else ['NA'] for feature_name in CSV_HEADER]
TARGET_FEATURE_NAME = 'income_bracket'
TARGET_LABELS = [' <=50K', ' >50K']
'\n## Configure the hyperparameters\n\nThe hyperparameters includes model architecture and training configurations.\n'
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
DROPOUT_RATE = 0.2
BATCH_SIZE = 265
NUM_EPOCHS = 15
NUM_TRANSFORMER_BLOCKS = 3
NUM_HEADS = 4
EMBEDDING_DIMS = 16
MLP_HIDDEN_UNITS_FACTORS = [2, 1]
NUM_MLP_BLOCKS = 2
'\n## Implement data reading pipeline\n\nWe define an input function that reads and parses the file, then converts features\nand labels into a[`tf.data.Dataset`](https://www.tensorflow.org/guide/datasets)\nfor training or evaluation.\n'
target_label_lookup = layers.StringLookup(vocabulary=TARGET_LABELS, mask_token=None, num_oov_indices=0)

def prepare_example(features, target):
    if False:
        i = 10
        return i + 15
    target_index = target_label_lookup(target)
    weights = features.pop(WEIGHT_COLUMN_NAME)
    return (features, target_index, weights)
lookup_dict = {}
for feature_name in CATEGORICAL_FEATURE_NAMES:
    vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
    lookup = layers.StringLookup(vocabulary=vocabulary, mask_token=None, num_oov_indices=0)
    lookup_dict[feature_name] = lookup

def encode_categorical(batch_x, batch_y, weights):
    if False:
        for i in range(10):
            print('nop')
    for feature_name in CATEGORICAL_FEATURE_NAMES:
        batch_x[feature_name] = lookup_dict[feature_name](batch_x[feature_name])
    return (batch_x, batch_y, weights)

def get_dataset_from_csv(csv_file_path, batch_size=128, shuffle=False):
    if False:
        i = 10
        return i + 15
    dataset = tf.data.experimental.make_csv_dataset(csv_file_path, batch_size=batch_size, column_names=CSV_HEADER, column_defaults=COLUMN_DEFAULTS, label_name=TARGET_FEATURE_NAME, num_epochs=1, header=False, na_value='?', shuffle=shuffle).map(prepare_example, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).map(encode_categorical)
    return dataset.cache()
'\n## Implement a training and evaluation procedure\n'

def run_experiment(model, train_data_file, test_data_file, num_epochs, learning_rate, weight_decay, batch_size):
    if False:
        return 10
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy(name='accuracy')])
    train_dataset = get_dataset_from_csv(train_data_file, batch_size, shuffle=True)
    validation_dataset = get_dataset_from_csv(test_data_file, batch_size)
    print('Start training the model...')
    history = model.fit(train_dataset, epochs=num_epochs, validation_data=validation_dataset)
    print('Model training finished')
    (_, accuracy) = model.evaluate(validation_dataset, verbose=0)
    print(f'Validation accuracy: {round(accuracy * 100, 2)}%')
    return history
'\n## Create model inputs\n\nNow, define the inputs for the models as a dictionary, where the key is the feature name,\nand the value is a `keras.layers.Input` tensor with the corresponding feature shape\nand data type.\n'

def create_model_inputs():
    if False:
        for i in range(10):
            print('nop')
    inputs = {}
    for feature_name in FEATURE_NAMES:
        if feature_name in NUMERIC_FEATURE_NAMES:
            inputs[feature_name] = layers.Input(name=feature_name, shape=(), dtype='float32')
        else:
            inputs[feature_name] = layers.Input(name=feature_name, shape=(), dtype='float32')
    return inputs
'\n## Encode features\n\nThe `encode_inputs` method returns `encoded_categorical_feature_list` and `numerical_feature_list`.\nWe encode the categorical features as embeddings, using a fixed `embedding_dims` for all the features,\nregardless their vocabulary sizes. This is required for the Transformer model.\n'

def encode_inputs(inputs, embedding_dims):
    if False:
        while True:
            i = 10
    encoded_categorical_feature_list = []
    numerical_feature_list = []
    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURE_NAMES:
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            embedding = layers.Embedding(input_dim=len(vocabulary), output_dim=embedding_dims)
            encoded_categorical_feature = embedding(inputs[feature_name])
            encoded_categorical_feature_list.append(encoded_categorical_feature)
        else:
            numerical_feature = ops.expand_dims(inputs[feature_name], -1)
            numerical_feature_list.append(numerical_feature)
    return (encoded_categorical_feature_list, numerical_feature_list)
'\n## Implement an MLP block\n'

def create_mlp(hidden_units, dropout_rate, activation, normalization_layer, name=None):
    if False:
        i = 10
        return i + 15
    mlp_layers = []
    for units in hidden_units:
        (mlp_layers.append(normalization_layer()),)
        mlp_layers.append(layers.Dense(units, activation=activation))
        mlp_layers.append(layers.Dropout(dropout_rate))
    return keras.Sequential(mlp_layers, name=name)
'\n## Experiment 1: a baseline model\n\nIn the first experiment, we create a simple multi-layer feed-forward network.\n'

def create_baseline_model(embedding_dims, num_mlp_blocks, mlp_hidden_units_factors, dropout_rate):
    if False:
        for i in range(10):
            print('nop')
    inputs = create_model_inputs()
    (encoded_categorical_feature_list, numerical_feature_list) = encode_inputs(inputs, embedding_dims)
    features = layers.concatenate(encoded_categorical_feature_list + numerical_feature_list)
    feedforward_units = [features.shape[-1]]
    for layer_idx in range(num_mlp_blocks):
        features = create_mlp(hidden_units=feedforward_units, dropout_rate=dropout_rate, activation=keras.activations.gelu, normalization_layer=layers.LayerNormalization, name=f'feedforward_{layer_idx}')(features)
    mlp_hidden_units = [factor * features.shape[-1] for factor in mlp_hidden_units_factors]
    features = create_mlp(hidden_units=mlp_hidden_units, dropout_rate=dropout_rate, activation=keras.activations.selu, normalization_layer=layers.BatchNormalization, name='MLP')(features)
    outputs = layers.Dense(units=1, activation='sigmoid', name='sigmoid')(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
baseline_model = create_baseline_model(embedding_dims=EMBEDDING_DIMS, num_mlp_blocks=NUM_MLP_BLOCKS, mlp_hidden_units_factors=MLP_HIDDEN_UNITS_FACTORS, dropout_rate=DROPOUT_RATE)
print('Total model weights:', baseline_model.count_params())
keras.utils.plot_model(baseline_model, show_shapes=True, rankdir='LR')
"\nLet's train and evaluate the baseline model:\n"
history = run_experiment(model=baseline_model, train_data_file=train_data_file, test_data_file=test_data_file, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE)
'\nThe baseline linear model achieves ~81% validation accuracy.\n'
'\n## Experiment 2: TabTransformer\n\nThe TabTransformer architecture works as follows:\n\n1. All the categorical features are encoded as embeddings, using the same `embedding_dims`.\nThis means that each value in each categorical feature will have its own embedding vector.\n2. A column embedding, one embedding vector for each categorical feature, is added (point-wise) to the categorical feature embedding.\n3. The embedded categorical features are fed into a stack of Transformer blocks.\nEach Transformer block consists of a multi-head self-attention layer followed by a feed-forward layer.\n3. The outputs of the final Transformer layer, which are the *contextual embeddings* of the categorical features,\nare concatenated with the input numerical features, and fed into a final MLP block.\n4. A `softmax` classifer is applied at the end of the model.\n\nThe [paper](https://arxiv.org/abs/2012.06678) discusses both addition and concatenation of the column embedding in the\n*Appendix: Experiment and Model Details* section.\nThe architecture of TabTransformer is shown below, as presented in the paper.\n\n<img src="https://raw.githubusercontent.com/keras-team/keras-io/master/examples/structured_data/img/tabtransformer/tabtransformer.png" width="500"/>\n'

def create_tabtransformer_classifier(num_transformer_blocks, num_heads, embedding_dims, mlp_hidden_units_factors, dropout_rate, use_column_embedding=False):
    if False:
        return 10
    inputs = create_model_inputs()
    (encoded_categorical_feature_list, numerical_feature_list) = encode_inputs(inputs, embedding_dims)
    encoded_categorical_features = ops.stack(encoded_categorical_feature_list, axis=1)
    numerical_features = layers.concatenate(numerical_feature_list)
    if use_column_embedding:
        num_columns = encoded_categorical_features.shape[1]
        column_embedding = layers.Embedding(input_dim=num_columns, output_dim=embedding_dims)
        column_indices = ops.arange(start=0, stop=num_columns, step=1)
        encoded_categorical_features = encoded_categorical_features + column_embedding(column_indices)
    for block_idx in range(num_transformer_blocks):
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dims, dropout=dropout_rate, name=f'multihead_attention_{block_idx}')(encoded_categorical_features, encoded_categorical_features)
        x = layers.Add(name=f'skip_connection1_{block_idx}')([attention_output, encoded_categorical_features])
        x = layers.LayerNormalization(name=f'layer_norm1_{block_idx}', epsilon=1e-06)(x)
        feedforward_output = create_mlp(hidden_units=[embedding_dims], dropout_rate=dropout_rate, activation=keras.activations.gelu, normalization_layer=partial(layers.LayerNormalization, epsilon=1e-06), name=f'feedforward_{block_idx}')(x)
        x = layers.Add(name=f'skip_connection2_{block_idx}')([feedforward_output, x])
        encoded_categorical_features = layers.LayerNormalization(name=f'layer_norm2_{block_idx}', epsilon=1e-06)(x)
    categorical_features = layers.Flatten()(encoded_categorical_features)
    numerical_features = layers.LayerNormalization(epsilon=1e-06)(numerical_features)
    features = layers.concatenate([categorical_features, numerical_features])
    mlp_hidden_units = [factor * features.shape[-1] for factor in mlp_hidden_units_factors]
    features = create_mlp(hidden_units=mlp_hidden_units, dropout_rate=dropout_rate, activation=keras.activations.selu, normalization_layer=layers.BatchNormalization, name='MLP')(features)
    outputs = layers.Dense(units=1, activation='sigmoid', name='sigmoid')(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
tabtransformer_model = create_tabtransformer_classifier(num_transformer_blocks=NUM_TRANSFORMER_BLOCKS, num_heads=NUM_HEADS, embedding_dims=EMBEDDING_DIMS, mlp_hidden_units_factors=MLP_HIDDEN_UNITS_FACTORS, dropout_rate=DROPOUT_RATE)
print('Total model weights:', tabtransformer_model.count_params())
keras.utils.plot_model(tabtransformer_model, show_shapes=True, rankdir='LR')
"\nLet's train and evaluate the TabTransformer model:\n"
history = run_experiment(model=tabtransformer_model, train_data_file=train_data_file, test_data_file=test_data_file, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE)
'\nThe TabTransformer model achieves ~85% validation accuracy.\nNote that, with the default parameter configurations, both the baseline and the TabTransformer\nhave similar number of trainable weights: 109,629 and 92,151 respectively, and both use the same training hyperparameters.\n'
'\n## Conclusion\n\nTabTransformer significantly outperforms MLP and recent\ndeep networks for tabular data while matching the performance of tree-based ensemble models.\nTabTransformer can be learned in end-to-end supervised training using labeled examples.\nFor a scenario where there are a few labeled examples and a large number of unlabeled\nexamples, a pre-training procedure can be employed to train the Transformer layers using unlabeled data.\nThis is followed by fine-tuning of the pre-trained Transformer layers along with\nthe top MLP layer using the labeled data.\n\nExample available on HuggingFace.\n\n| Trained Model | Demo |\n| :--: | :--: |\n| [![Generic badge](https://img.shields.io/badge/🤗%20Model-TabTransformer-black.svg)](https://huggingface.co/keras-io/tab_transformer) | [![Generic badge](https://img.shields.io/badge/🤗%20Spaces-TabTransformer-black.svg)](https://huggingface.co/spaces/keras-io/TabTransformer_Classification) |\n\n'