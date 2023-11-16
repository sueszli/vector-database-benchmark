"""
Title: Classification with Neural Decision Forests
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Date created: 2021/01/15
Last modified: 2021/01/15
Description: How to train differentiable decision trees for end-to-end learning in deep neural networks.
Accelerator: GPU
"""
'\n## Introduction\n\nThis example provides an implementation of the\n[Deep Neural Decision Forest](https://ieeexplore.ieee.org/document/7410529)\nmodel introduced by P. Kontschieder et al. for structured data classification.\nIt demonstrates how to build a stochastic and differentiable decision tree model,\ntrain it end-to-end, and unify decision trees with deep representation learning.\n\n## The dataset\n\nThis example uses the\n[United States Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/census+income)\nprovided by the\n[UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).\nThe task is binary classification\nto predict whether a person is likely to be making over USD 50,000 a year.\n\nThe dataset includes 48,842 instances with 14 input features (such as age, work class, education, occupation, and so on): 5 numerical features\nand 9 categorical features.\n'
'\n## Setup\n'
import keras
from keras import layers
from keras.layers import StringLookup
from keras import ops
from tensorflow import data as tf_data
import numpy as np
import pandas as pd
import math
_dtype = 'float32'
'\n## Prepare the data\n'
CSV_HEADER = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'gender', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income_bracket']
train_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
train_data = pd.read_csv(train_data_url, header=None, names=CSV_HEADER)
test_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
test_data = pd.read_csv(test_data_url, header=None, names=CSV_HEADER)
print(f'Train dataset shape: {train_data.shape}')
print(f'Test dataset shape: {test_data.shape}')
"\nRemove the first record (because it is not a valid data example) and a trailing\n'dot' in the class labels.\n"
test_data = test_data[1:]
test_data.income_bracket = test_data.income_bracket.apply(lambda value: value.replace('.', ''))
'\nWe store the training and test data splits locally as CSV files.\n'
train_data_file = 'train_data.csv'
test_data_file = 'test_data.csv'
train_data.to_csv(train_data_file, index=False, header=False)
test_data.to_csv(test_data_file, index=False, header=False)
'\n## Define dataset metadata\n\nHere, we define the metadata of the dataset that will be useful for reading and parsing\nand encoding input features.\n'
NUMERIC_FEATURE_NAMES = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
CATEGORICAL_FEATURES_WITH_VOCABULARY = {'workclass': sorted(list(train_data['workclass'].unique())), 'education': sorted(list(train_data['education'].unique())), 'marital_status': sorted(list(train_data['marital_status'].unique())), 'occupation': sorted(list(train_data['occupation'].unique())), 'relationship': sorted(list(train_data['relationship'].unique())), 'race': sorted(list(train_data['race'].unique())), 'gender': sorted(list(train_data['gender'].unique())), 'native_country': sorted(list(train_data['native_country'].unique()))}
IGNORE_COLUMN_NAMES = ['fnlwgt']
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
COLUMN_DEFAULTS = [[0.0] if feature_name in NUMERIC_FEATURE_NAMES + IGNORE_COLUMN_NAMES else ['NA'] for feature_name in CSV_HEADER]
TARGET_FEATURE_NAME = 'income_bracket'
TARGET_LABELS = [' <=50K', ' >50K']
'\n## Create `tf_data.Dataset` objects for training and validation\n\nWe create an input function to read and parse the file, and convert features and labels\ninto a [`tf_data.Dataset`](https://www.tensorflow.org/guide/datasets)\nfor training and validation. We also preprocess the input by mapping the target label\nto an index.\n'
target_label_lookup = StringLookup(vocabulary=TARGET_LABELS, mask_token=None, num_oov_indices=0)
lookup_dict = {}
for feature_name in CATEGORICAL_FEATURE_NAMES:
    vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
    lookup = StringLookup(vocabulary=vocabulary, mask_token=None, num_oov_indices=0)
    lookup_dict[feature_name] = lookup

def encode_categorical(batch_x, batch_y):
    if False:
        print('Hello World!')
    for feature_name in CATEGORICAL_FEATURE_NAMES:
        batch_x[feature_name] = lookup_dict[feature_name](batch_x[feature_name])
    return (batch_x, batch_y)

def get_dataset_from_csv(csv_file_path, shuffle=False, batch_size=128):
    if False:
        while True:
            i = 10
    dataset = tf_data.experimental.make_csv_dataset(csv_file_path, batch_size=batch_size, column_names=CSV_HEADER, column_defaults=COLUMN_DEFAULTS, label_name=TARGET_FEATURE_NAME, num_epochs=1, header=False, na_value='?', shuffle=shuffle).map(lambda features, target: (features, target_label_lookup(target))).map(encode_categorical)
    return dataset.cache()
'\n## Create model inputs\n'

def create_model_inputs():
    if False:
        while True:
            i = 10
    inputs = {}
    for feature_name in FEATURE_NAMES:
        if feature_name in NUMERIC_FEATURE_NAMES:
            inputs[feature_name] = layers.Input(name=feature_name, shape=(), dtype=_dtype)
        else:
            inputs[feature_name] = layers.Input(name=feature_name, shape=(), dtype='int32')
    return inputs
'\n## Encode input features\n'

def encode_inputs(inputs):
    if False:
        i = 10
        return i + 15
    encoded_features = []
    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURE_NAMES:
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            value_index = inputs[feature_name]
            embedding_dims = int(math.sqrt(lookup.vocabulary_size()))
            embedding = layers.Embedding(input_dim=lookup.vocabulary_size(), output_dim=embedding_dims)
            encoded_feature = embedding(value_index)
        else:
            encoded_feature = inputs[feature_name]
            if inputs[feature_name].shape[-1] is None:
                encoded_feature = keras.ops.expand_dims(encoded_feature, -1)
        encoded_features.append(encoded_feature)
    encoded_features = layers.concatenate(encoded_features)
    return encoded_features
'\n## Deep Neural Decision Tree\n\nA neural decision tree model has two sets of weights to learn. The first set is `pi`,\nwhich represents the probability distribution of the classes in the tree leaves.\nThe second set is the weights of the routing layer `decision_fn`, which represents the probability\nof going to each leave. The forward pass of the model works as follows:\n\n1. The model expects input `features` as a single vector encoding all the features of an instance\nin the batch. This vector can be generated from a Convolution Neural Network (CNN) applied to images\nor dense transformations applied to structured data features.\n2. The model first applies a `used_features_mask` to randomly select a subset of input features to use.\n3. Then, the model computes the probabilities (`mu`) for the input instances to reach the tree leaves\nby iteratively performing a *stochastic* routing throughout the tree levels.\n4. Finally, the probabilities of reaching the leaves are combined by the class probabilities at the\nleaves to produce the final `outputs`.\n'

class NeuralDecisionTree(keras.Model):

    def __init__(self, depth, num_features, used_features_rate, num_classes):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.depth = depth
        self.num_leaves = 2 ** depth
        self.num_classes = num_classes
        num_used_features = int(num_features * used_features_rate)
        one_hot = np.eye(num_features)
        sampled_feature_indices = np.random.choice(np.arange(num_features), num_used_features, replace=False)
        self.used_features_mask = ops.convert_to_tensor(one_hot[sampled_feature_indices], dtype=_dtype)
        self.pi = self.add_weight(initializer='random_normal', shape=[self.num_leaves, self.num_classes], dtype=_dtype, trainable=True)
        self.decision_fn = layers.Dense(units=self.num_leaves, activation='sigmoid', name='decision')

    def call(self, features):
        if False:
            i = 10
            return i + 15
        batch_size = ops.shape(features)[0]
        features = ops.matmul(features, ops.transpose(self.used_features_mask))
        decisions = ops.expand_dims(self.decision_fn(features), axis=2)
        decisions = layers.concatenate([decisions, 1 - decisions], axis=2)
        mu = ops.ones([batch_size, 1, 1])
        begin_idx = 1
        end_idx = 2
        for level in range(self.depth):
            mu = ops.reshape(mu, [batch_size, -1, 1])
            mu = ops.tile(mu, (1, 1, 2))
            level_decisions = decisions[:, begin_idx:end_idx, :]
            mu = mu * level_decisions
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (level + 1)
        mu = ops.reshape(mu, [batch_size, self.num_leaves])
        probabilities = keras.activations.softmax(self.pi)
        outputs = ops.matmul(mu, probabilities)
        return outputs
'\n## Deep Neural Decision Forest\n\nThe neural decision forest model consists of a set of neural decision trees that are\ntrained simultaneously. The output of the forest model is the average outputs of its trees.\n'

class NeuralDecisionForest(keras.Model):

    def __init__(self, num_trees, depth, num_features, used_features_rate, num_classes):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.ensemble = []
        for _ in range(num_trees):
            self.ensemble.append(NeuralDecisionTree(depth, num_features, used_features_rate, num_classes))

    def call(self, inputs):
        if False:
            while True:
                i = 10
        batch_size = ops.shape(inputs)[0]
        outputs = ops.zeros([batch_size, num_classes])
        for tree in self.ensemble:
            outputs += tree(inputs)
        outputs /= len(self.ensemble)
        return outputs
"\nFinally, let's set up the code that will train and evaluate the model.\n"
learning_rate = 0.01
batch_size = 265
num_epochs = 10

def run_experiment(model):
    if False:
        while True:
            i = 10
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=keras.losses.SparseCategoricalCrossentropy(), metrics=[keras.metrics.SparseCategoricalAccuracy()])
    print('Start training the model...')
    train_dataset = get_dataset_from_csv(train_data_file, shuffle=True, batch_size=batch_size)
    model.fit(train_dataset, epochs=num_epochs)
    print('Model training finished')
    print('Evaluating the model on the test data...')
    test_dataset = get_dataset_from_csv(test_data_file, batch_size=batch_size)
    (_, accuracy) = model.evaluate(test_dataset)
    print(f'Test accuracy: {round(accuracy * 100, 2)}%')
'\n## Experiment 1: train a decision tree model\n\nIn this experiment, we train a single neural decision tree model\nwhere we use all input features.\n'
num_trees = 10
depth = 10
used_features_rate = 1.0
num_classes = len(TARGET_LABELS)

def create_tree_model():
    if False:
        print('Hello World!')
    inputs = create_model_inputs()
    features = encode_inputs(inputs)
    features = layers.BatchNormalization()(features)
    num_features = features.shape[1]
    tree = NeuralDecisionTree(depth, num_features, used_features_rate, num_classes)
    outputs = tree(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
tree_model = create_tree_model()
run_experiment(tree_model)
'\n## Experiment 2: train a forest model\n\nIn this experiment, we train a neural decision forest with `num_trees` trees\nwhere each tree uses randomly selected 50% of the input features. You can control the number\nof features to be used in each tree by setting the `used_features_rate` variable.\nIn addition, we set the depth to 5 instead of 10 compared to the previous experiment.\n'
num_trees = 25
depth = 5
used_features_rate = 0.5

def create_forest_model():
    if False:
        while True:
            i = 10
    inputs = create_model_inputs()
    features = encode_inputs(inputs)
    features = layers.BatchNormalization()(features)
    num_features = features.shape[1]
    forest_model = NeuralDecisionForest(num_trees, depth, num_features, used_features_rate, num_classes)
    outputs = forest_model(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
forest_model = create_forest_model()
run_experiment(forest_model)
'\nYou can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/neural-decision-forest)\nand try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/Neural-Decision-Forest).\n'