"""Utilities for producing synthetic test data that is convergence-friendly."""
from collections import namedtuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
RANDOM_SEED = 42
NUMBER_OBSERVATIONS = 200
GeneratedData = namedtuple('GeneratedData', 'train_df validation_df test_df')

def get_feature_configs():
    if False:
        print('Hello World!')
    input_features = [{'name': 'x', 'type': 'number'}]
    output_features = [{'name': 'y', 'type': 'number', 'loss': {'type': 'mean_squared_error'}, 'decoder': {'num_fc_layers': 2, 'fc_output_size': 64}}]
    return (input_features, output_features)

def get_generated_data():
    if False:
        while True:
            i = 10
    np.random.seed(RANDOM_SEED)
    x = np.array(range(NUMBER_OBSERVATIONS)).reshape(-1, 1)
    y = 2 * x + 1 + np.random.normal(size=x.shape[0]).reshape(-1, 1)
    raw_df = pd.DataFrame(np.concatenate((x, y), axis=1), columns=['x', 'y'])
    (train, valid_test) = train_test_split(raw_df, train_size=0.7)
    (validation, test) = train_test_split(valid_test, train_size=0.5)
    return GeneratedData(train, validation, test)

def get_generated_data_for_optimizer():
    if False:
        print('Hello World!')
    np.random.seed(RANDOM_SEED)
    x = np.array(range(NUMBER_OBSERVATIONS)).reshape(-1, 1)
    y = 2 * x + 1 + np.random.normal(size=x.shape[0]).reshape(-1, 1)
    raw_df = pd.DataFrame(np.concatenate((x, y), axis=1), columns=['x', 'y'])
    raw_df['x'] = (raw_df['x'] - raw_df['x'].min()) / (raw_df['x'].max() - raw_df['x'].min())
    raw_df['y'] = (raw_df['y'] - raw_df['y'].min()) / (raw_df['y'].max() - raw_df['y'].min())
    (train, valid_test) = train_test_split(raw_df, train_size=0.7)
    (validation, test) = train_test_split(valid_test, train_size=0.5)
    return GeneratedData(train, validation, test)