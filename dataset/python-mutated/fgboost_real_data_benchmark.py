import numpy as np
import pandas as pd
import argparse
import time
from bigdl.ppml.fl import *
from bigdl.ppml.fl.data_utils import *
from bigdl.ppml.fl.algorithms.fgboost_regression import FGBoostRegression
from urllib.parse import urlparse
from os.path import exists
from bigdl.dllib.utils import log4Error

def is_local_and_existing_uri(uri):
    if False:
        return 10
    parsed_uri = urlparse(uri)
    log4Error.invalidInputError(not parsed_uri.scheme or parsed_uri.scheme == 'file', 'Not Local File!')
    log4Error.invalidInputError(not parsed_uri.netloc or parsed_uri.netloc.lower() == 'localhost', 'Not Local File!')
    log4Error.invalidInputError(exists(parsed_uri.path), 'File Not Exist!')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='PPML FGBoost Benchmark', description='Run PPML FGBoost Benchmark using real data')
    parser.add_argument('--train_path', type=str, help='The train data path.')
    parser.add_argument('--test_path', type=str, help='The test data path.')
    parser.add_argument('--data_size', type=int, default=1, help='The size of data copy, e.g. 10 means using 10 copies of data.')
    parser.add_argument('--num_round', type=int, default=10, help='The boosting rounds.')
    args = parser.parse_args()
    init_fl_context(1)
    is_local_and_existing_uri(args.train_path)
    df_train = pd.read_csv(args.train_path)
    fgboost_regression = FGBoostRegression()
    df_x = df_train.drop('SalePrice', 1)
    df_y = df_train.filter(items=['SalePrice'])
    x = convert_to_numpy(df_x)
    y = convert_to_numpy(df_y)
    x_stacked = []
    y_stacked = []
    for i in range(args.data_size):
        x_stacked.append(x)
        y_stacked.append(y)
    x_stacked = np.array(x_stacked)
    y_stacked = np.array(y_stacked)
    fgboost_regression.fit(x_stacked.reshape(-1, x_stacked.shape[-1]), y_stacked.reshape(-1, y_stacked.shape[-1]), num_round=args.num_round)
    is_local_and_existing_uri(args.test_path)
    df_test = pd.read_csv(args.test_path)
    result = fgboost_regression.predict(df_test, feature_columns=df_x.columns)