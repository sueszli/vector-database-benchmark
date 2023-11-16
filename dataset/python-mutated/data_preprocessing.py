"""Preprocess dataset and construct any necessary artifacts."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import pickle
import time
import timeit
import typing
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import logging
from official.recommendation import constants as rconst
from official.recommendation import data_pipeline
from official.recommendation import movielens
from official.utils.logs import mlperf_helper
DATASET_TO_NUM_USERS_AND_ITEMS = {'ml-1m': (6040, 3706), 'ml-20m': (138493, 26744)}
_EXPECTED_CACHE_KEYS = (rconst.TRAIN_USER_KEY, rconst.TRAIN_ITEM_KEY, rconst.EVAL_USER_KEY, rconst.EVAL_ITEM_KEY, rconst.USER_MAP, rconst.ITEM_MAP)

def _filter_index_sort(raw_rating_path, cache_path):
    if False:
        while True:
            i = 10
    "Read in data CSV, and output structured data.\n\n  This function reads in the raw CSV of positive items, and performs three\n  preprocessing transformations:\n\n  1)  Filter out all users who have not rated at least a certain number\n      of items. (Typically 20 items)\n\n  2)  Zero index the users and items such that the largest user_id is\n      `num_users - 1` and the largest item_id is `num_items - 1`\n\n  3)  Sort the dataframe by user_id, with timestamp as a secondary sort key.\n      This allows the dataframe to be sliced by user in-place, and for the last\n      item to be selected simply by calling the `-1` index of a user's slice.\n\n  While all of these transformations are performed by Pandas (and are therefore\n  single-threaded), they only take ~2 minutes, and the overhead to apply a\n  MapReduce pattern to parallel process the dataset adds significant complexity\n  for no computational gain. For a larger dataset parallelizing this\n  preprocessing could yield speedups. (Also, this preprocessing step is only\n  performed once for an entire run.\n\n  Args:\n    raw_rating_path: The path to the CSV which contains the raw dataset.\n    cache_path: The path to the file where results of this function are saved.\n\n  Returns:\n    A filtered, zero-index remapped, sorted dataframe, a dict mapping raw user\n    IDs to regularized user IDs, and a dict mapping raw item IDs to regularized\n    item IDs.\n  "
    valid_cache = tf.io.gfile.exists(cache_path)
    if valid_cache:
        with tf.io.gfile.GFile(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        for key in _EXPECTED_CACHE_KEYS:
            if key not in cached_data:
                valid_cache = False
        if not valid_cache:
            logging.info('Removing stale raw data cache file.')
            tf.io.gfile.remove(cache_path)
    if valid_cache:
        data = cached_data
    else:
        with tf.io.gfile.GFile(raw_rating_path) as f:
            df = pd.read_csv(f)
        grouped = df.groupby(movielens.USER_COLUMN)
        df = grouped.filter(lambda x: len(x) >= rconst.MIN_NUM_RATINGS)
        original_users = df[movielens.USER_COLUMN].unique()
        original_items = df[movielens.ITEM_COLUMN].unique()
        logging.info('Generating user_map and item_map...')
        user_map = {user: index for (index, user) in enumerate(original_users)}
        item_map = {item: index for (index, item) in enumerate(original_items)}
        df[movielens.USER_COLUMN] = df[movielens.USER_COLUMN].apply(lambda user: user_map[user])
        df[movielens.ITEM_COLUMN] = df[movielens.ITEM_COLUMN].apply(lambda item: item_map[item])
        num_users = len(original_users)
        num_items = len(original_items)
        mlperf_helper.ncf_print(key=mlperf_helper.TAGS.PREPROC_HP_NUM_EVAL, value=rconst.NUM_EVAL_NEGATIVES)
        assert num_users <= np.iinfo(rconst.USER_DTYPE).max
        assert num_items <= np.iinfo(rconst.ITEM_DTYPE).max
        assert df[movielens.USER_COLUMN].max() == num_users - 1
        assert df[movielens.ITEM_COLUMN].max() == num_items - 1
        logging.info('Sorting by user, timestamp...')
        df.sort_values(by=movielens.TIMESTAMP_COLUMN, inplace=True)
        df.sort_values([movielens.USER_COLUMN, movielens.TIMESTAMP_COLUMN], inplace=True, kind='mergesort')
        df = df.reset_index()
        grouped = df.groupby(movielens.USER_COLUMN, group_keys=False)
        (eval_df, train_df) = (grouped.tail(1), grouped.apply(lambda x: x.iloc[:-1]))
        data = {rconst.TRAIN_USER_KEY: train_df[movielens.USER_COLUMN].values.astype(rconst.USER_DTYPE), rconst.TRAIN_ITEM_KEY: train_df[movielens.ITEM_COLUMN].values.astype(rconst.ITEM_DTYPE), rconst.EVAL_USER_KEY: eval_df[movielens.USER_COLUMN].values.astype(rconst.USER_DTYPE), rconst.EVAL_ITEM_KEY: eval_df[movielens.ITEM_COLUMN].values.astype(rconst.ITEM_DTYPE), rconst.USER_MAP: user_map, rconst.ITEM_MAP: item_map, 'create_time': time.time()}
        logging.info('Writing raw data cache.')
        with tf.io.gfile.GFile(cache_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return (data, valid_cache)

def instantiate_pipeline(dataset, data_dir, params, constructor_type=None, deterministic=False, epoch_dir=None, generate_data_offline=False):
    if False:
        for i in range(10):
            print('nop')
    'Load and digest data CSV into a usable form.\n\n  Args:\n    dataset: The name of the dataset to be used.\n    data_dir: The root directory of the dataset.\n    params: dict of parameters for the run.\n    constructor_type: The name of the constructor subclass that should be used\n      for the input pipeline.\n    deterministic: Tell the data constructor to produce deterministically.\n    epoch_dir: Directory in which to store the training epochs.\n    generate_data_offline: Boolean, whether current pipeline is done offline\n      or while training.\n  '
    logging.info('Beginning data preprocessing.')
    st = timeit.default_timer()
    raw_rating_path = os.path.join(data_dir, dataset, movielens.RATINGS_FILE)
    cache_path = os.path.join(data_dir, dataset, rconst.RAW_CACHE_FILE)
    (raw_data, _) = _filter_index_sort(raw_rating_path, cache_path)
    (user_map, item_map) = (raw_data['user_map'], raw_data['item_map'])
    (num_users, num_items) = DATASET_TO_NUM_USERS_AND_ITEMS[dataset]
    if num_users != len(user_map):
        raise ValueError('Expected to find {} users, but found {}'.format(num_users, len(user_map)))
    if num_items != len(item_map):
        raise ValueError('Expected to find {} items, but found {}'.format(num_items, len(item_map)))
    producer = data_pipeline.get_constructor(constructor_type or 'materialized')(maximum_number_epochs=params['train_epochs'], num_users=num_users, num_items=num_items, user_map=user_map, item_map=item_map, train_pos_users=raw_data[rconst.TRAIN_USER_KEY], train_pos_items=raw_data[rconst.TRAIN_ITEM_KEY], train_batch_size=params['batch_size'], batches_per_train_step=params['batches_per_step'], num_train_negatives=params['num_neg'], eval_pos_users=raw_data[rconst.EVAL_USER_KEY], eval_pos_items=raw_data[rconst.EVAL_ITEM_KEY], eval_batch_size=params['eval_batch_size'], batches_per_eval_step=params['batches_per_step'], stream_files=params['stream_files'], deterministic=deterministic, epoch_dir=epoch_dir, create_data_offline=generate_data_offline)
    run_time = timeit.default_timer() - st
    logging.info('Data preprocessing complete. Time: {:.1f} sec.'.format(run_time))
    print(producer)
    return (num_users, num_items, producer)