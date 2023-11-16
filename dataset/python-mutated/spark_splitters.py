import numpy as np
try:
    from pyspark.sql import functions as F, Window
    from pyspark.storagelevel import StorageLevel
except ImportError:
    pass
from recommenders.utils.constants import DEFAULT_ITEM_COL, DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL
from recommenders.datasets.split_utils import process_split_ratio, min_rating_filter_spark

def spark_random_split(data, ratio=0.75, seed=42):
    if False:
        for i in range(10):
            print('nop')
    'Spark random splitter.\n\n    Randomly split the data into several splits.\n\n    Args:\n        data (pyspark.sql.DataFrame): Spark DataFrame to be split.\n        ratio (float or list): Ratio for splitting data. If it is a single float number\n            it splits data into two halves and the ratio argument indicates the ratio of\n            training data set; if it is a list of float numbers, the splitter splits\n            data into several portions corresponding to the split ratios. If a list\n            is provided and the ratios are not summed to 1, they will be normalized.\n        seed (int): Seed.\n\n    Returns:\n        list: Splits of the input data as pyspark.sql.DataFrame.\n    '
    (multi_split, ratio) = process_split_ratio(ratio)
    if multi_split:
        return data.randomSplit(ratio, seed=seed)
    else:
        return data.randomSplit([ratio, 1 - ratio], seed=seed)

def _do_stratification_spark(data, ratio=0.75, min_rating=1, filter_by='user', is_partitioned=True, is_random=True, seed=42, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_timestamp=DEFAULT_TIMESTAMP_COL):
    if False:
        for i in range(10):
            print('nop')
    'Helper function to perform stratified splits.\n\n    This function splits data in a stratified manner. That is, the same values for the\n    filter_by column are retained in each split, but the corresponding set of entries\n    are divided according to the ratio provided.\n\n    Args:\n        data (pyspark.sql.DataFrame): Spark DataFrame to be split.\n        ratio (float or list): Ratio for splitting data. If it is a single float number\n            it splits data into two sets and the ratio argument indicates the ratio of\n            training data set; if it is a list of float numbers, the splitter splits\n            data into several portions corresponding to the split ratios. If a list is\n            provided and the ratios are not summed to 1, they will be normalized.\n        min_rating (int): minimum number of ratings for user or item.\n        filter_by (str): either "user" or "item", depending on which of the two is to filter\n            with min_rating.\n        is_partitioned (bool): flag to partition data by filter_by column\n        is_random (bool): flag to make split randomly or use timestamp column\n        seed (int): Seed.\n        col_user (str): column name of user IDs.\n        col_item (str): column name of item IDs.\n        col_timestamp (str): column name of timestamps.\n\n    Args:\n\n    Returns:\n    '
    if filter_by not in ['user', 'item']:
        raise ValueError("filter_by should be either 'user' or 'item'.")
    if min_rating < 1:
        raise ValueError('min_rating should be integer and larger than or equal to 1.')
    if col_user not in data.columns:
        raise ValueError('Schema of data not valid. Missing User Col')
    if col_item not in data.columns:
        raise ValueError('Schema of data not valid. Missing Item Col')
    if not is_random:
        if col_timestamp not in data.columns:
            raise ValueError('Schema of data not valid. Missing Timestamp Col')
    if min_rating > 1:
        data = min_rating_filter_spark(data=data, min_rating=min_rating, filter_by=filter_by, col_user=col_user, col_item=col_item)
    split_by = col_user if filter_by == 'user' else col_item
    partition_by = split_by if is_partitioned else []
    col_random = '_random'
    if is_random:
        data = data.withColumn(col_random, F.rand(seed=seed))
        order_by = F.col(col_random)
    else:
        order_by = F.col(col_timestamp)
    window_count = Window.partitionBy(partition_by)
    window_spec = Window.partitionBy(partition_by).orderBy(order_by)
    data = data.withColumn('_count', F.count(split_by).over(window_count)).withColumn('_rank', F.row_number().over(window_spec) / F.col('_count')).drop('_count', col_random)
    data.persist(StorageLevel.MEMORY_AND_DISK_2).count()
    (multi_split, ratio) = process_split_ratio(ratio)
    ratio = ratio if multi_split else [ratio, 1 - ratio]
    splits = []
    prev_split = None
    for split in np.cumsum(ratio):
        condition = F.col('_rank') <= split
        if prev_split is not None:
            condition &= F.col('_rank') > prev_split
        splits.append(data.filter(condition).drop('_rank'))
        prev_split = split
    return splits

def spark_chrono_split(data, ratio=0.75, min_rating=1, filter_by='user', col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_timestamp=DEFAULT_TIMESTAMP_COL, no_partition=False):
    if False:
        while True:
            i = 10
    'Spark chronological splitter.\n\n    This function splits data in a chronological manner. That is, for each user / item, the\n    split function takes proportions of ratings which is specified by the split ratio(s).\n    The split is stratified.\n\n    Args:\n        data (pyspark.sql.DataFrame): Spark DataFrame to be split.\n        ratio (float or list): Ratio for splitting data. If it is a single float number\n            it splits data into two sets and the ratio argument indicates the ratio of\n            training data set; if it is a list of float numbers, the splitter splits\n            data into several portions corresponding to the split ratios. If a list is\n            provided and the ratios are not summed to 1, they will be normalized.\n        min_rating (int): minimum number of ratings for user or item.\n        filter_by (str): either "user" or "item", depending on which of the two is to filter\n            with min_rating.\n        col_user (str): column name of user IDs.\n        col_item (str): column name of item IDs.\n        col_timestamp (str): column name of timestamps.\n        no_partition (bool): set to enable more accurate and less efficient splitting.\n\n    Returns:\n        list: Splits of the input data as pyspark.sql.DataFrame.\n    '
    return _do_stratification_spark(data=data, ratio=ratio, min_rating=min_rating, filter_by=filter_by, is_random=False, col_user=col_user, col_item=col_item, col_timestamp=col_timestamp)

def spark_stratified_split(data, ratio=0.75, min_rating=1, filter_by='user', col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, seed=42):
    if False:
        for i in range(10):
            print('nop')
    'Spark stratified splitter.\n\n    For each user / item, the split function takes proportions of ratings which is\n    specified by the split ratio(s). The split is stratified.\n\n    Args:\n        data (pyspark.sql.DataFrame): Spark DataFrame to be split.\n        ratio (float or list): Ratio for splitting data. If it is a single float number\n            it splits data into two halves and the ratio argument indicates the ratio of\n            training data set; if it is a list of float numbers, the splitter splits\n            data into several portions corresponding to the split ratios. If a list is\n            provided and the ratios are not summed to 1, they will be normalized.\n            Earlier indexed splits will have earlier times\n            (e.g. the latest time per user or item in split[0] <= the earliest time per user or item in split[1])\n        seed (int): Seed.\n        min_rating (int): minimum number of ratings for user or item.\n        filter_by (str): either "user" or "item", depending on which of the two is to filter\n            with min_rating.\n        col_user (str): column name of user IDs.\n        col_item (str): column name of item IDs.\n\n    Returns:\n        list: Splits of the input data as pyspark.sql.DataFrame.\n    '
    return _do_stratification_spark(data=data, ratio=ratio, min_rating=min_rating, filter_by=filter_by, seed=seed, col_user=col_user, col_item=col_item)

def spark_timestamp_split(data, ratio=0.75, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_timestamp=DEFAULT_TIMESTAMP_COL):
    if False:
        for i in range(10):
            print('nop')
    'Spark timestamp based splitter.\n\n    The splitter splits the data into sets by timestamps without stratification on either user or item.\n    The ratios are applied on the timestamp column which is divided accordingly into several partitions.\n\n    Args:\n        data (pyspark.sql.DataFrame): Spark DataFrame to be split.\n        ratio (float or list): Ratio for splitting data. If it is a single float number\n            it splits data into two sets and the ratio argument indicates the ratio of\n            training data set; if it is a list of float numbers, the splitter splits\n            data into several portions corresponding to the split ratios. If a list is\n            provided and the ratios are not summed to 1, they will be normalized.\n            Earlier indexed splits will have earlier times\n            (e.g. the latest time in split[0] <= the earliest time in split[1])\n        col_user (str): column name of user IDs.\n        col_item (str): column name of item IDs.\n        col_timestamp (str): column name of timestamps. Float number represented in\n        seconds since Epoch.\n\n    Returns:\n        list: Splits of the input data as pyspark.sql.DataFrame.\n    '
    return _do_stratification_spark(data=data, ratio=ratio, is_random=False, is_partitioned=False, col_user=col_user, col_item=col_item, col_timestamp=col_timestamp)