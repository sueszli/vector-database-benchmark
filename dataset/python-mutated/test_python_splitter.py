import pytest
import pandas as pd
import numpy as np
from recommenders.utils.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL
from recommenders.datasets.split_utils import min_rating_filter_pandas, split_pandas_data_with_ratios
from recommenders.datasets.python_splitters import python_chrono_split, python_random_split, python_stratified_split, numpy_stratified_split

@pytest.fixture(scope='module')
def test_specs():
    if False:
        i = 10
        return i + 15
    return {'number_of_rows': 1000, 'seed': 123, 'ratio': 0.6, 'ratios': [0.2, 0.3, 0.5], 'split_numbers': [2, 3, 5], 'tolerance': 0.01, 'number_of_items': 50, 'number_of_users': 20, 'fluctuation': 0.02}

@pytest.fixture(scope='module')
def python_dataset(test_specs):
    if False:
        while True:
            i = 10

    def random_date_generator(start_date, range_in_days):
        if False:
            print('Hello World!')
        'Helper function to generate random timestamps.\n\n        Reference: https://stackoverflow.com/questions/41006182/generate-random-dates-within-a-range-in-numpy\n        '
        days_to_add = np.arange(0, range_in_days)
        random_dates = []
        for i in range(range_in_days):
            random_date = np.datetime64(start_date) + np.random.choice(days_to_add)
            random_dates.append(random_date)
        return random_dates
    np.random.seed(test_specs['seed'])
    rating = pd.DataFrame({DEFAULT_USER_COL: np.random.randint(1, 5, test_specs['number_of_rows']), DEFAULT_ITEM_COL: np.random.randint(1, 15, test_specs['number_of_rows']), DEFAULT_RATING_COL: np.random.randint(1, 6, test_specs['number_of_rows']), DEFAULT_TIMESTAMP_COL: random_date_generator('2018-01-01', test_specs['number_of_rows'])})
    return rating

@pytest.fixture(scope='module')
def python_int_dataset(test_specs):
    if False:
        while True:
            i = 10
    np.random.seed(test_specs['seed'])
    return np.random.randint(low=0, high=6, size=(test_specs['number_of_users'], test_specs['number_of_items']))

@pytest.fixture(scope='module')
def python_float_dataset(test_specs):
    if False:
        print('Hello World!')
    np.random.seed(test_specs['seed'])
    return np.random.random(size=(test_specs['number_of_users'], test_specs['number_of_items'])) * 5

def test_split_pandas_data(pandas_dummy_timestamp):
    if False:
        for i in range(10):
            print('nop')
    splits = split_pandas_data_with_ratios(pandas_dummy_timestamp, ratios=[0.5, 0.5])
    assert len(splits[0]) == 5
    assert len(splits[1]) == 5
    splits = split_pandas_data_with_ratios(pandas_dummy_timestamp, ratios=[0.12, 0.36, 0.52])
    shape = pandas_dummy_timestamp.shape[0]
    assert len(splits[0]) == round(shape * 0.12)
    assert len(splits[1]) == round(shape * 0.36)
    assert len(splits[2]) == round(shape * 0.52)
    with pytest.raises(ValueError):
        splits = split_pandas_data_with_ratios(pandas_dummy_timestamp, ratios=[0.6, 0.2, 0.4])

def test_min_rating_filter():
    if False:
        while True:
            i = 10
    python_dataset = pd.DataFrame({DEFAULT_USER_COL: [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5], DEFAULT_ITEM_COL: [5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2, 1], DEFAULT_RATING_COL: np.random.randint(1, 6, 15)})

    def count_filtered_rows(data, filter_by='user'):
        if False:
            for i in range(10):
                print('nop')
        split_by_column = DEFAULT_USER_COL if filter_by == 'user' else DEFAULT_ITEM_COL
        data_grouped = data.groupby(split_by_column)
        row_counts = []
        for (name, group) in data_grouped:
            data_group = data_grouped.get_group(name)
            row_counts.append(data_group.shape[0])
        return row_counts
    df_user = min_rating_filter_pandas(python_dataset, min_rating=3, filter_by='user')
    df_item = min_rating_filter_pandas(python_dataset, min_rating=2, filter_by='item')
    user_rating_counts = count_filtered_rows(df_user, filter_by='user')
    item_rating_counts = count_filtered_rows(df_item, filter_by='item')
    assert all((u >= 3 for u in user_rating_counts))
    assert all((i >= 2 for i in item_rating_counts))

def test_random_splitter(test_specs, python_dataset):
    if False:
        while True:
            i = 10
    'NOTE: some split results may not match exactly with the ratios, which may be owing to the  limited number of\n    rows in the testing data. A approximate match with certain level of tolerance is therefore used instead for tests.\n    '
    splits = python_random_split(python_dataset, ratio=test_specs['ratio'], seed=test_specs['seed'])
    assert len(splits[0]) / test_specs['number_of_rows'] == pytest.approx(test_specs['ratio'], test_specs['tolerance'])
    assert len(splits[1]) / test_specs['number_of_rows'] == pytest.approx(1 - test_specs['ratio'], test_specs['tolerance'])
    for split in splits:
        assert set(split.columns) == set(python_dataset.columns)
    splits = python_random_split(python_dataset, ratio=test_specs['ratios'], seed=test_specs['seed'])
    assert len(splits) == 3
    assert len(splits[0]) / test_specs['number_of_rows'] == pytest.approx(test_specs['ratios'][0], test_specs['tolerance'])
    assert len(splits[1]) / test_specs['number_of_rows'] == pytest.approx(test_specs['ratios'][1], test_specs['tolerance'])
    assert len(splits[2]) / test_specs['number_of_rows'] == pytest.approx(test_specs['ratios'][2], test_specs['tolerance'])
    for split in splits:
        assert set(split.columns) == set(python_dataset.columns)
    splits = python_random_split(python_dataset, ratio=test_specs['split_numbers'], seed=test_specs['seed'])
    assert len(splits) == 3
    assert len(splits[0]) / test_specs['number_of_rows'] == pytest.approx(test_specs['ratios'][0], test_specs['tolerance'])
    assert len(splits[1]) / test_specs['number_of_rows'] == pytest.approx(test_specs['ratios'][1], test_specs['tolerance'])
    assert len(splits[2]) / test_specs['number_of_rows'] == pytest.approx(test_specs['ratios'][2], test_specs['tolerance'])
    for split in splits:
        assert set(split.columns) == set(python_dataset.columns)
    splits = python_random_split(python_dataset, ratio=[0.7, 0.2, 0.1], seed=test_specs['seed'])
    assert len(splits) == 3

def test_chrono_splitter(test_specs, python_dataset):
    if False:
        return 10
    splits = python_chrono_split(python_dataset, ratio=test_specs['ratio'], min_rating=10, filter_by='user')
    assert len(splits[0]) / test_specs['number_of_rows'] == pytest.approx(test_specs['ratio'], test_specs['tolerance'])
    assert len(splits[1]) / test_specs['number_of_rows'] == pytest.approx(1 - test_specs['ratio'], test_specs['tolerance'])
    for split in splits:
        assert set(split.columns) == set(python_dataset.columns)
    users_train = splits[0][DEFAULT_USER_COL].unique()
    users_test = splits[1][DEFAULT_USER_COL].unique()
    assert set(users_train) == set(users_test)
    max_train_times = splits[0][[DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL]].groupby(DEFAULT_USER_COL).max()
    min_test_times = splits[1][[DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL]].groupby(DEFAULT_USER_COL).min()
    check_times = max_train_times.join(min_test_times, lsuffix='_0', rsuffix='_1')
    assert all((check_times[DEFAULT_TIMESTAMP_COL + '_0'] < check_times[DEFAULT_TIMESTAMP_COL + '_1']).values)
    splits = python_chrono_split(python_dataset, ratio=test_specs['ratios'], min_rating=10, filter_by='user')
    assert len(splits) == 3
    assert len(splits[0]) / test_specs['number_of_rows'] == pytest.approx(test_specs['ratios'][0], test_specs['tolerance'])
    assert len(splits[1]) / test_specs['number_of_rows'] == pytest.approx(test_specs['ratios'][1], test_specs['tolerance'])
    assert len(splits[2]) / test_specs['number_of_rows'] == pytest.approx(test_specs['ratios'][2], test_specs['tolerance'])
    for split in splits:
        assert set(split.columns) == set(python_dataset.columns)
    users_train = splits[0][DEFAULT_USER_COL].unique()
    users_test = splits[1][DEFAULT_USER_COL].unique()
    users_val = splits[2][DEFAULT_USER_COL].unique()
    assert set(users_train) == set(users_test)
    assert set(users_train) == set(users_val)
    max_train_times = splits[0][[DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL]].groupby(DEFAULT_USER_COL).max()
    min_test_times = splits[1][[DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL]].groupby(DEFAULT_USER_COL).min()
    check_times = max_train_times.join(min_test_times, lsuffix='_0', rsuffix='_1')
    assert all((check_times[DEFAULT_TIMESTAMP_COL + '_0'] < check_times[DEFAULT_TIMESTAMP_COL + '_1']).values)
    max_test_times = splits[1][[DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL]].groupby(DEFAULT_USER_COL).max()
    min_val_times = splits[2][[DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL]].groupby(DEFAULT_USER_COL).min()
    check_times = max_test_times.join(min_val_times, lsuffix='_1', rsuffix='_2')
    assert all((check_times[DEFAULT_TIMESTAMP_COL + '_1'] < check_times[DEFAULT_TIMESTAMP_COL + '_2']).values)

def test_stratified_splitter(test_specs, python_dataset):
    if False:
        for i in range(10):
            print('nop')
    splits = python_stratified_split(python_dataset, ratio=test_specs['ratio'], min_rating=10, filter_by='user')
    assert len(splits[0]) / test_specs['number_of_rows'] == pytest.approx(test_specs['ratio'], test_specs['tolerance'])
    assert len(splits[1]) / test_specs['number_of_rows'] == pytest.approx(1 - test_specs['ratio'], test_specs['tolerance'])
    for split in splits:
        assert set(split.columns) == set(python_dataset.columns)
    users_train = splits[0][DEFAULT_USER_COL].unique()
    users_test = splits[1][DEFAULT_USER_COL].unique()
    assert set(users_train) == set(users_test)
    splits = python_stratified_split(python_dataset, ratio=test_specs['ratios'], min_rating=10, filter_by='user')
    assert len(splits) == 3
    assert len(splits[0]) / test_specs['number_of_rows'] == pytest.approx(test_specs['ratios'][0], test_specs['tolerance'])
    assert len(splits[1]) / test_specs['number_of_rows'] == pytest.approx(test_specs['ratios'][1], test_specs['tolerance'])
    assert len(splits[2]) / test_specs['number_of_rows'] == pytest.approx(test_specs['ratios'][2], test_specs['tolerance'])
    for split in splits:
        assert set(split.columns) == set(python_dataset.columns)

def test_int_numpy_stratified_splitter(test_specs, python_int_dataset):
    if False:
        return 10
    X = python_int_dataset
    (Xtr, Xtst) = numpy_stratified_split(X, ratio=test_specs['ratio'], seed=test_specs['seed'])
    assert (Xtr.shape[0] == X.shape[0]) & (Xtr.shape[1] == X.shape[1])
    assert (Xtst.shape[0] == X.shape[0]) & (Xtst.shape[1] == X.shape[1])
    X_rated = np.sum(X != 0, axis=1)
    Xtr_rated = np.sum(Xtr != 0, axis=1)
    Xtst_rated = np.sum(Xtst != 0, axis=1)
    assert Xtr_rated.sum() / X_rated.sum() == pytest.approx(test_specs['ratio'], test_specs['tolerance'])
    assert Xtst_rated.sum() / X_rated.sum() == pytest.approx(1 - test_specs['ratio'], test_specs['tolerance'])
    assert (Xtr_rated / X_rated <= test_specs['ratio'] + test_specs['fluctuation']).all() & (Xtr_rated / X_rated >= test_specs['ratio'] - test_specs['fluctuation']).all()
    assert (Xtst_rated / X_rated <= 1 - test_specs['ratio'] + test_specs['fluctuation']).all() & (Xtst_rated / X_rated >= 1 - test_specs['ratio'] - test_specs['fluctuation']).all()

def test_float_numpy_stratified_splitter(test_specs, python_float_dataset):
    if False:
        return 10
    X = python_float_dataset
    (Xtr, Xtst) = numpy_stratified_split(X, ratio=test_specs['ratio'], seed=test_specs['seed'])
    assert (Xtr.shape[0] == X.shape[0]) & (Xtr.shape[1] == X.shape[1])
    assert (Xtst.shape[0] == X.shape[0]) & (Xtst.shape[1] == X.shape[1])
    X_rated = np.sum(X != 0, axis=1)
    Xtr_rated = np.sum(Xtr != 0, axis=1)
    Xtst_rated = np.sum(Xtst != 0, axis=1)
    assert Xtr_rated.sum() / X_rated.sum() == pytest.approx(test_specs['ratio'], test_specs['tolerance'])
    assert Xtst_rated.sum() / X_rated.sum() == pytest.approx(1 - test_specs['ratio'], test_specs['tolerance'])
    assert Xtr_rated / X_rated == pytest.approx(test_specs['ratio'], rel=test_specs['fluctuation'])
    assert Xtst_rated / X_rated == pytest.approx(1 - test_specs['ratio'], rel=test_specs['fluctuation'])