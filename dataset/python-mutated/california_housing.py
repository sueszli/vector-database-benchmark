"""Boston housing price regression dataset."""
import numpy as np
from keras.api_export import keras_export
from keras.utils.file_utils import get_file

@keras_export('keras.datasets.boston_housing.load_data')
def load_data(path='california_housing.npz', test_split=0.2, seed=113):
    if False:
        while True:
            i = 10
    "Loads the California Housing dataset.\n\n    This dataset was obtained from the [StatLib repository](\n    https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html).\n\n    It's a continuous regression dataset with 20,640 samples with\n    8 features each.\n\n    The target variable is a scalar: the median house value\n    for California districts, in dollars.\n\n    The 8 input features are the following:\n\n    - MedInc: median income in block group\n    - HouseAge: median house age in block group\n    - AveRooms: average number of rooms per household\n    - AveBedrms: average number of bedrooms per household\n    - Population: block group population\n    - AveOccup: average number of household members\n    - Latitude: block group latitude\n    - Longitude: block group longitude\n\n    This dataset was derived from the 1990 U.S. census, using one row\n    per census block group. A block group is the smallest geographical\n    unit for which the U.S. Census Bureau publishes sample data\n    (a block group typically has a population of 600 to 3,000 people).\n\n    A household is a group of people residing within a home.\n    Since the average number of rooms and bedrooms in this dataset are\n    provided per household, these columns may take surprisingly large\n    values for block groups with few households and many empty houses,\n    such as vacation resorts.\n\n    Args:\n        path: path where to cache the dataset locally\n            (relative to `~/.keras/datasets`).\n        test_split: fraction of the data to reserve as test set.\n        seed: Random seed for shuffling the data\n            before computing the test split.\n\n    Returns:\n        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.\n\n    **`x_train`, `x_test`**: numpy arrays with shape `(num_samples, 8)`\n      containing either the training samples (for `x_train`),\n      or test samples (for `y_train`).\n\n    **`y_train`, `y_test`**: numpy arrays of shape `(num_samples,)`\n        containing the target scalars. The targets are float scalars\n        typically between 25,000 and 500,000 that represent\n        the home prices in dollars.\n    "
    assert 0 <= test_split < 1
    origin_folder = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    path = get_file(path, origin=origin_folder + 'california_housing.npz', file_hash='1a2e3a52e0398de6463aebe6f4a8da34fb21fbb6b934cf88c3425e766f2a1a6f')
    with np.load(path, allow_pickle=True) as f:
        x = f['x']
        y = f['y']
    rng = np.random.RandomState(seed)
    indices = np.arange(len(x))
    rng.shuffle(indices)
    x = x[indices]
    y = y[indices]
    x_train = np.array(x[:int(len(x) * (1 - test_split))])
    y_train = np.array(y[:int(len(x) * (1 - test_split))])
    x_test = np.array(x[int(len(x) * (1 - test_split)):])
    y_test = np.array(y[int(len(x) * (1 - test_split)):])
    return ((x_train, y_train), (x_test, y_test))