"""The wine quality dataset contains data on different wines and their overall quality.

The data has 1599 records with 11 features and one ordinal target column, referring to the overall quality
of a specific wine. see https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009
for additional information.

The typical ML task in this dataset is to build a model that predicts the overall quality of Wine.

This dataset is licensed under the Open Data Commons Open Database License (ODbL) v1.0
(https://opendatacommons.org/licenses/odbl/1-0/).
Right reserved to P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
Modeling wine preferences by data mining from physicochemical properties.
In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
"""
import typing as t
from urllib.request import urlopen
import joblib
import pandas as pd
import sklearn
from category_encoders import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from deepchecks.tabular.dataset import Dataset
__all__ = ['load_data', 'load_fitted_model']
_MODEL_URL = 'https://ndownloader.figshare.com/files/36146916'
_FULL_DATA_URL = 'https://ndownloader.figshare.com/files/36146853'
_TRAIN_DATA_URL = 'https://ndownloader.figshare.com/files/36146856'
_TEST_DATA_URL = 'https://ndownloader.figshare.com/files/36146859'
_MODEL_VERSION = '1.0.2'
_target = 'quality'
_CAT_FEATURES = []
_NUM_FEATURES = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

def load_data(data_format: str='Dataset', as_train_test: bool=True) -> t.Union[t.Tuple, t.Union[Dataset, pd.DataFrame]]:
    if False:
        for i in range(10):
            print('nop')
    "Load and returns the Wine Quality dataset (regression).\n\n    Parameters\n    ----------\n    data_format : str , default: Dataset\n        Represent the format of the returned value. Can be 'Dataset'|'Dataframe'\n        'Dataset' will return the data as a Dataset object\n        'Dataframe' will return the data as a pandas Dataframe object\n    as_train_test : bool , default: True\n        If True, the returned data is splitted into train and test exactly like the toy model\n        was trained. The first return value is the train data and the second is the test data.\n        In order to get this model, call the load_fitted_model() function.\n        Otherwise, returns a single object.\n\n    Returns\n    -------\n    dataset : Union[deepchecks.Dataset, pd.DataFrame]\n        the data object, corresponding to the data_format attribute.\n    train_data, test_data : Tuple[Union[deepchecks.Dataset, pd.DataFrame],Union[deepchecks.Dataset, pd.DataFrame]\n        tuple if as_train_test = True. Tuple of two objects represents the dataset splitted to train and test sets.\n    "
    if not as_train_test:
        dataset = pd.read_csv(_FULL_DATA_URL)
        if data_format == 'Dataset':
            dataset = Dataset(dataset, label=_target, cat_features=_CAT_FEATURES)
        return dataset
    else:
        train = pd.read_csv(_TRAIN_DATA_URL)
        test = pd.read_csv(_TEST_DATA_URL)
        if data_format == 'Dataset':
            train = Dataset(train, label=_target, cat_features=_CAT_FEATURES)
            test = Dataset(test, label=_target, cat_features=_CAT_FEATURES)
        return (train, test)

def load_fitted_model(pretrained=True):
    if False:
        for i in range(10):
            print('nop')
    'Load and return a fitted regression model to predict the quality in the Wine Quality dataset.\n\n    Returns\n    -------\n    model : Joblib\n        the model/pipeline that was trained on the Wine Quality dataset.\n\n    '
    if sklearn.__version__ == _MODEL_VERSION and pretrained:
        with urlopen(_MODEL_URL) as f:
            model = joblib.load(f)
    else:
        model = _build_model()
        (train, _) = load_data()
        model.fit(train.data[train.features], train.data[train.label_name])
    return model

def _build_model():
    if False:
        while True:
            i = 10
    'Build the model to fit.'
    return Pipeline(steps=[('preprocessor', ColumnTransformer(transformers=[('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), _NUM_FEATURES), ('cat', OneHotEncoder(), _CAT_FEATURES)])), ('classifier', RandomForestRegressor(random_state=0, max_depth=7, n_estimators=30))])