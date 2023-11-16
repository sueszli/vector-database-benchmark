from unittest import mock
import numpy as np
import pandas as pd
import pytest
from tensorflow import nest
import autokeras as ak
from autokeras import test_utils

def test_raise_error_unknown_str_in_col_type(tmp_path):
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError) as info:
        ak.StructuredDataClassifier(column_types={'age': 'num', 'parch': 'categorical'}, directory=tmp_path, seed=test_utils.SEED)
    assert 'column_types should be either "categorical"' in str(info.value)

def test_structured_data_input_name_type_mismatch_error(tmp_path):
    if False:
        print('Hello World!')
    with pytest.raises(ValueError) as info:
        clf = ak.StructuredDataClassifier(column_types={'_age': 'numerical', 'parch': 'categorical'}, column_names=['age', 'fare'], directory=tmp_path, seed=test_utils.SEED)
        clf.fit(x=test_utils.TRAIN_CSV_PATH, y='survived')
    assert 'column_names and column_types are mismatched.' in str(info.value)

def test_structured_data_col_type_no_name_error(tmp_path):
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError) as info:
        clf = ak.StructuredDataClassifier(column_types={'age': 'numerical', 'parch': 'categorical'}, directory=tmp_path, seed=test_utils.SEED)
        clf.fit(x=np.random.rand(100, 30), y=np.random.rand(100, 1))
    assert 'column_names must be specified' in str(info.value)

@mock.patch('autokeras.AutoModel.fit')
def test_structured_data_get_col_names_from_df(fit, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    clf = ak.StructuredDataClassifier(directory=tmp_path, seed=test_utils.SEED)
    clf.fit(x=test_utils.TRAIN_CSV_PATH, y='survived')
    assert nest.flatten(clf.inputs)[0].column_names[0] == 'sex'

@mock.patch('autokeras.AutoModel.fit')
@mock.patch('autokeras.AutoModel.evaluate')
def test_structured_clf_evaluate_call_automodel_evaluate(evaluate, fit, tmp_path):
    if False:
        while True:
            i = 10
    auto_model = ak.StructuredDataClassifier(directory=tmp_path, seed=test_utils.SEED)
    auto_model.fit(x=test_utils.TRAIN_CSV_PATH, y='survived')
    auto_model.evaluate(x=test_utils.TRAIN_CSV_PATH, y='survived')
    assert evaluate.is_called

@mock.patch('autokeras.AutoModel.fit')
@mock.patch('autokeras.AutoModel.predict')
def test_structured_clf_predict_csv_call_automodel_predict(predict, fit, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    auto_model = ak.StructuredDataClassifier(directory=tmp_path, seed=test_utils.SEED)
    auto_model.fit(x=test_utils.TRAIN_CSV_PATH, y='survived')
    auto_model.predict(x=test_utils.TEST_CSV_PATH)
    assert predict.is_called

@mock.patch('autokeras.AutoModel.fit')
def test_structured_clf_fit_call_auto_model_fit(fit, tmp_path):
    if False:
        print('Hello World!')
    auto_model = ak.StructuredDataClassifier(directory=tmp_path, seed=test_utils.SEED)
    auto_model.fit(x=pd.read_csv(test_utils.TRAIN_CSV_PATH).to_numpy().astype(str)[:100], y=test_utils.generate_one_hot_labels(num_instances=100, num_classes=3))
    assert fit.is_called

@mock.patch('autokeras.AutoModel.fit')
def test_structured_reg_fit_call_auto_model_fit(fit, tmp_path):
    if False:
        i = 10
        return i + 15
    auto_model = ak.StructuredDataRegressor(directory=tmp_path, seed=test_utils.SEED)
    auto_model.fit(x=pd.read_csv(test_utils.TRAIN_CSV_PATH).to_numpy().astype(str)[:100], y=test_utils.generate_data(num_instances=100, shape=(1,)))
    assert fit.is_called

@mock.patch('autokeras.AutoModel.fit')
def test_structured_data_clf_convert_csv_to_df_and_np(fit, tmp_path):
    if False:
        print('Hello World!')
    auto_model = ak.StructuredDataClassifier(directory=tmp_path, seed=test_utils.SEED)
    auto_model.fit(x=test_utils.TRAIN_CSV_PATH, y='survived', epochs=2, validation_data=(test_utils.TEST_CSV_PATH, 'survived'))
    (_, kwargs) = fit.call_args_list[0]
    assert isinstance(kwargs['x'], pd.DataFrame)
    assert isinstance(kwargs['y'], np.ndarray)