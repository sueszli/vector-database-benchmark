from unittest import mock
import autokeras as ak
from autokeras import test_utils

@mock.patch('autokeras.AutoModel.fit')
@mock.patch('autokeras.AutoModel.evaluate')
def test_tsf_evaluate_call_automodel_evaluate(evaluate, fit, tmp_path):
    if False:
        while True:
            i = 10
    auto_model = ak.TimeseriesForecaster(lookback=10, directory=tmp_path, seed=test_utils.SEED)
    auto_model.fit(x=test_utils.TRAIN_CSV_PATH, y='survived')
    auto_model.evaluate(x=test_utils.TRAIN_CSV_PATH, y='survived')
    assert evaluate.is_called

@mock.patch('autokeras.AutoModel.fit')
@mock.patch('autokeras.AutoModel.predict')
def test_tsf_predict_call_automodel_predict(predict, fit, tmp_path):
    if False:
        while True:
            i = 10
    auto_model = ak.TimeseriesForecaster(lookback=10, directory=tmp_path, seed=test_utils.SEED)
    auto_model.fit(x=test_utils.TRAIN_CSV_PATH, y='survived')
    auto_model.predict(x=test_utils.TRAIN_CSV_PATH, y='survived')
    assert predict.is_called

@mock.patch('autokeras.AutoModel.fit')
@mock.patch('autokeras.AutoModel.predict')
def test_tsf_predict_call_automodel_predict_fails(predict, fit, tmp_path):
    if False:
        i = 10
        return i + 15
    auto_model = ak.TimeseriesForecaster(lookback=10, directory=tmp_path, seed=test_utils.SEED)
    auto_model.fit(x=test_utils.TRAIN_CSV_PATH, y='survived')
    try:
        auto_model.predict(x=test_utils.TEST_CSV_PATH, y='survived')
    except ValueError as e:
        assert fit.is_called
        assert 'The prediction data requires the original training data to make'
        ' predictions on subsequent data points' in str(e)

@mock.patch('autokeras.AutoModel.fit')
def test_tsf_fit_call_automodel_fit(fit, tmp_path):
    if False:
        while True:
            i = 10
    auto_model = ak.TimeseriesForecaster(lookback=10, directory=tmp_path, seed=test_utils.SEED)
    auto_model.fit(x=test_utils.TRAIN_CSV_PATH, y='survived', validation_data=(test_utils.TRAIN_CSV_PATH, 'survived'))
    assert fit.is_called