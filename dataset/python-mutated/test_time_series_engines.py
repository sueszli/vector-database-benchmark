"""Module to test setting of engines in time series
"""
from daal4py.sklearn.linear_model._linear import LinearRegression as SklearnexLinearRegression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sktime.forecasting.arima import AutoARIMA as PmdAutoARIMA
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
from pycaret.time_series import TSForecastingExperiment

def test_engines_setup_global_args(load_pos_and_neg_data):
    if False:
        for i in range(10):
            print('nop')
    'Tests the setting of engines using global arguments in setup.\n    We test for both statistical models and regression models.\n    '
    exp = TSForecastingExperiment()
    data = load_pos_and_neg_data
    exp.setup(data=data, fold=2, fh=12, fold_strategy='sliding', verbose=False, engine={'auto_arima': 'statsforecast', 'lr_cds_dt': 'sklearnex'})
    assert exp.get_engine('auto_arima') == 'statsforecast'
    model = exp.create_model('auto_arima', cross_validation=False)
    assert isinstance(model, StatsForecastAutoARIMA)
    assert exp.get_engine('auto_arima') == 'statsforecast'
    assert exp.get_engine('lr_cds_dt') == 'sklearnex'
    model = exp.create_model('lr_cds_dt', cross_validation=False)
    assert isinstance(model.regressor, SklearnexLinearRegression)
    assert exp.get_engine('lr_cds_dt') == 'sklearnex'

def test_engines_global_methods(load_pos_and_neg_data):
    if False:
        return 10
    'Tests the setting of engines using methods like set_engine (global changes).\n    We test for both statistical models and regression models.\n    '
    exp = TSForecastingExperiment()
    data = load_pos_and_neg_data
    exp.setup(data=data, fold=2, fh=12, fold_strategy='sliding', verbose=False, engine={'auto_arima': 'statsforecast', 'lr_cds_dt': 'sklearnex'})
    assert exp.get_engine('auto_arima') == 'statsforecast'
    exp._set_engine('auto_arima', 'pmdarima')
    assert exp.get_engine('auto_arima') == 'pmdarima'
    model = exp.create_model('auto_arima', cross_validation=False)
    assert isinstance(model, PmdAutoARIMA)
    assert exp.get_engine('lr_cds_dt') == 'sklearnex'
    exp._set_engine('lr_cds_dt', 'sklearn')
    assert exp.get_engine('lr_cds_dt') == 'sklearn'
    model = exp.create_model('lr_cds_dt', cross_validation=False)
    assert isinstance(model.regressor, SklearnLinearRegression)

def test_create_model_engines_local_args(load_pos_and_neg_data):
    if False:
        i = 10
        return i + 15
    'Tests the setting of engines for create_model using local args.\n    We test for both statistical models and regression models.\n    '
    exp = TSForecastingExperiment()
    data = load_pos_and_neg_data
    exp.setup(data=data, fold=2, fh=12, fold_strategy='sliding', verbose=False)
    assert exp.get_engine('auto_arima') == 'pmdarima'
    model = exp.create_model('auto_arima', cross_validation=False)
    assert isinstance(model, PmdAutoARIMA)
    assert exp.get_engine('auto_arima') == 'pmdarima'
    assert exp.get_engine('lr_cds_dt') == 'sklearn'
    model = exp.create_model('lr_cds_dt', cross_validation=False)
    assert isinstance(model.regressor, SklearnLinearRegression)
    assert exp.get_engine('lr_cds_dt') == 'sklearn'
    model = exp.create_model('auto_arima', engine='statsforecast', cross_validation=False)
    assert isinstance(model, StatsForecastAutoARIMA)
    assert exp.get_engine('auto_arima') == 'pmdarima'
    model = exp.create_model('auto_arima')
    assert isinstance(model, PmdAutoARIMA)
    model = exp.create_model('lr_cds_dt', engine='sklearnex', cross_validation=False)
    assert isinstance(model.regressor, SklearnexLinearRegression)
    assert exp.get_engine('lr_cds_dt') == 'sklearn'
    model = exp.create_model('lr_cds_dt')
    assert isinstance(model.regressor, SklearnLinearRegression)

def test_compare_models_engines_local_args(load_pos_and_neg_data):
    if False:
        print('Hello World!')
    'Tests the setting of engines for compare_models using local args.\n    We test for both statistical models and regression models.\n    '
    exp = TSForecastingExperiment()
    data = load_pos_and_neg_data
    exp.setup(data=data, fold=2, fh=12, fold_strategy='sliding', verbose=False)
    assert exp.get_engine('auto_arima') == 'pmdarima'
    model = exp.compare_models(include=['auto_arima'])
    assert isinstance(model, PmdAutoARIMA)
    assert exp.get_engine('auto_arima') == 'pmdarima'
    assert exp.get_engine('lr_cds_dt') == 'sklearn'
    model = exp.compare_models(include=['lr_cds_dt'])
    assert isinstance(model.regressor, SklearnLinearRegression)
    assert exp.get_engine('lr_cds_dt') == 'sklearn'
    model = exp.compare_models(include=['auto_arima'], engine={'auto_arima': 'statsforecast'})
    assert isinstance(model, StatsForecastAutoARIMA)
    assert exp.get_engine('auto_arima') == 'pmdarima'
    model = exp.compare_models(include=['auto_arima'])
    assert isinstance(model, PmdAutoARIMA)
    model = exp.compare_models(include=['lr_cds_dt'], engine={'lr_cds_dt': 'sklearnex'})
    assert isinstance(model.regressor, SklearnexLinearRegression)
    assert exp.get_engine('lr_cds_dt') == 'sklearn'
    model = exp.compare_models(include=['lr_cds_dt'])
    assert isinstance(model.regressor, SklearnLinearRegression)