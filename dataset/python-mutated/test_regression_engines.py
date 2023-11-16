import uuid
import daal4py
import pytest
import sklearn
import pycaret.datasets
import pycaret.regression

def test_engines_setup_global_args():
    if False:
        return 10
    'Tests the setting of engines using global arguments in setup.'
    boston_dataframe = pycaret.datasets.get_data('boston')
    exp = pycaret.regression.RegressionExperiment()
    exp.setup(data=boston_dataframe, target='medv', remove_multicollinearity=True, multicollinearity_threshold=0.95, log_experiment=True, html=False, session_id=123, n_jobs=1, experiment_name=uuid.uuid4().hex, engine={'lr': 'sklearnex'})
    assert exp.get_engine('lr') == 'sklearnex'
    model = exp.create_model('lr')
    print(type(model))
    assert isinstance(model, daal4py.sklearn.linear_model._linear.LinearRegression)

def test_engines_global_methods():
    if False:
        i = 10
        return i + 15
    'Tests the setting of engines using methods like set_engine (global changes).'
    boston_dataframe = pycaret.datasets.get_data('boston')
    exp = pycaret.regression.RegressionExperiment()
    exp.setup(data=boston_dataframe, target='medv', remove_multicollinearity=True, multicollinearity_threshold=0.95, log_experiment=True, html=False, session_id=123, n_jobs=1, experiment_name=uuid.uuid4().hex, engine={'lr': 'sklearnex'})
    assert exp.get_engine('lr') == 'sklearnex'
    exp._set_engine('lr', 'sklearn')
    assert exp.get_engine('lr') == 'sklearn'
    model = exp.create_model('lr')
    assert isinstance(model, sklearn.linear_model._base.LinearRegression)

def test_create_model_engines_local_args():
    if False:
        print('Hello World!')
    'Tests the setting of engines for create_model using local args.'
    boston_dataframe = pycaret.datasets.get_data('boston')
    exp = pycaret.regression.RegressionExperiment()
    exp.setup(data=boston_dataframe, target='medv', remove_multicollinearity=True, multicollinearity_threshold=0.95, log_experiment=True, html=False, session_id=123, n_jobs=1, experiment_name=uuid.uuid4().hex)
    assert exp.get_engine('lr') == 'sklearn'
    model = exp.create_model('lr')
    assert isinstance(model, sklearn.linear_model._base.LinearRegression)
    model = exp.create_model('lr', engine='sklearnex')
    assert isinstance(model, daal4py.sklearn.linear_model._linear.LinearRegression)
    assert exp.get_engine('lr') == 'sklearn'

def test_compare_models_engines_local_args():
    if False:
        i = 10
        return i + 15
    'Tests the setting of engines for compare_models using local args.'
    boston_dataframe = pycaret.datasets.get_data('boston')
    exp = pycaret.regression.RegressionExperiment()
    exp.setup(data=boston_dataframe, target='medv', remove_multicollinearity=True, multicollinearity_threshold=0.95, log_experiment=True, html=False, session_id=123, n_jobs=1, experiment_name=uuid.uuid4().hex)
    assert exp.get_engine('lr') == 'sklearn'
    model = exp.compare_models(include=['lr'])
    assert isinstance(model, sklearn.linear_model._base.LinearRegression)
    assert exp.get_engine('lr') == 'sklearn'
    model = exp.compare_models(include=['lr'], engine={'lr': 'sklearnex'})
    assert isinstance(model, daal4py.sklearn.linear_model._linear.LinearRegression)
    assert exp.get_engine('lr') == 'sklearn'
    model = exp.compare_models(include=['lr'])
    assert isinstance(model, sklearn.linear_model._base.LinearRegression)

@pytest.mark.parametrize('algo', ('lr', 'lasso', 'ridge', 'en', 'knn', 'svm'))
def test_sklearnex_model(algo: str):
    if False:
        print('Hello World!')
    boston_dataframe = pycaret.datasets.get_data('boston')
    exp = pycaret.regression.RegressionExperiment()
    exp.setup(data=boston_dataframe, target='medv', remove_multicollinearity=True, multicollinearity_threshold=0.95, log_experiment=True, html=False, session_id=123, n_jobs=1, experiment_name=uuid.uuid4().hex)
    model = exp.create_model(algo)
    parent_library = model.__module__
    assert parent_library.startswith('sklearn')
    model = exp.create_model(algo, engine='sklearnex')
    parent_library = model.__module__
    assert parent_library.startswith('sklearnex') or parent_library.startswith('daal4py')