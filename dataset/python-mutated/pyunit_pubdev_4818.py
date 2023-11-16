import h2o
from tests import pyunit_utils
from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.exceptions import H2OResponseError, H2OValueError
from h2o.grid.grid_search import H2OGridSearch
TRAIN_DATASET = pyunit_utils.locate('smalldata/iris/iris_train.csv')
TEST_DATASET = pyunit_utils.locate('smalldata/iris/iris_test.csv')

def init_data():
    if False:
        print('Hello World!')
    train = h2o.import_file(TRAIN_DATASET)
    test = h2o.import_file(TEST_DATASET)
    return {'predictors': train.columns, 'response': 'species', 'train': train, 'test': test}

def test_grid_search():
    if False:
        for i in range(10):
            print('nop')
    'This function tests, whether H2O GridSearch with XGBoostEstimator\n        can be passed unknown argument, which may possibly crash the H2O instance\n    '
    assert H2OXGBoostEstimator.available(), 'H2O XGBoost is not available! Please check machine env!'
    data = init_data()
    hyper_parameters = {'ntrees': 1, 'seed': 1, 'col_sample_rate_change_per_level': [0.9, 0.3, 0.2, 0.4]}
    raised = False
    try:
        grid_search = H2OGridSearch(H2OXGBoostEstimator, hyper_params=hyper_parameters)
        grid_search.train(x=data['predictors'], y=data['response'], training_frame=data['train'], validation_frame=data['test'])
    except H2OResponseError:
        raised = True
    assert raised is True, 'H2O should throw an exception if unknown parameter is passed to GridSearch with XGBoostEstimator!'

def test_estimator():
    if False:
        i = 10
        return i + 15
    data = init_data()
    raised = False
    try:
        estimator_xgb = H2OXGBoostEstimator(col_sample_rate_change_per_level=0.9, seed=1234)
        estimator_xgb.train(x=data['predictors'], y=data['response'], training_frame=data['train'], validation_frame=data['test'])
    except TypeError as e:
        raised = True
        assert "unexpected keyword argument 'col_sample_rate_change_per_level'" in str(e)
    assert raised, 'H2O should throw an exception if unknown parameter is passed to XGBoostEstimator!'
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_grid_search)
    pyunit_utils.standalone_test(test_estimator)
else:
    test_grid_search()
    test_estimator()