import unittest
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from flaml.automl import AutoML
from flaml.automl.model import XGBoostSklearnEstimator
from flaml import tune
dataset = 'credit-g'

class XGBoost2D(XGBoostSklearnEstimator):

    @classmethod
    def search_space(cls, data_size, task):
        if False:
            for i in range(10):
                print('nop')
        upper = min(32768, int(data_size))
        return {'n_estimators': {'domain': tune.lograndint(lower=4, upper=upper), 'init_value': 4}, 'max_leaves': {'domain': tune.lograndint(lower=4, upper=upper), 'init_value': 4}}

def _test_simple(method=None, size_ratio=1.0):
    if False:
        print('Hello World!')
    automl = AutoML()
    automl.add_learner(learner_name='XGBoost2D', learner_class=XGBoost2D)
    (X, y) = fetch_openml(name=dataset, return_X_y=True)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=42)
    final_size = int(len(y_train) * size_ratio)
    X_train = X_train[:final_size]
    y_train = y_train[:final_size]
    automl_settings = {'estimator_list': ['XGBoost2D'], 'task': 'classification', 'log_file_name': f'test/xgboost2d_{dataset}_{method}_{final_size}.log', 'n_jobs': 1, 'hpo_method': method, 'log_type': 'all', 'time_budget': 3600}
    automl.fit(X_train=X_train, y_train=y_train, **automl_settings)

def _test_grid_1():
    if False:
        i = 10
        return i + 15
    _test_simple(method='grid', size_ratio=1.0 / 3.0)

def _test_grid_2():
    if False:
        i = 10
        return i + 15
    _test_simple(method='grid', size_ratio=2.0 / 3.0)

def _test_grid_4():
    if False:
        i = 10
        return i + 15
    _test_simple(method='grid', size_ratio=0.5)

def _test_grid_3():
    if False:
        while True:
            i = 10
    _test_simple(method='grid', size_ratio=1.0)
if __name__ == '__main__':
    unittest.main()