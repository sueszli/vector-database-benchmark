import pandas as pd
import pycaret.classification
import pycaret.datasets
import pycaret.regression

def test_check_fairness_binary_classification():
    if False:
        print('Hello World!')
    data = pycaret.datasets.get_data('income')
    pycaret.classification.setup(data, target='income >50K', html=False, n_jobs=1)
    lightgbm = pycaret.classification.create_model('lightgbm', fold=3)
    lightgbm_fairness = pycaret.classification.check_fairness(lightgbm, ['sex'])
    assert isinstance(lightgbm_fairness, pd.DataFrame)

def test_check_fairness_multiclass_classification():
    if False:
        print('Hello World!')
    data = pycaret.datasets.get_data('iris')
    pycaret.classification.setup(data, target='species', html=False, n_jobs=1, train_size=0.8)
    lightgbm = pycaret.classification.create_model('lightgbm', cross_validation=False)
    lightgbm_fairness = pycaret.classification.check_fairness(lightgbm, ['sepal_length'])
    assert isinstance(lightgbm_fairness, pd.DataFrame)

def test_check_fairness_regression():
    if False:
        while True:
            i = 10
    data = pycaret.datasets.get_data('boston')
    pycaret.regression.setup(data, target='medv', html=False, n_jobs=1)
    lightgbm = pycaret.regression.create_model('lightgbm', fold=3)
    lightgbm_fairness = pycaret.regression.check_fairness(lightgbm, ['chas'])
    assert isinstance(lightgbm_fairness, pd.DataFrame)
if __name__ == '__main__':
    test_check_fairness_binary_classification()
    test_check_fairness_multiclass_classification()
    test_check_fairness_regression()