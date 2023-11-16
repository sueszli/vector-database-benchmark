import sys
import pytest
import pycaret.classification
import pycaret.datasets
import pycaret.regression
if sys.platform == 'win32':
    pytest.skip('Skipping test module on Windows', allow_module_level=True)

def test_classification_create_docker():
    if False:
        i = 10
        return i + 15
    data = pycaret.datasets.get_data('blood')
    pycaret.classification.setup(data, target='Class', html=False, n_jobs=1)
    lr = pycaret.classification.create_model('lr')
    pycaret.classification.create_api(lr, 'blood_api')
    pycaret.classification.create_docker('blood_api')
    assert 1 == 1

def test_regression_create_docker():
    if False:
        print('Hello World!')
    data = pycaret.datasets.get_data('boston')
    pycaret.regression.setup(data, target='medv', html=False, n_jobs=1)
    lr = pycaret.regression.create_model('lr')
    pycaret.regression.create_api(lr, 'boston_api')
    pycaret.regression.create_docker('boston_api')
    assert 1 == 1
if __name__ == '__main__':
    test_classification_create_docker()
    test_regression_create_docker()