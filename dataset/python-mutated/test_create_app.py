import pycaret.classification
import pycaret.datasets
import pycaret.regression

def test_classification_create_app():
    if False:
        print('Hello World!')
    data = pycaret.datasets.get_data('blood')
    pycaret.classification.setup(data, target='Class', html=False, n_jobs=1)
    pycaret.classification.create_model('lr')
    assert 1 == 1

def test_regression_create_app():
    if False:
        i = 10
        return i + 15
    data = pycaret.datasets.get_data('boston')
    pycaret.regression.setup(data, target='medv', html=False, n_jobs=1)
    pycaret.regression.create_model('lr')
    assert 1 == 1
if __name__ == '__main__':
    test_classification_create_app()
    test_regression_create_app()