import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from tests.pyunit_utils import CustomMaeFunc, CustomRmseFunc, assert_correct_custom_metric, regression_model, multinomial_model, binomial_model
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.exceptions import H2OResponseError

def custom_mae_mm():
    if False:
        while True:
            i = 10
    return h2o.upload_custom_metric(CustomMaeFunc, func_name='mae', func_file='mm_mae.py')

def custom_rmse_mm():
    if False:
        return 10
    return h2o.upload_custom_metric(CustomRmseFunc, func_name='rmse', func_file='mm_rmse.py')

def test_custom_metric_computation_regression():
    if False:
        i = 10
        return i + 15
    (model, f_test) = regression_model(H2ODeepLearningEstimator, custom_mae_mm())
    print(model)
    assert_correct_custom_metric(model, f_test, 'mae', 'Regression on prostate')

def test_custom_metric_computation_binomial():
    if False:
        for i in range(10):
            print('nop')
    (model, f_test) = binomial_model(H2ODeepLearningEstimator, custom_rmse_mm())
    print(model)
    assert_correct_custom_metric(model, f_test, 'rmse', 'Binomial on prostate')

def test_custom_metric_computation_multinomial():
    if False:
        return 10
    (model, f_test) = multinomial_model(H2ODeepLearningEstimator, custom_rmse_mm())
    print(model)
    assert_correct_custom_metric(model, f_test, 'rmse', 'Multinomial on iris')

def test_custom_metric_computation_multinomial_autoencoder():
    if False:
        print('Hello World!')
    params = {'autoencoder': True}
    try:
        multinomial_model(H2ODeepLearningEstimator, custom_rmse_mm(), params)
    except H2OResponseError as e:
        assert 'Custom metric is not supported for Autoencoder.' in str(e)
__TESTS__ = [test_custom_metric_computation_binomial, test_custom_metric_computation_regression, test_custom_metric_computation_multinomial, test_custom_metric_computation_multinomial_autoencoder]
if __name__ == '__main__':
    for func in __TESTS__:
        pyunit_utils.standalone_test(func)
else:
    for func in __TESTS__:
        func()