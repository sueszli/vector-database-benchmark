import sys
sys.path.insert(1, '../../../')
import h2o
from h2o.estimators.gam import H2OGeneralizedAdditiveEstimator
import pandas as pd
import numpy as np
import tempfile
from tests import pyunit_utils

def import_gam_mojo_regression(family, bs):
    if False:
        return 10
    np.random.seed(1234)
    n_rows = 10
    data = {'X1': np.random.randn(n_rows), 'X2': np.random.randn(n_rows), 'X3': np.random.randn(n_rows), 'W': np.random.choice([10, 20], size=n_rows), 'Y': np.random.choice([0, 0, 0, 0, 0, 10, 20, 30], size=n_rows) + 0.1}
    train = h2o.H2OFrame(pd.DataFrame(data))
    test = train.drop('W')
    print(train)
    h2o_model = H2OGeneralizedAdditiveEstimator(family=family, gam_columns=['X3'], weights_column='W', lambda_=0, tweedie_variance_power=1.5, bs=bs, tweedie_link_power=0)
    h2o_model.train(x=['X1', 'X2'], y='Y', training_frame=train)
    print(h2o_model)
    predict_w = h2o_model.predict(train)
    predict = h2o_model.predict(test)
    train_clone = h2o.H2OFrame(train.as_data_frame(use_pandas=True))
    model_perf_on_train = h2o_model.model_performance(test_data=train_clone)
    test_clone = h2o.H2OFrame(test.as_data_frame(use_pandas=True))
    model_perf_on_test = h2o_model.model_performance(test_data=test_clone)
    pyunit_utils.compare_frames_local(predict_w, predict, prob=1, tol=1e-06)
    original_model_filename = tempfile.mkdtemp()
    original_model_filename = h2o_model.save_mojo(original_model_filename)
    mojo_model = h2o.import_mojo(original_model_filename)
    predict_mojo_w = mojo_model.predict(train)
    predict_mojo = mojo_model.predict(test)
    pyunit_utils.compare_frames_local(predict_mojo_w, predict, prob=1, tol=1e-06)
    pyunit_utils.compare_frames_local(predict_mojo, predict, prob=1, tol=1e-06)
    mojo_perf_on_train = mojo_model.model_performance(test_data=train_clone)
    assert abs(mojo_perf_on_train._metric_json['MSE'] - model_perf_on_train._metric_json['MSE']) < 1e-06
    mojo_perf_on_test = mojo_model.model_performance(test_data=test_clone)
    assert abs(mojo_perf_on_test._metric_json['MSE'] - model_perf_on_test._metric_json['MSE']) < 1e-06

def import_gam_mojo_poisson():
    if False:
        for i in range(10):
            print('nop')
    import_gam_mojo_regression('poisson', [3])

def import_gam_mojo_tweedie():
    if False:
        print('Hello World!')
    import_gam_mojo_regression('tweedie', [1])

def import_gam_mojo_gamma():
    if False:
        return 10
    import_gam_mojo_regression('gamma', [2])

def import_gam_mojo_gaussian():
    if False:
        i = 10
        return i + 15
    import_gam_mojo_regression('gaussian', [0])
pyunit_utils.run_tests([import_gam_mojo_poisson, import_gam_mojo_tweedie, import_gam_mojo_gamma, import_gam_mojo_gaussian])