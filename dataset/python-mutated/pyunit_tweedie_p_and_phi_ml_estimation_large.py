import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

def test_tweedie_p_and_phi_estimation_2p6_disp2_est():
    if False:
        return 10
    training_data = h2o.import_file(pyunit_utils.locate('bigdata/laptop/glm_test/tweedie_p2p6_disp2_5Cols_10krows_est1p94.csv'))
    Y = 'resp'
    x = ['abs.C1.', 'abs.C2.', 'abs.C3.', 'abs.C4.', 'abs.C5.']
    training_data = training_data[training_data[Y] > 0, :]
    trueDisp = 2.6
    model_ml = H2OGeneralizedLinearEstimator(family='tweedie', fix_dispersion_parameter=False, fix_tweedie_variance_power=False, tweedie_variance_power=trueDisp, lambda_=0, compute_p_values=False, init_dispersion_parameter=2.0, dispersion_parameter_method='ml', seed=12345)
    model_ml.train(training_frame=training_data, x=x, y=Y)
    print('p = 2.6, phi = 2 converged to p =', model_ml.actual_params['tweedie_variance_power'], '; phi =', model_ml.actual_params['init_dispersion_parameter'])
    assert abs(model_ml.actual_params['tweedie_variance_power'] - 2.6724) < 0.0002
    assert abs(model_ml.actual_params['init_dispersion_parameter'] - 1.9585) < 0.0002

def test_tweedie_p_and_phi_estimation_3_no_link_power_est():
    if False:
        for i in range(10):
            print('nop')
    training_data = h2o.import_file(pyunit_utils.locate('bigdata/laptop/glm_test/tweedie_p3_phi1_10KRows.csv'))
    Y = 'x'
    x = ['abs.C1.', 'abs.C2.', 'abs.C3.', 'abs.C4.', 'abs.C5.']
    training_data = training_data[training_data[Y] > 0, :]
    model_ml = H2OGeneralizedLinearEstimator(family='tweedie', fix_dispersion_parameter=False, fix_tweedie_variance_power=False, tweedie_variance_power=2.5, lambda_=0, compute_p_values=False, dispersion_parameter_method='ml', seed=12345)
    model_ml.train(training_frame=training_data, x=x, y=Y)
    print('p = 3, phi = 1 converged to p =', model_ml.actual_params['tweedie_variance_power'], '; phi =', model_ml.actual_params['init_dispersion_parameter'])
    assert abs(model_ml.actual_params['tweedie_variance_power'] - 2.98921) < 0.0002
    assert abs(model_ml.actual_params['init_dispersion_parameter'] - 0.9918) < 0.0002

def test_tweedie_p_and_phi_estimation_5_no_link_power_est():
    if False:
        while True:
            i = 10
    training_data = h2o.import_file(pyunit_utils.locate('bigdata/laptop/glm_test/tweedie_p5_phi1_10KRows.csv'))
    Y = 'x'
    x = ['abs.C1.', 'abs.C2.', 'abs.C3.', 'abs.C4.', 'abs.C5.']
    training_data = training_data[training_data[Y] > 0, :]
    model_ml = H2OGeneralizedLinearEstimator(family='tweedie', fix_dispersion_parameter=False, fix_tweedie_variance_power=False, tweedie_variance_power=1.5, lambda_=0, compute_p_values=False, dispersion_parameter_method='ml', seed=12345)
    model_ml.train(training_frame=training_data, x=x, y=Y)
    print('p = 5, phi = 1 converged to p =', model_ml.actual_params['tweedie_variance_power'], '; phi =', model_ml.actual_params['init_dispersion_parameter'])
    assert abs(model_ml.actual_params['tweedie_variance_power'] - 5.02711) < 0.0002
    assert abs(model_ml.actual_params['init_dispersion_parameter'] - 1.01034) < 0.0002

def test_tweedie_p_and_phi_estimation_5_phi_0p5_no_link_power_est():
    if False:
        return 10
    training_data = h2o.import_file(pyunit_utils.locate('bigdata/laptop/glm_test/tweedie_p5_phi0p5_10KRows.csv'))
    Y = 'x'
    x = ['abs.C1.', 'abs.C2.', 'abs.C3.', 'abs.C4.', 'abs.C5.']
    training_data = training_data[training_data[Y] > 0, :]
    model_ml = H2OGeneralizedLinearEstimator(family='tweedie', fix_dispersion_parameter=False, fix_tweedie_variance_power=False, tweedie_variance_power=1.5, init_dispersion_parameter=0.5, lambda_=0, compute_p_values=False, dispersion_parameter_method='ml', seed=12345)
    model_ml.train(training_frame=training_data, x=x, y=Y)
    print('p = 5, phi = 0.5 converged to p =', model_ml.actual_params['tweedie_variance_power'], '; phi =', model_ml.actual_params['init_dispersion_parameter'])
    assert abs(model_ml.actual_params['tweedie_variance_power'] - 4.94311) < 0.0002
    assert abs(model_ml.actual_params['init_dispersion_parameter'] - 0.488076) < 0.0002

def test_tweedie_p_and_phi_estimation_3_phi_1p5_no_link_power_est():
    if False:
        while True:
            i = 10
    training_data = h2o.import_file(pyunit_utils.locate('bigdata/laptop/glm_test/tweedie_p3_phi1p5_10KRows.csv'))
    Y = 'x'
    x = ['abs.C1.', 'abs.C2.', 'abs.C3.', 'abs.C4.', 'abs.C5.']
    training_data = training_data[training_data[Y] > 0, :]
    model_ml = H2OGeneralizedLinearEstimator(family='tweedie', fix_dispersion_parameter=False, fix_tweedie_variance_power=False, tweedie_variance_power=1.5, init_dispersion_parameter=1.5, lambda_=0, compute_p_values=False, dispersion_parameter_method='ml', seed=12345)
    model_ml.train(training_frame=training_data, x=x, y=Y)
    print('p = 3, phi = 1.5 converged to p =', model_ml.actual_params['tweedie_variance_power'], '; phi =', model_ml.actual_params['init_dispersion_parameter'])
    assert abs(model_ml.actual_params['tweedie_variance_power'] - 2.997) < 0.0002
    assert abs(model_ml.actual_params['init_dispersion_parameter'] - 1.481355) < 0.0002

def test_tweedie_p_and_phi_estimation_3_phi_0p5_no_link_power_est():
    if False:
        return 10
    training_data = h2o.import_file(pyunit_utils.locate('bigdata/laptop/glm_test/tweedie_p3_phi0p5_10KRows.csv'))
    Y = 'x'
    x = ['abs.C1.', 'abs.C2.', 'abs.C3.', 'abs.C4.', 'abs.C5.']
    training_data = training_data[training_data[Y] > 0, :]
    model_ml = H2OGeneralizedLinearEstimator(family='tweedie', fix_dispersion_parameter=False, fix_tweedie_variance_power=False, tweedie_variance_power=1.5, init_dispersion_parameter=0.5, lambda_=0, compute_p_values=False, dispersion_parameter_method='ml', seed=12345)
    model_ml.train(training_frame=training_data, x=x, y=Y)
    print('p = 3, phi = 0.5 converged to p =', model_ml.actual_params['tweedie_variance_power'], '; phi =', model_ml.actual_params['init_dispersion_parameter'])
    assert abs(model_ml.actual_params['tweedie_variance_power'] - 3.0038) < 0.0002
    assert abs(model_ml.actual_params['init_dispersion_parameter'] - 0.50215) < 0.0002

def test_tweedie_p_and_phi_estimation_2p5_phi_2p5_no_link_power_est():
    if False:
        for i in range(10):
            print('nop')
    training_data = h2o.import_file(pyunit_utils.locate('bigdata/laptop/glm_test/tweedie_p2p5_phi2p5_5Cols_10KRows.csv'))
    Y = 'resp'
    x = ['abs.C1.', 'abs.C2.', 'abs.C3.', 'abs.C4.', 'abs.C5.']
    training_data = training_data[training_data[Y] > 0, :]
    model_ml = H2OGeneralizedLinearEstimator(family='tweedie', fix_dispersion_parameter=False, fix_tweedie_variance_power=False, tweedie_variance_power=1.5, init_dispersion_parameter=2.5, lambda_=0, compute_p_values=False, dispersion_parameter_method='ml', seed=12345)
    model_ml.train(training_frame=training_data, x=x, y=Y)
    print('p = 2.5, phi = 2.5 converged to p =', model_ml.actual_params['tweedie_variance_power'], '; phi =', model_ml.actual_params['init_dispersion_parameter'])
    assert abs(model_ml.actual_params['tweedie_variance_power'] - 2.592835) < 0.0002
    assert abs(model_ml.actual_params['init_dispersion_parameter'] - 2.63012) < 0.0002

def measure_time(t):
    if False:
        while True:
            i = 10

    def _():
        if False:
            return 10
        import time
        start = time.monotonic()
        t()
        print(f'The {t.__name__} took {time.monotonic() - start}s')
    _.__name__ = t.__name__
    return _

def run_random_test():
    if False:
        for i in range(10):
            print('nop')
    import random
    tests = [test_tweedie_p_and_phi_estimation_2p6_disp2_est, test_tweedie_p_and_phi_estimation_2p5_phi_2p5_no_link_power_est, test_tweedie_p_and_phi_estimation_3_phi_0p5_no_link_power_est, test_tweedie_p_and_phi_estimation_3_no_link_power_est, test_tweedie_p_and_phi_estimation_3_phi_1p5_no_link_power_est, test_tweedie_p_and_phi_estimation_5_no_link_power_est, test_tweedie_p_and_phi_estimation_5_phi_0p5_no_link_power_est]
    return [random.choice(tests)]
pyunit_utils.run_tests(run_random_test())