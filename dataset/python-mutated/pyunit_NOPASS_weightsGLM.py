from builtins import zip
from builtins import range
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
import random
import copy

def weights_check():
    if False:
        i = 10
        return i + 15

    def check_same(data1, data2):
        if False:
            while True:
                i = 10
        glm1_regression = H2OGeneralizedLinearEstimator()
        glm1_regression.train(x=list(range(2, 20)), y=1, training_frame=data1)
        glm2_regression = H2OGeneralizedLinearEstimator(weights_column='weights')
        glm2_regression.train(x=list(range(2, 21)), y=1, training_frame=data2)
        glm1_binomial = H2OGeneralizedLinearEstimator()
        glm1_binomial.train(x=list(range(1, 20)), y=0, training_frame=data1)
        glm2_binomial = H2OGeneralizedLinearEstimator(weights_column='weights', family='binomial')
        glm2_binomial.train(x=list(range(1, 21)), y=0, training_frame=data2)
        assert abs(glm1_regression.mse() - glm2_regression.mse()) < 1e-06, "Expected mse's to be the same, but got {0}, and {1}".format(glm1_regression.mse(), glm2_regression.mse())
        assert abs(glm1_binomial.null_deviance() - glm2_binomial.null_deviance()) < 1e-06, 'Expected null deviances to be the same, but got {0}, and {1}'.format(glm1_binomial.null_deviance(), glm2_binomial.null_deviance())
        assert abs(glm1_binomial.residual_deviance() - glm2_binomial.residual_deviance()) < 1e-06, 'Expected residual deviances to be the same, but got {0}, and {1}'.format(glm1_binomial.residual_deviance(), glm2_binomial.residual_deviance())
    data = [['ab'[random.randint(0, 1)] if c == 0 else random.gauss(0, 1) for c in range(20)] for r in range(100)]
    h2o_data = h2o.H2OFrame(data)
    zero_weights = [[0] if random.randint(0, 1) else [1] for r in range(100)]
    h2o_zero_weights = h2o.H2OFrame(zero_weights)
    h2o_zero_weights.set_names(['weights'])
    h2o_data_zero_weights = h2o_data.cbind(h2o_zero_weights)
    h2o_data_zeros_removed = h2o_data[h2o_zero_weights['weights'] == 1]
    print('Checking that using some zero weights is equivalent to removing those observations:')
    print()
    check_same(h2o_data_zeros_removed, h2o_data_zero_weights)
    doubled_weights = [[1] if random.randint(0, 1) else [2] for r in range(100)]
    h2o_doubled_weights = h2o.H2OFrame(doubled_weights)
    h2o_doubled_weights.set_names(['weights'])
    h2o_data_doubled_weights = h2o_data.cbind(h2o_doubled_weights)
    doubled_data = copy.deepcopy(data)
    for (d, w) in zip(data, doubled_weights):
        if w[0] == 2:
            doubled_data.append(d)
    h2o_data_doubled = h2o.H2OFrame(doubled_data)
    print('Checking that doubling some weights is equivalent to doubling those observations:')
    print()
    check_same(h2o_data_doubled, h2o_data_doubled_weights)
if __name__ == '__main__':
    pyunit_utils.standalone_test(weights_check)
else:
    weights_check()