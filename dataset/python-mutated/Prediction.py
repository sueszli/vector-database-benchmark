import os
import sys
import numpy
sys.path.insert(1, os.path.join(sys.path[0], '..'))

def predict(parameters_value, regressor_gp):
    if False:
        return 10
    '\n    Predict by Gaussian Process Model\n    '
    parameters_value = numpy.array(parameters_value).reshape(-1, len(parameters_value))
    (mu, sigma) = regressor_gp.predict(parameters_value, return_std=True)
    return (mu[0], sigma[0])