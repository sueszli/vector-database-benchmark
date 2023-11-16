"""
lib_acquisition_function.py
"""
import sys
import numpy
from scipy.stats import norm
from scipy.optimize import minimize
from . import lib_data

def next_hyperparameter_expected_improvement(fun_prediction, fun_prediction_args, x_bounds, x_types, samples_y_aggregation, minimize_starting_points, minimize_constraints_fun=None):
    if False:
        while True:
            i = 10
    '\n    "Expected Improvement" acquisition function\n    '
    best_x = None
    best_acquisition_value = None
    x_bounds_minmax = [[i[0], i[-1]] for i in x_bounds]
    x_bounds_minmax = numpy.array(x_bounds_minmax)
    for starting_point in numpy.array(minimize_starting_points):
        res = minimize(fun=_expected_improvement, x0=starting_point.reshape(1, -1), bounds=x_bounds_minmax, method='L-BFGS-B', args=(fun_prediction, fun_prediction_args, x_bounds, x_types, samples_y_aggregation, minimize_constraints_fun))
        if best_acquisition_value is None or res.fun < best_acquisition_value:
            res.x = numpy.ndarray.tolist(res.x)
            res.x = lib_data.match_val_type(res.x, x_bounds, x_types)
            if minimize_constraints_fun is None or minimize_constraints_fun(res.x) is True:
                best_acquisition_value = res.fun
                best_x = res.x
    outputs = None
    if best_x is not None:
        (mu, sigma) = fun_prediction(best_x, *fun_prediction_args)
        outputs = {'hyperparameter': best_x, 'expected_mu': mu, 'expected_sigma': sigma, 'acquisition_func': 'ei'}
    return outputs

def _expected_improvement(x, fun_prediction, fun_prediction_args, x_bounds, x_types, samples_y_aggregation, minimize_constraints_fun):
    if False:
        print('Hello World!')
    x = lib_data.match_val_type(x, x_bounds, x_types)
    expected_improvement = sys.maxsize
    if minimize_constraints_fun is None or minimize_constraints_fun(x) is True:
        (mu, sigma) = fun_prediction(x, *fun_prediction_args)
        loss_optimum = min(samples_y_aggregation)
        scaling_factor = -1
        with numpy.errstate(divide='ignore'):
            Z = scaling_factor * (mu - loss_optimum) / sigma
            expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
            expected_improvement = 0.0 if sigma == 0.0 else expected_improvement
        expected_improvement = -1 * expected_improvement
    return expected_improvement

def next_hyperparameter_lowest_confidence(fun_prediction, fun_prediction_args, x_bounds, x_types, minimize_starting_points, minimize_constraints_fun=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    "Lowest Confidence" acquisition function\n    '
    best_x = None
    best_acquisition_value = None
    x_bounds_minmax = [[i[0], i[-1]] for i in x_bounds]
    x_bounds_minmax = numpy.array(x_bounds_minmax)
    for starting_point in numpy.array(minimize_starting_points):
        res = minimize(fun=_lowest_confidence, x0=starting_point.reshape(1, -1), bounds=x_bounds_minmax, method='L-BFGS-B', args=(fun_prediction, fun_prediction_args, x_bounds, x_types, minimize_constraints_fun))
        if best_acquisition_value is None or res.fun < best_acquisition_value:
            res.x = numpy.ndarray.tolist(res.x)
            res.x = lib_data.match_val_type(res.x, x_bounds, x_types)
            if minimize_constraints_fun is None or minimize_constraints_fun(res.x) is True:
                best_acquisition_value = res.fun
                best_x = res.x
    outputs = None
    if best_x is not None:
        (mu, sigma) = fun_prediction(best_x, *fun_prediction_args)
        outputs = {'hyperparameter': best_x, 'expected_mu': mu, 'expected_sigma': sigma, 'acquisition_func': 'lc'}
    return outputs

def _lowest_confidence(x, fun_prediction, fun_prediction_args, x_bounds, x_types, minimize_constraints_fun):
    if False:
        print('Hello World!')
    x = lib_data.match_val_type(x, x_bounds, x_types)
    ci = sys.maxsize
    if minimize_constraints_fun is None or minimize_constraints_fun(x) is True:
        (mu, sigma) = fun_prediction(x, *fun_prediction_args)
        ci = sigma * 1.96 * 2 / mu
        ci = -1 * ci
    return ci

def next_hyperparameter_lowest_mu(fun_prediction, fun_prediction_args, x_bounds, x_types, minimize_starting_points, minimize_constraints_fun=None):
    if False:
        print('Hello World!')
    '\n    "Lowest Mu" acquisition function\n    '
    best_x = None
    best_acquisition_value = None
    x_bounds_minmax = [[i[0], i[-1]] for i in x_bounds]
    x_bounds_minmax = numpy.array(x_bounds_minmax)
    for starting_point in numpy.array(minimize_starting_points):
        res = minimize(fun=_lowest_mu, x0=starting_point.reshape(1, -1), bounds=x_bounds_minmax, method='L-BFGS-B', args=(fun_prediction, fun_prediction_args, x_bounds, x_types, minimize_constraints_fun))
        if best_acquisition_value is None or res.fun < best_acquisition_value:
            res.x = numpy.ndarray.tolist(res.x)
            res.x = lib_data.match_val_type(res.x, x_bounds, x_types)
            if minimize_constraints_fun is None or minimize_constraints_fun(res.x) is True:
                best_acquisition_value = res.fun
                best_x = res.x
    outputs = None
    if best_x is not None:
        (mu, sigma) = fun_prediction(best_x, *fun_prediction_args)
        outputs = {'hyperparameter': best_x, 'expected_mu': mu, 'expected_sigma': sigma, 'acquisition_func': 'lm'}
    return outputs

def _lowest_mu(x, fun_prediction, fun_prediction_args, x_bounds, x_types, minimize_constraints_fun):
    if False:
        while True:
            i = 10
    '\n    Calculate the lowest mu\n    '
    x = lib_data.match_val_type(x, x_bounds, x_types)
    mu = sys.maxsize
    if minimize_constraints_fun is None or minimize_constraints_fun(x) is True:
        (mu, _) = fun_prediction(x, *fun_prediction_args)
    return mu