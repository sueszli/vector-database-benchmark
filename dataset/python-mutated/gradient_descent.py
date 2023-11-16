import math
import os
import numpy
from viztracer import VizCounter, VizTracer
train_data = (((5, 2, 3), 15), ((6, 5, 9), 25), ((11, 12, 13), 41), ((1, 1, 1), 8), ((11, 12, 13), 41))
test_data = (((515, 22, 13), 555), ((61, 35, 49), 150))
parameter_vector = [2, 4, 1, 5]
m = len(train_data)
LEARNING_RATE = 0.009

def _error(example_no, data_set='train'):
    if False:
        i = 10
        return i + 15
    '\n    :param data_set: train data or test data\n    :param example_no: example number whose error has to be checked\n    :return: error in example pointed by example number.\n    '
    return calculate_hypothesis_value(example_no, data_set) - output(example_no, data_set)

def _hypothesis_value(data_input_tuple):
    if False:
        print('Hello World!')
    "\n    Calculates hypothesis function value for a given input\n    :param data_input_tuple: Input tuple of a particular example\n    :return: Value of hypothesis function at that point.\n    Note that there is an 'biased input' whose value is fixed as 1.\n    It is not explicitly mentioned in input data.. But, ML hypothesis functions use it.\n    So, we have to take care of it separately. Line 36 takes care of it.\n    "
    hyp_val = 0
    for i in range(len(parameter_vector) - 1):
        hyp_val += data_input_tuple[i] * parameter_vector[i + 1]
    hyp_val += parameter_vector[0]
    return hyp_val

def output(example_no, data_set):
    if False:
        print('Hello World!')
    '\n    :param data_set: test data or train data\n    :param example_no: example whose output is to be fetched\n    :return: output for that example\n    '
    if data_set == 'train':
        return train_data[example_no][1]
    elif data_set == 'test':
        return test_data[example_no][1]

def calculate_hypothesis_value(example_no, data_set):
    if False:
        while True:
            i = 10
    '\n    Calculates hypothesis value for a given example\n    :param data_set: test data or train_data\n    :param example_no: example whose hypothesis value is to be calculated\n    :return: hypothesis value for that example\n    '
    if data_set == 'train':
        return _hypothesis_value(train_data[example_no][0])
    elif data_set == 'test':
        return _hypothesis_value(test_data[example_no][0])

def summation_of_cost_derivative(index, end=m):
    if False:
        while True:
            i = 10
    '\n    Calculates the sum of cost function derivative\n    :param index: index wrt derivative is being calculated\n    :param end: value where summation ends, default is m, number of examples\n    :return: Returns the summation of cost derivative\n    Note: If index is -1, this means we are calculating summation wrt to biased\n        parameter.\n    '
    summation_value = 0
    for i in range(end):
        if index == -1:
            summation_value += _error(i)
        else:
            summation_value += _error(i) * train_data[i][0][index]
    return summation_value

def get_cost_derivative(index):
    if False:
        while True:
            i = 10
    '\n    :param index: index of the parameter vector wrt to derivative is to be calculated\n    :return: derivative wrt to that index\n    Note: If index is -1, this means we are calculating summation wrt to biased\n        parameter.\n    '
    cost_derivative_value = summation_of_cost_derivative(index, m) / m
    return cost_derivative_value

def run_gradient_descent():
    if False:
        while True:
            i = 10
    global parameter_vector
    absolute_error_limit = 0.004
    relative_error_limit = 0
    j = 0
    while True:
        j += 1
        temp_parameter_vector = [0, 0, 0, 0]
        err = 0
        for i in range(0, len(parameter_vector)):
            cost_derivative = get_cost_derivative(i - 1)
            err += abs(cost_derivative)
            temp_parameter_vector[i] = parameter_vector[i] - LEARNING_RATE * cost_derivative
        counter.cost = math.log(1 + err)
        if numpy.allclose(parameter_vector, temp_parameter_vector, atol=absolute_error_limit, rtol=relative_error_limit):
            break
        parameter_vector = temp_parameter_vector
    print(('Number of iterations:', j))

def test_gradient_descent():
    if False:
        while True:
            i = 10
    for i in range(len(test_data)):
        print(('Actual output value:', output(i, 'test')))
        print(('Hypothesis output:', calculate_hypothesis_value(i, 'test')))
if __name__ == '__main__':
    with VizTracer(log_print=True, output_file=os.path.join(os.path.dirname(__file__), '../', 'json/gradient_descent.json'), file_info=True) as tracer:
        counter = VizCounter(tracer, 'log(1 + cost)')
        run_gradient_descent()
        test_gradient_descent()