"""
Helper functions internally used in cleanlab.regression.
"""
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from typing import Tuple, Union

def assert_valid_prediction_inputs(labels: ArrayLike, predictions: ArrayLike, method: str) -> Tuple[np.ndarray, np.ndarray]:
    if False:
        while True:
            i = 10
    'Checks that ``labels``, ``predictions``, ``method`` are correctly formatted.'
    try:
        labels = np.asarray(labels)
    except:
        raise ValueError(f'labels must be array_like.')
    try:
        predictions = np.asarray(predictions)
    except:
        raise ValueError(f'predictions must be array_like.')
    valid_labels = check_dimension_and_datatype(check_input=labels, text='labels')
    valid_predictions = check_dimension_and_datatype(check_input=predictions, text='predictions')
    assert valid_labels.shape == valid_predictions.shape, f'Number of examples in labels {labels.shape} and predictions {predictions.shape} are not same.'
    check_missing_values(valid_labels, text='labels')
    check_missing_values(valid_predictions, text='predictions')
    scoring_methods = ['residual', 'outre']
    if method not in scoring_methods:
        raise ValueError(f"Specified method '{method}' must be one of: {scoring_methods}.")
    return (valid_labels, valid_predictions)

def assert_valid_regression_inputs(X: Union[np.ndarray, pd.DataFrame], y: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    if False:
        print('Hello World!')
    '\n    Checks that regression inputs are properly formatted and returns the inputs in numpy array format.\n    '
    try:
        X = np.asarray(X)
    except:
        raise ValueError(f'X must be array_like.')
    y = check_dimension_and_datatype(y, 'y')
    check_missing_values(y, text='y')
    if len(X) != len(y):
        raise ValueError('X and y must have same length.')
    return (X, y)

def check_dimension_and_datatype(check_input: ArrayLike, text: str) -> np.ndarray:
    if False:
        return 10
    '\n    Raises errors related to:\n    1. If input is empty\n    2. If input is not 1-D\n    3. If input is not numeric\n\n    If all the checks are passed, it returns the squeezed 1-D array required by the main algorithm.\n    '
    try:
        check_input = np.asarray(check_input)
    except:
        raise ValueError(f'{text} could not be converted to numpy array, check input.')
    if not check_input.size:
        raise ValueError(f'{text} cannot be empty array.')
    check_input = np.squeeze(check_input)
    if check_input.ndim != 1:
        raise ValueError(f'Expected 1-Dimensional inputs for {text}, got {check_input.ndim} dimensions.')
    if not np.issubdtype(check_input.dtype, np.number):
        raise ValueError(f'Expected {text} to contain numeric values, got values of type {check_input.dtype}.')
    return check_input

def check_missing_values(check_input: np.ndarray, text: str):
    if False:
        print('Hello World!')
    'Raise error if there are any missing values in Numpy array.'
    if np.isnan(check_input).any():
        raise ValueError(f'{text} cannot contain missing values.')