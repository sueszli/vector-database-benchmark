import pytest
import numpy as np
from art.estimators.classification.blackbox import BlackBoxClassifierNeuralNetwork, BlackBoxClassifier
from tests.utils import ARTTestException

def test_blackbox_existing_predictions(art_warning, get_mnist_dataset):
    if False:
        return 10
    try:
        (_, (x_test, y_test)) = get_mnist_dataset
        limited_x_test = x_test[:500]
        limited_y_test = y_test[:500]
        bb = BlackBoxClassifier((limited_x_test, limited_y_test), (28, 28, 1), 10, clip_values=(0, 255))
        assert np.array_equal(bb.predict(limited_x_test), limited_y_test)
        with pytest.raises(ValueError):
            bb.predict(x_test[:600])
    except ARTTestException as e:
        art_warning(e)

def test_blackbox_existing_predictions_fuzzy(art_warning):
    if False:
        i = 10
        return i + 15
    try:
        x = np.array([0, 3])
        fuzzy_x = np.array([0, 3.00001])
        y = np.array([[1, 0], [0, 1]])
        bb = BlackBoxClassifier((x, y), (1,), 2, fuzzy_float_compare=True)
        assert np.array_equal(bb.predict(fuzzy_x), y)
    except ARTTestException as e:
        art_warning(e)

def test_blackbox_nn_existing_predictions(art_warning, get_mnist_dataset):
    if False:
        i = 10
        return i + 15
    try:
        (_, (x_test, y_test)) = get_mnist_dataset
        limited_x_test = x_test[:500]
        limited_y_test = y_test[:500]
        bb = BlackBoxClassifierNeuralNetwork((limited_x_test, limited_y_test), (28, 28, 1), 10, clip_values=(0, 255))
        assert np.array_equal(bb.predict(limited_x_test), limited_y_test)
        with pytest.raises(ValueError):
            bb.predict(x_test[:600])
    except ARTTestException as e:
        art_warning(e)

def test_blackbox_nn_existing_predictions_fuzzy(art_warning):
    if False:
        for i in range(10):
            print('nop')
    try:
        x = np.array([0, 3])
        fuzzy_x = np.array([0, 3.00001])
        y = np.array([[1, 0], [0, 1]])
        bb = BlackBoxClassifierNeuralNetwork((x, y), (1,), 2, fuzzy_float_compare=True)
        assert np.array_equal(bb.predict(fuzzy_x), y)
    except ARTTestException as e:
        art_warning(e)