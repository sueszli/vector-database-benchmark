"""Tests for NaiveBayes module."""
import copy
import unittest
from Bio import NaiveBayes
import numpy as np
try:
    hash(np.float64(123.456))
except TypeError:
    from Bio import MissingPythonDependencyError
    raise MissingPythonDependencyError('Please update NumPy if you want to use Bio.NaiveBayes (under this version np.float64 is unhashable).') from None
del np

class CarTest(unittest.TestCase):

    def test_car_data(self):
        if False:
            while True:
                i = 10
        'Simple example using car data.'
        xcar = [['Red', 'Sports', 'Domestic'], ['Red', 'Sports', 'Domestic'], ['Red', 'Sports', 'Domestic'], ['Yellow', 'Sports', 'Domestic'], ['Yellow', 'Sports', 'Imported'], ['Yellow', 'SUV', 'Imported'], ['Yellow', 'SUV', 'Imported'], ['Yellow', 'SUV', 'Domestic'], ['Red', 'SUV', 'Imported'], ['Red', 'Sports', 'Imported']]
        ycar = ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes']
        carmodel = NaiveBayes.train(xcar, ycar)
        self.assertEqual('Yes', NaiveBayes.classify(carmodel, ['Red', 'Sports', 'Domestic']))
        self.assertEqual('No', NaiveBayes.classify(carmodel, ['Red', 'SUV', 'Domestic']))

class NaiveBayesTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.xs = [[6, 180, 12], [5.92, 190, 11], [5.58, 170, 12], [5.92, 165, 10], [5, 100, 6], [5.5, 150, 8], [5.42, 130, 7], [5.75, 150, 9]]
        self.ys = ['male', 'male', 'male', 'male', 'female', 'female', 'female', 'female']
        self.model = NaiveBayes.train(self.xs, self.ys)
        self.test = [6, 130, 8]

    def test_train_function_no_training_set(self):
        if False:
            print('Hello World!')
        self.assertRaises(ValueError, NaiveBayes.train, [], self.ys)

    def test_train_function_input_lengths(self):
        if False:
            while True:
                i = 10
        ys = copy.copy(self.ys)
        ys.pop()
        self.assertRaises(ValueError, NaiveBayes.train, self.xs, ys)

    def test_train_function_uneven_dimension_of_training_set(self):
        if False:
            i = 10
            return i + 15
        xs = copy.copy(self.xs)
        xs[0] = [1]
        self.assertRaises(ValueError, NaiveBayes.train, xs, self.ys)

    def test_train_function_with_priors(self):
        if False:
            for i in range(10):
                print('nop')
        model = NaiveBayes.train(self.xs, self.ys, priors={'male': 0.1, 'female': 0.9})
        result = NaiveBayes.calculate(model, self.test, scale=True)
        expected = -692.0
        self.assertEqual(expected, round(result['male']))

    def test_classify_function(self):
        if False:
            while True:
                i = 10
        expected = 'female'
        result = NaiveBayes.classify(self.model, self.test)
        self.assertEqual(expected, result)

    def test_calculate_function_wrong_dimensionality(self):
        if False:
            while True:
                i = 10
        xs = self.xs[0]
        xs.append(100)
        self.assertRaises(ValueError, NaiveBayes.calculate, self.model, xs)

    def test_calculate_function_with_scale(self):
        if False:
            i = 10
            return i + 15
        result = NaiveBayes.calculate(self.model, self.test, scale=True)
        expected = -689.0
        self.assertEqual(expected, round(result['male']))
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)