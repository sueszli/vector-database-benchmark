import unittest
import numpy as np
import pandas as pd
from streamlit.util import index_

class Index_Test(unittest.TestCase):

    def test_index_list(self):
        if False:
            print('Hello World!')
        self.assertEqual(index_([1, 2, 3, 4], 1), 0)
        self.assertEqual(index_([1, 2, 3, 4], 4), 3)

    def test_index_list_fails(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            index_([1, 2, 3, 4], 5)

    def test_index_tuple(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(index_((1, 2, 3, 4), 1), 0)
        self.assertEqual(index_((1, 2, 3, 4), 4), 3)

    def test_index_tuple_fails(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            index_((1, 2, 3, 4), 5)

    def test_index_numpy_array(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(index_(np.array([1, 2, 3, 4]), 1), 0)
        self.assertEqual(index_(np.array([1, 2, 3, 4]), 4), 3)

    def test_index_numpy_array_fails(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            index_(np.array([1, 2, 3, 4]), 5)

    def test_index_pandas_series(self):
        if False:
            print('Hello World!')
        self.assertEqual(index_(pd.Series([1, 2, 3, 4]), 1), 0)
        self.assertEqual(index_(pd.Series([1, 2, 3, 4]), 4), 3)

    def test_index_pandas_series_fails(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            index_(pd.Series([1, 2, 3, 4]), 5)