"""Tests coercing various objects to DataFrames"""
import unittest
import numpy as np
import pandas as pd
import pyarrow as pa
from streamlit import type_util

class DataFrameCoercionTest(unittest.TestCase):

    def test_dict_of_lists(self):
        if False:
            while True:
                i = 10
        'Test that a DataFrame can be constructed from a dict\n        of equal-length lists\n        '
        d = {'a': [1], 'b': [2], 'c': [3]}
        df = type_util.convert_anything_to_df(d)
        self.assertEqual(type(df), pd.DataFrame)
        self.assertEqual(df.shape, (1, 3))

    def test_empty_numpy_array(self):
        if False:
            print('Hello World!')
        'Test that a single-column empty DataFrame can be constructed\n        from an empty numpy array.\n        '
        arr = np.array([])
        df = type_util.convert_anything_to_df(arr)
        self.assertEqual(type(df), pd.DataFrame)
        self.assertEqual(df.shape, (0, 1))

    def test_styler(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that a DataFrame can be constructed from a pandas.Styler'
        d = {'a': [1], 'b': [2], 'c': [3]}
        styler = pd.DataFrame(d).style.format('{:.2%}')
        df = type_util.convert_anything_to_df(styler)
        self.assertEqual(type(df), pd.DataFrame)
        self.assertEqual(df.shape, (1, 3))

    def test_pyarrow_table(self):
        if False:
            return 10
        'Test that a DataFrame can be constructed from a pyarrow.Table'
        d = {'a': [1], 'b': [2], 'c': [3]}
        table = pa.Table.from_pandas(pd.DataFrame(d))
        df = type_util.convert_anything_to_df(table)
        self.assertEqual(type(df), pd.DataFrame)
        self.assertEqual(df.shape, (1, 3))