from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import unittest
import pandas
import array
from .. import SFrame
from pandas.util.testing import assert_frame_equal
from sys import version_info
import pytest
pytestmark = [pytest.mark.minimal]

class DataFrameTest(unittest.TestCase):

    def test_empty(self):
        if False:
            return 10
        expected = pandas.DataFrame()
        assert_frame_equal(SFrame(expected).to_dataframe(), expected)
        expected['int'] = []
        expected['float'] = []
        expected['str'] = []
        assert_frame_equal(SFrame(expected).to_dataframe(), expected)

    def test_simple_dataframe(self):
        if False:
            i = 10
            return i + 15
        expected = pandas.DataFrame()
        expected['int'] = [i for i in range(10)]
        expected['float'] = [float(i) for i in range(10)]
        expected['str'] = [str(i) for i in range(10)]
        if version_info.major == 2:
            expected['unicode'] = [unicode(i) for i in range(10)]
        expected['array'] = [array.array('d', [i]) for i in range(10)]
        expected['ls'] = [[str(i)] for i in range(10)]
        assert_frame_equal(SFrame(expected).to_dataframe(), expected)

    def test_sparse_dataframe(self):
        if False:
            while True:
                i = 10
        expected = pandas.DataFrame()
        expected['sparse_int'] = [i if i % 2 == 0 else None for i in range(10)]
        expected['sparse_float'] = [float(i) if i % 2 == 1 else None for i in range(10)]
        expected['sparse_str'] = [str(i) if i % 3 == 0 else None for i in range(10)]
        expected['sparse_array'] = [array.array('d', [i]) if i % 5 == 0 else None for i in range(10)]
        expected['sparse_list'] = [[str(i)] if i % 7 == 0 else None for i in range(10)]
        assert_frame_equal(SFrame(expected).to_dataframe(), expected)