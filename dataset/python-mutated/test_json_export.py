from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import json
import numpy as np
import struct
import tempfile
import unittest
import turicreate as tc
_TEST_CASE_SIZE = 1000
import pytest
pytestmark = [pytest.mark.minimal]

class JSONExporterTest(unittest.TestCase):

    def test_simple_types(self):
        if False:
            return 10
        np.random.seed(42)
        sf = tc.SFrame()
        sf['idx'] = range(_TEST_CASE_SIZE)
        sf['ints'] = np.random.randint(-100000, 100000, _TEST_CASE_SIZE)
        sf['strings'] = sf['ints'].astype(str)
        sf['floats'] = np.random.random(_TEST_CASE_SIZE)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as json_file:
            sf.save(json_file.name, format='json')
            with open(json_file.name) as json_data:
                loaded = json.load(json_data)

    def test_array_dtype(self):
        if False:
            while True:
                i = 10
        np.random.seed(42)
        sf = tc.SFrame()
        sf['arr'] = np.random.rand(100, 3)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as json_file:
            sf.save(json_file.name, format='json')
            with open(json_file.name) as json_data:
                loaded = json.load(json_data)