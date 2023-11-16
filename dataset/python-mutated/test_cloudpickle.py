from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as tc
import unittest
from pickle import PicklingError
import pickle
import turicreate.util._cloudpickle as cloudpickle
from sys import version_info
import pytest
pytestmark = [pytest.mark.minimal]

class CloudPickleTest(unittest.TestCase):

    def test_pickle_unity_object_exception(self):
        if False:
            i = 10
            return i + 15
        sa = tc.SArray()
        sf = tc.SFrame()
        g = tc.SGraph()
        sk = sa.summary()
        m = tc.pagerank.create(g)
        expected_error = TypeError if version_info[0] == 3 else PicklingError
        for obj in [sa, sf, g, sk, m]:
            self.assertRaises(expected_error, lambda : cloudpickle.dumps(obj))

    def test_memoize_subclass(self):
        if False:
            return 10

        class A(object):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.name = 'A'

        class B(A):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super(B, self).__init__()
                self.name2 = 'B'
        b = B()
        self.assertEqual(b.name, 'A')
        self.assertEqual(b.name2, 'B')
        b2 = pickle.loads(cloudpickle.dumps(b))
        self.assertEqual(b.name, b2.name)
        self.assertEqual(b.name2, b2.name2)