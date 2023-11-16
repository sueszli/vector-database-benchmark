"""Tests for logical module."""
import unittest
from nvidia.dali._autograph.operators import logical

class LogicalOperatorsTest(unittest.TestCase):

    def assertNotCalled(self):
        if False:
            while True:
                i = 10
        self.fail('this should not be called')

    def test_and_python(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(logical.and_(lambda : True, lambda : True))
        self.assertTrue(logical.and_(lambda : [1], lambda : True))
        self.assertListEqual(logical.and_(lambda : True, lambda : [1]), [1])
        self.assertFalse(logical.and_(lambda : False, lambda : True))
        self.assertFalse(logical.and_(lambda : False, self.assertNotCalled))

    def test_or_python(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFalse(logical.or_(lambda : False, lambda : False))
        self.assertFalse(logical.or_(lambda : [], lambda : False))
        self.assertListEqual(logical.or_(lambda : False, lambda : [1]), [1])
        self.assertTrue(logical.or_(lambda : False, lambda : True))
        self.assertTrue(logical.or_(lambda : True, self.assertNotCalled))

    def test_not_python(self):
        if False:
            print('Hello World!')
        self.assertFalse(logical.not_(True))
        self.assertFalse(logical.not_([1]))
        self.assertTrue(logical.not_([]))