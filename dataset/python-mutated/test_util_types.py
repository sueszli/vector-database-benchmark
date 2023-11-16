import unittest2
from st2common.util.types import OrderedSet
__all__ = ['OrderedTestTypeTestCase']

class OrderedTestTypeTestCase(unittest2.TestCase):

    def test_ordered_set(self):
        if False:
            i = 10
            return i + 15
        set1 = OrderedSet([1, 2, 3, 3, 4, 2, 1, 5])
        self.assertEqual(set1, [1, 2, 3, 4, 5])
        set2 = OrderedSet([5, 4, 3, 2, 1])
        self.assertEqual(set2, [5, 4, 3, 2, 1])
        set3 = OrderedSet([1, 2, 3, 4, 5, 5, 4, 3, 2, 1])
        self.assertEqual(set3, [1, 2, 3, 4, 5])
        set4 = OrderedSet([1, 1, 1, 1, 4, 4, 4, 9])
        self.assertEqual(set4, [1, 4, 9])