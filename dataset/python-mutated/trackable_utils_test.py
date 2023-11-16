"""Tests for trackable_utils."""
from tensorflow.python.eager import test
from tensorflow.python.trackable import trackable_utils

class TrackableUtilsTest(test.TestCase):

    def test_order_by_dependency(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests order_by_dependency correctness.'
        dependencies = {1: [2, 3], 2: [3], 3: [5], 4: [3], 5: []}
        sorted_arr = list(trackable_utils.order_by_dependency(dependencies))
        indices = {x: sorted_arr.index(x) for x in range(1, 6)}
        self.assertEqual(indices[5], 0)
        self.assertEqual(indices[3], 1)
        self.assertGreater(indices[1], indices[2])

    def test_order_by_no_dependency(self):
        if False:
            for i in range(10):
                print('nop')
        sorted_arr = list(trackable_utils.order_by_dependency({x: [] for x in range(15)}))
        self.assertEqual(set(sorted_arr), set(range(15)))

    def test_order_by_dependency_invalid_map(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(ValueError, 'Found values in the dependency map which are not keys'):
            trackable_utils.order_by_dependency({1: [2]})
if __name__ == '__main__':
    test.main()