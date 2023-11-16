import logging
import unittest
import numpy as np
from apache_beam.testing.extra_assertions import ExtraAssertionsMixin

class ExtraAssertionsMixinTest(ExtraAssertionsMixin, unittest.TestCase):

    def test_assert_array_count_equal_strings(self):
        if False:
            return 10
        data1 = ['±♠Ωℑ', 'hello', 'world']
        data2 = ['hello', '±♠Ωℑ', 'world']
        self.assertUnhashableCountEqual(data1, data2)

    def test_assert_array_count_equal_mixed(self):
        if False:
            return 10
        data1 = [{'a': 1, 123: 1.234}, ['d', 1], '±♠Ωℑ', np.zeros((3, 6)), (1, 2, 3, 'b'), 'def', 100, 'abc', ('a', 'b', 'c'), None]
        data2 = [{123: 1.234, 'a': 1}, ('a', 'b', 'c'), ['d', 1], None, 'abc', 'def', '±♠Ωℑ', 100, (1, 2, 3, 'b'), np.zeros((3, 6))]
        self.assertUnhashableCountEqual(data1, data2)
        self.assertUnhashableCountEqual(data1 * 2, data2 * 2)

    def test_assert_not_equal(self):
        if False:
            while True:
                i = 10
        data1 = [{'a': 123, 'b': 321}, [1, 2, 3]]
        data2 = [{'a': 123, 'c': 321}, [1, 2, 3]]
        with self.assertRaises(AssertionError):
            self.assertUnhashableCountEqual(data1, data2)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()