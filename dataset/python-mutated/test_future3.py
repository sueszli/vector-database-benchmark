from __future__ import nested_scopes
from __future__ import division
import unittest
x = 2

def nester():
    if False:
        print('Hello World!')
    x = 3

    def inner():
        if False:
            while True:
                i = 10
        return x
    return inner()

class TestFuture(unittest.TestCase):

    def test_floor_div_operator(self):
        if False:
            while True:
                i = 10
        self.assertEqual(7 // 2, 3)

    def test_true_div_as_default(self):
        if False:
            print('Hello World!')
        self.assertAlmostEqual(7 / 2, 3.5)

    def test_nested_scopes(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(nester(), 3)
if __name__ == '__main__':
    unittest.main()