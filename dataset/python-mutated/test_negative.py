import unittest
import numpy as np
import prml.nn as nn

class TestNegative(unittest.TestCase):

    def test_negative(self):
        if False:
            i = 10
            return i + 15
        npx = np.random.randn(8, 9)
        x = nn.asarray(npx)
        y = -x
        self.assertTrue(np.allclose(y.value, -npx))
        npg = np.random.randn(8, 9)
        y.backward(npg)
        self.assertTrue(np.allclose(x.grad, -npg))
if __name__ == '__main__':
    unittest.main()