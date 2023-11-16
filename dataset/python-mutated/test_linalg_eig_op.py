import unittest
import numpy as np
import paddle

class TestEigAPIError(unittest.TestCase):

    def test_errors(self):
        if False:
            while True:
                i = 10

        def test_0_size():
            if False:
                for i in range(10):
                    print('nop')
            array = np.array([], dtype=np.float32)
            x = paddle.to_tensor(np.reshape(array, [0, 0]), dtype='float32')
            paddle.linalg.eig(x)
        self.assertRaises(ValueError, test_0_size)
if __name__ == '__main__':
    unittest.main()