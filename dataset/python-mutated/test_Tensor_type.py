import unittest
import numpy as np
import paddle

class TensorTypeTest(unittest.TestCase):

    def test_type_totensor(self):
        if False:
            return 10
        paddle.disable_static()
        inx = np.array([1, 2])
        tensorx = paddle.to_tensor(inx)
        typex_str = str(type(tensorx))
        expectx = "<class 'paddle.Tensor'>"
        self.assertEqual(typex_str == expectx, True)

    def test_type_Tensor(self):
        if False:
            print('Hello World!')
        paddle.disable_static()
        inx = np.array([1, 2])
        tensorx = paddle.Tensor(inx)
        typex_str = str(type(tensorx))
        expectx = "<class 'paddle.Tensor'>"
        self.assertEqual(typex_str == expectx, True)
        tensorx = paddle.tensor.logic.Tensor(inx)
        typex_str = str(type(tensorx))
        expectx = "<class 'paddle.Tensor'>"
        self.assertEqual(typex_str == expectx, True)
if __name__ == '__main__':
    unittest.main()