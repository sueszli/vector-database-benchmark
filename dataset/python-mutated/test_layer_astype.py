import unittest
import numpy as np
import paddle

class LayerAstypeTest(unittest.TestCase):

    def test_layer_astype(self):
        if False:
            print('Hello World!')
        net = paddle.nn.Sequential(paddle.nn.Linear(2, 2), paddle.nn.Linear(2, 2))
        value = np.array([0]).astype('float32')
        buffer = paddle.to_tensor(value)
        net.register_buffer('test_buffer', buffer, persistable=True)
        valid_dtypes = ['bfloat16', 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'complex64', 'complex128', 'bool']
        for dtype in valid_dtypes:
            net = net.astype(dtype)
            typex_str = str(net._dtype)
            self.assertTrue(typex_str, 'paddle.' + dtype)
            param_typex_str = str(net.parameters()[0].dtype)
            self.assertTrue(param_typex_str, 'paddle.' + dtype)
            buffer_typex_str = str(net.buffers()[0].dtype)
            self.assertTrue(buffer_typex_str, 'paddle.' + dtype)

    def test_error(self):
        if False:
            return 10
        linear1 = paddle.nn.Linear(10, 3)
        try:
            linear1 = linear1.astype('invalid_type')
        except Exception as error:
            self.assertIsInstance(error, ValueError)
if __name__ == '__main__':
    unittest.main()