import unittest
import numpy as np
from op_test import OpTest

def dequantize_log(x, dict_data):
    if False:
        return 10
    output_data = np.zeros_like(x).astype('float32')
    x_f = x.flatten()
    output_data_f = output_data.flatten()
    for i in range(x_f.size):
        if x_f[i] < 0:
            output_data_f[i] = -dict_data[x_f[i] + 128]
        else:
            output_data_f[i] = dict_data[x_f[i]]
    return output_data_f.reshape(x.shape)

class TestDequantizeLogOp(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'dequantize_log'
        x = np.random.randint(low=-128, high=127, size=(20, 10)).astype('int8')
        dict_data = np.random.random(128).astype('float32')
        xdq = dequantize_log(x, dict_data)
        self.inputs = {'X': np.array(x).astype('int8'), 'Dict': np.array(dict_data).astype('float32')}
        self.outputs = {'Out': xdq}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output()
if __name__ == '__main__':
    unittest.main()