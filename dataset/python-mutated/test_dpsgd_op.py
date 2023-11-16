import unittest
import numpy as np
from op_test import OpTest

class TestDpsgdOp(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        'Test Dpsgd Operator with supplied attributes'
        self.op_type = 'dpsgd'
        param = np.random.uniform(-1, 1, (102, 105)).astype('float32')
        grad = np.random.uniform(-1, 1, (102, 105)).astype('float32')
        learning_rate = 0.001
        clip = 10000.0
        batch_size = 16.0
        sigma = 0.0
        self.inputs = {'Param': param, 'Grad': grad, 'LearningRate': np.array([learning_rate]).astype('float32')}
        self.attrs = {'clip': clip, 'batch_size': batch_size, 'sigma': sigma}
        param_out = dpsgd_step(self.inputs, self.attrs)
        self.outputs = {'ParamOut': param_out}

    def test_check_output(self):
        if False:
            return 10
        self.check_output()

def dpsgd_step(inputs, attributes):
    if False:
        i = 10
        return i + 15
    '\n    Simulate one step of the dpsgd optimizer\n    :param inputs: dict of inputs\n    :param attributes: dict of attributes\n    :return tuple: tuple of output param, moment, inf_norm and\n    beta1 power accumulator\n    '
    param = inputs['Param']
    grad = inputs['Grad']
    lr = inputs['LearningRate']
    clip = attributes['clip']
    batch_size = attributes['batch_size']
    sigma = attributes['sigma']
    param_out = param - lr * grad
    return param_out
if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    unittest.main()