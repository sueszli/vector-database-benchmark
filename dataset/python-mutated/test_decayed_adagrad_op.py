import unittest
import numpy as np
from op_test import OpTest

class TestDecayedAdagradOp1(OpTest):
    """Test DecayedAdagrad operator with explicit attributes"""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'decayed_adagrad'
        param = np.random.random((123, 321)).astype('float32')
        grad = np.random.random((123, 321)).astype('float32')
        moment = np.zeros((123, 321)).astype('float32')
        lr = 0.01
        decay = 0.8
        epsilon = 1e-08
        self.inputs = {'Param': param, 'Grad': grad, 'Moment': moment, 'LearningRate': np.array([lr]).astype('float32')}
        self.attrs = {'decay': decay, 'epsilon': epsilon}
        moment_out = decay * moment + (1 - decay) * grad * grad
        param_out = param - lr * grad / (np.sqrt(moment_out) + epsilon)
        self.outputs = {'ParamOut': param_out, 'MomentOut': moment_out}

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output(check_dygraph=False)

class TestDecayedAdagradOp2(OpTest):
    """Test DecayedAdagrad operator with default attributes"""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'decayed_adagrad'
        param = np.random.random((123, 321)).astype('float32')
        grad = np.random.random((123, 321)).astype('float32')
        moment = np.zeros((123, 321)).astype('float32')
        lr = 0.01
        decay = 0.95
        epsilon = 1e-06
        self.inputs = {'Param': param, 'Grad': grad, 'Moment': moment, 'LearningRate': np.array([lr]).astype('float32')}
        self.attrs = {'decay': decay, 'epsilon': epsilon}
        moment_out = decay * moment + (1 - decay) * grad * grad
        param_out = param - lr * grad / (np.sqrt(moment_out) + epsilon)
        self.outputs = {'ParamOut': param_out, 'MomentOut': moment_out}

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output(check_dygraph=False)
if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    unittest.main()