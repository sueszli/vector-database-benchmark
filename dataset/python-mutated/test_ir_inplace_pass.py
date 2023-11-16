import os
import unittest
import numpy as np
from parallel_executor_test_base import DeviceType, TestParallelExecutorBase
import paddle
from paddle import base
from paddle.base import core

def fc_with_batchnorm(use_feed):
    if False:
        for i in range(10):
            print('nop')
    img = paddle.static.data(name='image', shape=[-1, 784], dtype='float32')
    label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
    hidden = img
    for _ in range(3):
        hidden = paddle.static.nn.fc(hidden, size=200, activation='tanh', bias_attr=base.ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0)))
        hidden = paddle.static.nn.batch_norm(input=hidden)
    prediction = paddle.static.nn.fc(hidden, size=10, activation='softmax')
    loss = paddle.nn.functional.cross_entropy(input=prediction, label=label, reduction='none', use_softmax=False)
    loss = paddle.mean(loss)
    return loss

class TestIrInplace(TestParallelExecutorBase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        os.environ['CPU_NUM'] = str(4)

    def _fc_with_batchnorm(self, ir_memory_optimize, enable_inplace):
        if False:
            for i in range(10):
                print('nop')
        if not core.is_compiled_with_cuda():
            return
        np.random.seed(5)
        img = np.random.random(size=[32, 784]).astype(np.float32)
        label = np.ones(shape=[32, 1], dtype='int64')
        self.check_network_convergence(fc_with_batchnorm, feed_dict={'image': img, 'label': label}, use_device=DeviceType.CUDA, use_ir_memory_optimize=ir_memory_optimize, enable_inplace=enable_inplace)

    def test_fc_with_batchnorm(self, delta=0.001):
        if False:
            while True:
                i = 10
        loss00 = self._fc_with_batchnorm(False, False)
        loss10 = self._fc_with_batchnorm(True, False)
        loss01 = self._fc_with_batchnorm(False, True)
        loss11 = self._fc_with_batchnorm(True, True)
        self.assertAlmostEqual(loss00, loss10, delta=delta)
        self.assertAlmostEqual(loss00, loss01, delta=delta)
        self.assertAlmostEqual(loss00, loss11, delta=delta)
if __name__ == '__main__':
    unittest.main()