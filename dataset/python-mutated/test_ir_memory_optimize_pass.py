import unittest
import numpy as np
from parallel_executor_test_base import DeviceType, TestParallelExecutorBase
import paddle
from paddle.base import core

def _feed_data_helper():
    if False:
        return 10
    img = paddle.static.data(name='image', shape=[-1, 784], dtype='float32')
    label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
    return (img, label)

def simple_fc_net(use_feed):
    if False:
        print('Hello World!')
    assert use_feed
    (x, y) = _feed_data_helper()
    hidden_layer = 4
    for _ in range(hidden_layer):
        x = paddle.static.nn.fc(x, size=20, activation='relu')
    y_predict = paddle.static.nn.fc(x, size=10, activation='softmax')
    cost = paddle.nn.functional.cross_entropy(input=y_predict, label=y, reduction='none', use_softmax=False)
    avg_cost = paddle.mean(cost)
    return avg_cost

def fc_with_inplace_net(use_feed):
    if False:
        i = 10
        return i + 15
    assert use_feed
    (x, y) = _feed_data_helper()
    fc = paddle.static.nn.fc(x=x, size=20, activation='relu')
    fc = paddle.static.nn.fc(x=fc, size=10, activation='relu')
    reshape = paddle.reshape(x=fc, shape=[-1, 2, 5])
    reshape = paddle.reshape(x=reshape, shape=[-1, 5, 2])
    y_predict = paddle.static.nn.fc(x=reshape, size=10, activation='softmax')
    cost = paddle.nn.functional.cross_entropy(input=y_predict, label=y, reduction='none', use_softmax=False)
    avg_cost = paddle.mean(cost)
    return avg_cost

class TestMNIST(TestParallelExecutorBase):

    def _dummy_data(self):
        if False:
            print('Hello World!')
        np.random.seed(5)
        img = np.random.random(size=[32, 784]).astype(np.float32)
        label = np.ones(shape=[32, 1], dtype='int64')
        return (img, label)

    def _compare_ir_memory_optimize(self, model, use_device):
        if False:
            i = 10
            return i + 15
        if use_device == DeviceType.CUDA and (not core.is_compiled_with_cuda()):
            return
        (img, label) = self._dummy_data()
        (first_loss0, last_loss0, _) = self.check_network_convergence(model, feed_dict={'image': img, 'label': label}, use_device=use_device, use_ir_memory_optimize=False)
        (first_loss1, last_loss1, _) = self.check_network_convergence(model, feed_dict={'image': img, 'label': label}, use_device=use_device, use_ir_memory_optimize=True)
        self.assertAlmostEqual(first_loss0, first_loss1, delta=1e-06)
        self.assertAlmostEqual(last_loss0, last_loss1, delta=1e-06)

    def test_simple_fc_net(self):
        if False:
            i = 10
            return i + 15
        self._compare_ir_memory_optimize(simple_fc_net, DeviceType.CPU)
        self._compare_ir_memory_optimize(simple_fc_net, DeviceType.CUDA)

    def test_fc_with_reshape_net(self):
        if False:
            i = 10
            return i + 15
        self._compare_ir_memory_optimize(fc_with_inplace_net, DeviceType.CPU)
        self._compare_ir_memory_optimize(fc_with_inplace_net, DeviceType.CUDA)
if __name__ == '__main__':
    unittest.main()