import os
import unittest
from functools import partial
from fake_reader import fake_imdb_reader
from parallel_executor_test_base import DeviceType, TestParallelExecutorBase
from simple_nets import bow_net, fc_with_batchnorm, init_data
import paddle
from paddle import base
from paddle.base import core

class TestFuseOptimizationOps(TestParallelExecutorBase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        os.environ['CPU_NUM'] = str(4)

    def _get_feed_dict(self):
        if False:
            while True:
                i = 10
        (img, label) = init_data()
        return {'image': img, 'label': label}

    def _compare_fused_optimizer_ops(self, model, use_device, feed_dict=None, get_data_from_feeder=None, optimizer=paddle.optimizer.Adam):
        if False:
            print('Hello World!')
        if use_device == DeviceType.CUDA and (not core.is_compiled_with_cuda()):
            return
        (not_fuse_op_first_loss, not_fuse_op_last_loss, _) = self.check_network_convergence(model, feed_dict=feed_dict, get_data_from_feeder=get_data_from_feeder, use_device=use_device, fuse_all_optimizer_ops=False, optimizer=optimizer)
        (fuse_op_first_loss, fuse_op_last_loss, _) = self.check_network_convergence(model, feed_dict=feed_dict, get_data_from_feeder=get_data_from_feeder, use_device=use_device, fuse_all_optimizer_ops=True, optimizer=optimizer)
        self.assertAlmostEqual(not_fuse_op_first_loss, fuse_op_first_loss, delta=1e-06)
        self.assertAlmostEqual(not_fuse_op_last_loss, fuse_op_last_loss, delta=1e-06)

    def _decorate_compare_fused_optimizer_ops(self, model, use_device, optimizer):
        if False:
            for i in range(10):
                print('nop')
        self._compare_fused_optimizer_ops(model, use_device, feed_dict=self._get_feed_dict(), optimizer=optimizer)

class TestFuseAdamOps(TestFuseOptimizationOps):

    def optimizer(self, learning_rate=0.0001):
        if False:
            for i in range(10):
                print('nop')
        return paddle.optimizer.Adam(learning_rate=learning_rate)

    def test_batchnorm_fc_with_fuse_op(self):
        if False:
            i = 10
            return i + 15
        self._decorate_compare_fused_optimizer_ops(fc_with_batchnorm, DeviceType.CUDA, optimizer=self.optimizer)
        self._decorate_compare_fused_optimizer_ops(fc_with_batchnorm, DeviceType.CPU, optimizer=self.optimizer)

class TestFuseSGDOps(TestFuseAdamOps):

    def optimizer(self, learning_rate=0.001):
        if False:
            while True:
                i = 10
        return paddle.optimizer.SGD(learning_rate=learning_rate)

class TestFuseMomentumOps(TestFuseAdamOps):

    def optimizer(self, learning_rate=0.001):
        if False:
            print('Hello World!')
        return paddle.optimizer.Momentum(learning_rate=learning_rate, momentum=0.1)

class TestSpareFuseAdamOps(TestFuseOptimizationOps):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        os.environ['CPU_NUM'] = str(4)
        cls.word_dict_len = 5147
        batch_size = 64
        reader = fake_imdb_reader(cls.word_dict_len, batch_size * 100)
        reader = paddle.batch(reader, batch_size=batch_size)()
        cls.train_data = next(reader)

    def _get_data_from_feeder(self):
        if False:
            i = 10
            return i + 15
        place = base.CPUPlace()
        feeder = base.DataFeeder(feed_list=['words', 'label'], place=place)
        return feeder.feed(self.train_data)

    def _decorate_compare_fused_optimizer_ops(self, model, use_device, optimizer):
        if False:
            for i in range(10):
                print('nop')
        self._compare_fused_optimizer_ops(model, use_device, get_data_from_feeder=self._get_data_from_feeder, optimizer=optimizer)

    def optimizer(self, learning_rate=0.0001):
        if False:
            return 10
        return paddle.optimizer.Adam(learning_rate=learning_rate)

    def test_simple_bow_net_with_fuse_op(self):
        if False:
            for i in range(10):
                print('nop')
        model = partial(bow_net, dict_dim=self.word_dict_len, is_sparse=True)
        self._decorate_compare_fused_optimizer_ops(model, DeviceType.CUDA, optimizer=self.optimizer)
        self._decorate_compare_fused_optimizer_ops(model, DeviceType.CPU, optimizer=self.optimizer)

class TestSpareFuseSGDOps(TestSpareFuseAdamOps):

    def optimizer(self, learning_rate=0.001):
        if False:
            i = 10
            return i + 15
        return paddle.optimizer.SGD(learning_rate=learning_rate)

class TestSpareFuseMomentumOps(TestSpareFuseAdamOps):

    def optimizer(self, learning_rate=0.001):
        if False:
            return 10
        return paddle.optimizer.Momentum(learning_rate=learning_rate, momentum=0.1)

class TestPassConflictBase(TestFuseAdamOps):

    def _compare_fused_optimizer_ops(self, model, use_device, feed_dict=None, get_data_from_feeder=None, optimizer=paddle.optimizer.Adam):
        if False:
            i = 10
            return i + 15
        if use_device == DeviceType.CUDA and (not core.is_compiled_with_cuda()):
            return
        self.check_pass_conflict(model, feed_dict=feed_dict, get_data_from_feeder=get_data_from_feeder, use_device=use_device, fuse_all_optimizer_ops=True, optimizer=optimizer, enable_sequential_execution=True)

class TestFuseAdamOpsPassConflict(TestPassConflictBase):

    def optimizer(self, learning_rate=0.0001):
        if False:
            i = 10
            return i + 15
        return paddle.optimizer.Adam(learning_rate=learning_rate)

    def test_batchnorm_fc_with_fuse_op(self):
        if False:
            for i in range(10):
                print('nop')
        self._decorate_compare_fused_optimizer_ops(fc_with_batchnorm, DeviceType.CPU, optimizer=self.optimizer)
        self._decorate_compare_fused_optimizer_ops(fc_with_batchnorm, DeviceType.CUDA, optimizer=self.optimizer)

class TestFuseSGDOpsPassConflict(TestFuseAdamOpsPassConflict):

    def optimizer(self, learning_rate=0.001):
        if False:
            while True:
                i = 10
        return paddle.optimizer.SGD(learning_rate=learning_rate)

class TestFuseMomentumOpsPassConflict(TestFuseAdamOpsPassConflict):

    def optimizer(self, learning_rate=0.001):
        if False:
            print('Hello World!')
        return paddle.optimizer.Momentum(learning_rate=learning_rate, momentum=0.1)
if __name__ == '__main__':
    unittest.main()