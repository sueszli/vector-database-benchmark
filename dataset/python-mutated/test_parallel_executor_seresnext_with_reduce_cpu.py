import unittest
import seresnext_net
from parallel_executor_test_base import DeviceType, TestParallelExecutorBase
from paddle.base import core

class TestResnetWithReduceBase(TestParallelExecutorBase):

    def _compare_reduce_and_allreduce(self, use_device, delta2=1e-05):
        if False:
            i = 10
            return i + 15
        if use_device == DeviceType.CUDA and (not core.is_compiled_with_cuda()):
            return
        (all_reduce_first_loss, all_reduce_last_loss, _) = self.check_network_convergence(seresnext_net.model, feed_dict=seresnext_net.feed_dict(use_device), iter=seresnext_net.iter(use_device), batch_size=seresnext_net.batch_size(use_device), use_device=use_device, use_reduce=False, optimizer=seresnext_net.optimizer)
        (reduce_first_loss, reduce_last_loss, _) = self.check_network_convergence(seresnext_net.model, feed_dict=seresnext_net.feed_dict(use_device), iter=seresnext_net.iter(use_device), batch_size=seresnext_net.batch_size(use_device), use_device=use_device, use_reduce=True, optimizer=seresnext_net.optimizer)
        self.assertAlmostEqual(all_reduce_first_loss, reduce_first_loss, delta=1e-05)
        self.assertAlmostEqual(all_reduce_last_loss, reduce_last_loss, delta=all_reduce_last_loss * delta2)
        if not use_device:
            return
        (all_reduce_first_loss_seq, all_reduce_last_loss_seq, _) = self.check_network_convergence(seresnext_net.model, feed_dict=seresnext_net.feed_dict(use_device), iter=seresnext_net.iter(use_device), batch_size=seresnext_net.batch_size(use_device), use_device=use_device, use_reduce=False, optimizer=seresnext_net.optimizer, enable_sequential_execution=True)
        (reduce_first_loss_seq, reduce_last_loss_seq, _) = self.check_network_convergence(seresnext_net.model, feed_dict=seresnext_net.feed_dict(use_device), iter=seresnext_net.iter(use_device), batch_size=seresnext_net.batch_size(use_device), use_device=use_device, use_reduce=True, optimizer=seresnext_net.optimizer, enable_sequential_execution=True)
        self.assertAlmostEqual(all_reduce_first_loss, all_reduce_first_loss_seq, delta=1e-05)
        self.assertAlmostEqual(all_reduce_last_loss, all_reduce_last_loss_seq, delta=all_reduce_last_loss * delta2)
        self.assertAlmostEqual(reduce_first_loss, reduce_first_loss_seq, delta=1e-05)
        self.assertAlmostEqual(reduce_last_loss, reduce_last_loss_seq, delta=reduce_last_loss * delta2)
        self.assertAlmostEqual(all_reduce_first_loss_seq, reduce_first_loss_seq, delta=1e-05)
        self.assertAlmostEqual(all_reduce_last_loss_seq, reduce_last_loss_seq, delta=all_reduce_last_loss_seq * delta2)

class TestResnetWithReduceCPU(TestResnetWithReduceBase):

    def test_seresnext_with_reduce(self):
        if False:
            return 10
        self._compare_reduce_and_allreduce(use_device=DeviceType.CPU, delta2=0.001)
if __name__ == '__main__':
    unittest.main()