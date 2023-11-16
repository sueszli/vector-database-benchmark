import numpy as np
import seresnext_net
from parallel_executor_test_base import DeviceType, TestParallelExecutorBase
from paddle.base import core

class TestResnetBase(TestParallelExecutorBase):

    def _compare_result_with_origin_model(self, check_func, use_device, delta2=1e-05, compare_separately=True):
        if False:
            return 10
        if use_device == DeviceType.CUDA and (not core.is_compiled_with_cuda()):
            return
        (func_1_first_loss, func_1_last_loss, func_1_loss_area) = self.check_network_convergence(seresnext_net.model, feed_dict=seresnext_net.feed_dict(use_device), iter=seresnext_net.iter(use_device), batch_size=seresnext_net.batch_size(use_device), use_device=use_device, use_reduce=False, optimizer=seresnext_net.optimizer)
        (func_2_first_loss, func_2_last_loss, func_2_loss_area) = check_func(seresnext_net.model, feed_dict=seresnext_net.feed_dict(use_device), iter=seresnext_net.iter(use_device), batch_size=seresnext_net.batch_size(use_device), use_device=use_device)
        if compare_separately:
            self.assertAlmostEqual(func_1_first_loss, func_2_first_loss, delta=1e-05)
            self.assertAlmostEqual(func_1_last_loss, func_2_last_loss, delta=delta2)
        else:
            np.testing.assert_allclose(func_1_loss_area, func_2_loss_area, rtol=delta2)
            self.assertAlmostEqual(func_1_first_loss, func_2_first_loss, delta=1e-05)
            self.assertAlmostEqual(func_1_last_loss, func_2_last_loss, delta=delta2)