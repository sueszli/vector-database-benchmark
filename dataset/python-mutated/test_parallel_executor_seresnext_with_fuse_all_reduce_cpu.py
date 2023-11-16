from paddle import base
base.core._set_fuse_parameter_group_size(3)
base.core._set_fuse_parameter_memory_size(131072)
import unittest
from functools import partial
import seresnext_net
from seresnext_test_base import DeviceType, TestResnetBase

class TestResnetWithFuseAllReduceCPU(TestResnetBase):

    def test_seresnext_with_fused_all_reduce(self):
        if False:
            return 10
        check_func = partial(self.check_network_convergence, optimizer=seresnext_net.optimizer, fuse_all_reduce_ops=True)
        self._compare_result_with_origin_model(check_func, use_device=DeviceType.CPU)
if __name__ == '__main__':
    unittest.main()