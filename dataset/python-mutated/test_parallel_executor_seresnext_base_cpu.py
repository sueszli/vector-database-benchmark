import unittest
from functools import partial
import seresnext_net
from seresnext_test_base import DeviceType, TestResnetBase

class TestResnetCPU(TestResnetBase):

    def test_seresnext_with_learning_rate_decay(self):
        if False:
            print('Hello World!')
        check_func = partial(self.check_network_convergence, optimizer=seresnext_net.optimizer, use_parallel_executor=False)
        self._compare_result_with_origin_model(check_func, use_device=DeviceType.CPU, compare_separately=False, delta2=0.001)
if __name__ == '__main__':
    unittest.main()