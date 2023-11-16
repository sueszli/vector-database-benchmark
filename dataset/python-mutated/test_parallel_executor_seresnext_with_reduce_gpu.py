import unittest
from test_parallel_executor_seresnext_with_reduce_cpu import DeviceType, TestResnetWithReduceBase

class TestResnetWithReduceGPU(TestResnetWithReduceBase):

    def test_seresnext_with_reduce(self):
        if False:
            for i in range(10):
                print('nop')
        self._compare_reduce_and_allreduce(use_device=DeviceType.CUDA, delta2=0.01)
if __name__ == '__main__':
    unittest.main()