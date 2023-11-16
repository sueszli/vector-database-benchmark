import unittest
from legacy_test.test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestHybridParallel(TestMultipleGpus):

    def test_hybrid_parallel_mp_random(self):
        if False:
            print('Hello World!')
        self.run_mnist_2gpu('hybrid_parallel_mp_random.py')

    def test_hybrid_parallel_mp_model(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_mnist_2gpu('hybrid_parallel_mp_model.py')

    def test_hybrid_parallel_mp_model_with_sequence_parallel(self):
        if False:
            i = 10
            return i + 15
        self.run_mnist_2gpu('hybrid_parallel_mp_model_with_sequence_parallel.py')

    def test_hybrid_parallel_mp_amp(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_mnist_2gpu('hybrid_parallel_mp_amp.py')

    def test_hybrid_parallel_mp_fp16(self):
        if False:
            print('Hello World!')
        self.run_mnist_2gpu('hybrid_parallel_mp_fp16.py')

    def test_hybrid_parallel_mp_bf16(self):
        if False:
            return 10
        self.run_mnist_2gpu('hybrid_parallel_mp_bf16.py')

    def test_hybrid_parallel_mp_clip_grad(self):
        if False:
            print('Hello World!')
        self.run_mnist_2gpu('hybrid_parallel_mp_clip_grad.py')

    def test_hybrid_parallel_mp_broadcast_obj(self):
        if False:
            i = 10
            return i + 15
        self.run_mnist_2gpu('hybrid_parallel_mp_broadcast_obj.py')
if __name__ == '__main__':
    unittest.main()