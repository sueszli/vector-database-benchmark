import os
import unittest
from legacy_test.test_parallel_dygraph_dataparallel import TestMultipleGpus
import paddle

class TestHybridPipeParallel(TestMultipleGpus):

    def test_hybrid_parallel_pp_layer(self):
        if False:
            while True:
                i = 10
        self.run_mnist_2gpu(os.path.abspath('../../legacy_test/hybrid_parallel_pp_layer.py'))

    def test_hybrid_parallel_pp_tuple_inputs(self):
        if False:
            while True:
                i = 10
        self.run_mnist_2gpu('hybrid_parallel_pp_embedding.py')

    def test_hybrid_parallel_shared_weight(self):
        if False:
            print('Hello World!')
        self.run_mnist_2gpu('hybrid_parallel_shared_weight.py')

    def test_pipeline_parallel_amp(self):
        if False:
            i = 10
            return i + 15
        self.run_mnist_2gpu('hybrid_parallel_pp_amp.py')

    def test_pipeline_parallel_fp16(self):
        if False:
            return 10
        self.run_mnist_2gpu('hybrid_parallel_pp_fp16.py')

    def test_pipeline_parallel_bf16(self):
        if False:
            while True:
                i = 10
        self.run_mnist_2gpu('hybrid_parallel_pp_bf16.py')

    def test_hybrid_parallel_transformer(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_mnist_2gpu('hybrid_parallel_pp_transformer.py')

    def test_hybrid_parallel_save_load(self):
        if False:
            i = 10
            return i + 15
        self.run_mnist_2gpu('hybrid_parallel_pp_save_load.py')

    def test_hybrid_parallel_recompute(self):
        if False:
            print('Hello World!')
        self.run_mnist_2gpu('hybrid_parallel_pp_recompute.py')

    def test_hybrid_parallel_pp_clip_grad(self):
        if False:
            while True:
                i = 10
        self.run_mnist_2gpu('hybrid_parallel_pp_clip_grad.py')

    def test_hybrid_parallel_transformer_unbalanced_data(self):
        if False:
            while True:
                i = 10
        self.run_mnist_2gpu('hybrid_parallel_pp_transformer_unbalanced_data.py')

class TestFakeMicroDataSet(unittest.TestCase):

    def test_fake_micro_data_set(self):
        if False:
            return 10
        import numpy as np
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import FakeMicroDataset
        batch_size = 4
        micro_batch_size = 2
        acc_step = 2
        length = 4
        x_data = np.random.randint(0, batch_size, size=[batch_size, length])
        data1 = paddle.to_tensor(x_data)
        data1.stop_gradient = True
        data2 = [data1[i * micro_batch_size:(i + 1) * micro_batch_size, :].detach() for i in range(acc_step)]
        data3 = None
        batch = [(data1, data2, data3), None]
        for micro_batch in FakeMicroDataset(batch, True, False, acc_step, micro_batch_size):
            (x, y) = micro_batch
            self.assertEqual(len(x), 3)
            for e in [x[0], x[1]]:
                self.assertEqual(e.shape[0], micro_batch_size)
                self.assertEqual(e.shape[1], length)
            self.assertTrue(x[2] is None)
            self.assertTrue(y is None)
        micro_batches = FakeMicroDataset(batch, False, False, acc_step, micro_batch_size)
        (x, y) = micro_batches._load_micro_batch(0)
        self.assertTrue(x is None)
        self.assertTrue(y is None)
if __name__ == '__main__':
    unittest.main()