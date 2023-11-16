import unittest
from legacy_test.test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestHybridPipeParallelWithVirtualStage(TestMultipleGpus):

    def test_hybrid_parallel_pp_layer_with_virtual_stage(self):
        if False:
            print('Hello World!')
        pass

    def test_hybrid_parallel_pp_transformer_with_virtual_stage(self):
        if False:
            return 10
        pass

    def test_hybrid_parallel_save_load_with_virtual_stage(self):
        if False:
            while True:
                i = 10
        pass
if __name__ == '__main__':
    unittest.main()