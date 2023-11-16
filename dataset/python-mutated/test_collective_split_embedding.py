import unittest
from legacy_test.test_collective_api_base import TestDistBase
import paddle
paddle.enable_static()

class TestParallelEmbeddingAPI(TestDistBase):

    def _setup_config(self):
        if False:
            print('Hello World!')
        pass

    def test_parallel_embedding(self):
        if False:
            return 10
        self.check_with_place('parallel_embedding_api.py', 'parallel_embedding', 'nccl')
if __name__ == '__main__':
    unittest.main()