import os
import sys
import unittest
sys.path.append('../../legacy_test')
from parallel_dygraph_sparse_embedding import TestSparseEmbedding
from spawn_runner_base import TestDistSpawnRunner
from test_dist_base import TestDistBase
from paddle import base
flag_name = os.path.splitext(__file__)[0]

class TestParallelDygraphSparseEmdedding(TestDistBase):

    def _setup_config(self):
        if False:
            print('Hello World!')
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True

    def test_sparse_embedding(self):
        if False:
            i = 10
            return i + 15
        if base.core.is_compiled_with_cuda():
            self.check_with_place(os.path.abspath('../../legacy_test/parallel_dygraph_sparse_embedding.py'), delta=1e-05, check_error_log=True, log_name=flag_name)

class TestParallelDygraphSparseEmdeddingFP64(TestDistBase):

    def _setup_config(self):
        if False:
            i = 10
            return i + 15
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True

    def test_sparse_embedding_fp64(self):
        if False:
            i = 10
            return i + 15
        if base.core.is_compiled_with_cuda():
            self.check_with_place(os.path.abspath('../../legacy_test/parallel_dygraph_sparse_embedding_fp64.py'), delta=1e-05, check_error_log=True, log_name=flag_name)

class TestParallelDygraphSparseEmdeddingSpawn(TestDistSpawnRunner):

    def test_sparse_embedding_with_spawn(self):
        if False:
            print('Hello World!')
        if base.core.is_compiled_with_cuda():
            self.check_dist_result_with_spawn(test_class=TestSparseEmbedding, delta=1e-05)
if __name__ == '__main__':
    unittest.main()