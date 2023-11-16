import os
import unittest
from test_dist_base import TestDistBase
flag_name = os.path.splitext(__file__)[0]

class TestParallelDygraphSparseEmdedding_GLOO(TestDistBase):

    def _setup_config(self):
        if False:
            while True:
                i = 10
        self._sync_mode = False
        self._gloo_mode = True
        self._dygraph = True

    def test_sparse_embedding(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_with_place('parallel_dygraph_sparse_embedding.py', delta=1e-05, check_error_log=True, log_name=flag_name)

class TestParallelDygraphSparseEmdeddingFP64_GLOO(TestDistBase):

    def _setup_config(self):
        if False:
            for i in range(10):
                print('nop')
        self._sync_mode = False
        self._gloo_mode = True
        self._dygraph = True

    def test_sparse_embedding_fp64(self):
        if False:
            while True:
                i = 10
        self.check_with_place('parallel_dygraph_sparse_embedding_fp64.py', delta=1e-05, check_error_log=True, log_name=flag_name)
if __name__ == '__main__':
    unittest.main()