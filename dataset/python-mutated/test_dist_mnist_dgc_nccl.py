import os
import subprocess
import unittest
from legacy_test.test_dist_base import TestDistBase
import paddle
paddle.enable_static()
flag_name = os.path.splitext(__file__)[0]

def count_of_sparse_all_reduce_calls(file_name):
    if False:
        for i in range(10):
            print('nop')
    cmd = 'grep -a sparse_all_reduce_op_handle ' + file_name + ' | grep in_numel | wc -l'
    child = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    result = child.communicate()[0]
    print('test_info: result = ' + str(result))
    return int(result)

class TestDistMnistNCCL2DGC(TestDistBase):

    def _setup_config(self):
        if False:
            while True:
                i = 10
        self._sync_mode = True
        self._use_reduce = False
        self._use_reader_alloc = False
        self._nccl2_mode = True
        self._use_dgc = True

    def test_dist_train(self):
        if False:
            print('Hello World!')
        from paddle import base
        if base.core.is_compiled_with_cuda():
            self.check_with_place(os.path.abspath('../../legacy_test/dist_mnist_dgc.py'), delta=1e-05, check_error_log=True, log_name=flag_name)

    def tearDown(self):
        if False:
            print('Hello World!')
        from paddle import base
        if base.core.is_compiled_with_cuda():
            log_file = os.path.join(self.temp_dir.name, 'test_dist_mnist_dgc_nccl_tr0_err.log')
            result = count_of_sparse_all_reduce_calls(log_file)
        self.temp_dir.cleanup()
if __name__ == '__main__':
    unittest.main()