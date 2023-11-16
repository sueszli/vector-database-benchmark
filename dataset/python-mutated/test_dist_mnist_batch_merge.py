import os
import unittest
from test_dist_base import TestDistBase
import paddle
paddle.enable_static()
flag_name = os.path.splitext(__file__)[0]

class TestDistMnist2x2(TestDistBase):

    def _setup_config(self):
        if False:
            for i in range(10):
                print('nop')
        self._sync_mode = True
        self._use_reduce = False

    def test_dist_train(self):
        if False:
            return 10
        self.check_with_place('dist_mnist_batch_merge.py', delta=1e-05)

    def check_with_place(self, model_file, delta=0.001, check_error_log=False, need_envs={}):
        if False:
            i = 10
            return i + 15
        required_envs = {'PATH': os.getenv('PATH', ''), 'PYTHONPATH': os.getenv('PYTHONPATH', ''), 'LD_LIBRARY_PATH': os.getenv('LD_LIBRARY_PATH', ''), 'FLAGS_fraction_of_gpu_memory_to_use': '0.15', 'FLAGS_cudnn_deterministic': '1'}
        required_envs.update(need_envs)
        if check_error_log:
            required_envs['GLOG_vmodule'] = 'fused_all_reduce_op_handle=10,all_reduce_op_handle=10,alloc_continuous_space_op=10,fuse_all_reduce_op_pass=10,alloc_continuous_space_for_grad_pass=10,fast_threaded_ssa_graph_executor=10'
            required_envs['GLOG_logtostderr'] = '1'
        no_merge_losses = self._run_local(model_file, required_envs, check_error_log=check_error_log, batch_size=4, log_name=flag_name)
        batch_merge_losses = self._run_local(model_file, required_envs, check_error_log=check_error_log, batch_size=2, batch_merge_repeat=2, log_name=flag_name)
        self.assertGreater(len(no_merge_losses), 1)
        self.assertEqual(len(no_merge_losses), len(batch_merge_losses))
if __name__ == '__main__':
    unittest.main()