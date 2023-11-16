import unittest
from test_distributed_fused_lamb_op_with_clip import run_test

class TestDistributedFusedLambGradientMerge(unittest.TestCase):

    def test_gm(self):
        if False:
            for i in range(10):
                print('nop')
        run_test(clip_after_allreduce=True, max_global_norm=-1.0, gradient_merge_steps=2)

    def test_gm_with_fp16_acc_grad(self):
        if False:
            for i in range(10):
                print('nop')
        run_test(clip_after_allreduce=True, max_global_norm=-1.0, gradient_merge_steps=2, use_master_acc_grad=False)

    def test_gm_new_comm(self):
        if False:
            i = 10
            return i + 15
        run_test(clip_after_allreduce=True, max_global_norm=-1.0, gradient_merge_steps=2, need_env={'FLAGS_dynamic_static_unified_comm': 'true'})

    def test_gm_with_fp16_acc_grad_new_comm(self):
        if False:
            for i in range(10):
                print('nop')
        run_test(clip_after_allreduce=True, max_global_norm=-1.0, gradient_merge_steps=2, use_master_acc_grad=False, need_env={'FLAGS_dynamic_static_unified_comm': 'true'})
if __name__ == '__main__':
    unittest.main()