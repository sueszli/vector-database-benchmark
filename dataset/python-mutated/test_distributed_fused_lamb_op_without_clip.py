import unittest
from test_distributed_fused_lamb_op_with_clip import run_test

class TestDistributedFusedLambWithoutClip(unittest.TestCase):

    def test_1(self):
        if False:
            print('Hello World!')
        run_test(clip_after_allreduce=True, max_global_norm=-1.0)

    def test_2(self):
        if False:
            return 10
        run_test(clip_after_allreduce=False, max_global_norm=-1.0)
if __name__ == '__main__':
    unittest.main()