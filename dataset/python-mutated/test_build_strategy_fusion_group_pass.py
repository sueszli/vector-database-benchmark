import unittest
from test_eager_deletion_padding_rnn import PaddingRNNTestBase, RNNConfig
import paddle
from paddle import base
from paddle.base import core

class FusionGroupPaddingRNNTest(PaddingRNNTestBase):

    def set_customed_config(self):
        if False:
            i = 10
            return i + 15
        self.build_strategy.enable_auto_fusion = True
        if core.is_compiled_with_cuda():
            self.exe = base.Executor(base.CUDAPlace(0))

    def test_train_enable_fusion_group(self):
        if False:
            return 10
        rnn_model = 'static'
        config = RNNConfig('test', rnn_model)
        with base.scope_guard(base.Scope()):
            self.train(config, use_program_cache=False)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()