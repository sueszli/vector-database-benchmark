import os
import unittest
from test_imperative_qat import TestImperativeQat
import paddle
from paddle.framework import core, set_flags
paddle.enable_static()
os.environ['CPU_NUM'] = '1'
if core.is_compiled_with_cuda():
    set_flags({'FLAGS_cudnn_deterministic': True})

class TestImperativeQatChannelWise(TestImperativeQat):

    def set_vars(self):
        if False:
            print('Hello World!')
        self.weight_quantize_type = 'channel_wise_abs_max'
        self.activation_quantize_type = 'moving_average_abs_max'
        self.diff_threshold = 0.03125
        self.onnx_format = False
        self.fuse_conv_bn = False
        print('weight_quantize_type', self.weight_quantize_type)

class TestImperativeQatChannelWiseONNXFormat(TestImperativeQat):

    def set_vars(self):
        if False:
            while True:
                i = 10
        self.weight_quantize_type = 'channel_wise_abs_max'
        self.activation_quantize_type = 'moving_average_abs_max'
        self.onnx_format = True
        self.diff_threshold = 0.03125
        self.fuse_conv_bn = False
        print('weight_quantize_type', self.weight_quantize_type)
if __name__ == '__main__':
    unittest.main()