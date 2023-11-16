import itertools
import os
import shutil
import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig, PassVersionChecker
from paddle.static import nn

class TRTInstanceNormTest(InferencePassTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.bs = 4
        self.channel = 4
        self.height = 8
        self.width = 8
        self.precision = AnalysisConfig.Precision.Float32
        self.serialize = False
        self.enable_trt = True

    def build(self):
        if False:
            while True:
                i = 10
        self.trt_parameters = InferencePassTest.TensorRTParam(1 << 30, self.bs, 2, self.precision, self.serialize, False)
        with base.program_guard(self.main_program, self.startup_program):
            shape = [-1, self.channel, self.height, self.width]
            data = paddle.static.data(name='in', shape=shape, dtype='float32')
            instance_norm_out = nn.instance_norm(data)
            out = nn.batch_norm(instance_norm_out, is_test=True)
        shape[0] = self.bs
        self.feeds = {'in': np.random.random(shape).astype('float32')}
        self.fetch_list = [out]

    def check_output(self, remove_cache=False):
        if False:
            for i in range(10):
                print('nop')
        opt_path = os.path.join(self.path, '_opt_cache')
        if remove_cache and os.path.exists(opt_path):
            shutil.rmtree(opt_path)
        if core.is_compiled_with_cuda():
            use_gpu = True
            atol = 1e-05
            if self.trt_parameters.precision == AnalysisConfig.Precision.Half:
                atol = 0.02
            self.check_output_with_option(use_gpu, atol, flatten=True)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

    def run_test(self, remove_cache=False):
        if False:
            while True:
                i = 10
        self.build()
        self.check_output(remove_cache)

    def run_all_tests(self):
        if False:
            i = 10
            return i + 15
        precision_opt = [AnalysisConfig.Precision.Float32, AnalysisConfig.Precision.Half]
        serialize_opt = [False, True]
        for (precision, serialize) in itertools.product(precision_opt, serialize_opt):
            self.precision = precision
            self.serialize = serialize
            self.run_test()

    def test_base(self):
        if False:
            while True:
                i = 10
        self.run_test()

    def test_fp16(self):
        if False:
            i = 10
            return i + 15
        self.precision = AnalysisConfig.Precision.Half
        self.run_test()

    def test_serialize(self):
        if False:
            print('Hello World!')
        self.serialize = True
        self.run_test(remove_cache=True)

    def test_all(self):
        if False:
            i = 10
            return i + 15
        self.run_all_tests()
if __name__ == '__main__':
    unittest.main()