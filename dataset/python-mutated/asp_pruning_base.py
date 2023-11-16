import unittest
import numpy as np
import paddle
from paddle import base
from paddle.base import core
from paddle.incubate.asp import ASPHelper
paddle.enable_static()

class TestASPHelperPruningBase(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.main_program = base.Program()
        self.startup_program = base.Program()

        def build_model():
            if False:
                while True:
                    i = 10
            img = paddle.static.data(name='img', shape=[None, 3, 32, 32], dtype='float32')
            label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
            hidden = paddle.static.nn.conv2d(input=img, num_filters=4, filter_size=3, padding=2, act='relu')
            hidden = paddle.static.nn.fc(x=hidden, size=32, activation='relu')
            prediction = paddle.static.nn.fc(x=hidden, size=10, activation='softmax')
            return (img, label, prediction)
        with base.program_guard(self.main_program, self.startup_program):
            (self.img, self.label, self.predict) = build_model()

    def run_inference_pruning_test(self, get_mask_gen_func, get_mask_check_func):
        if False:
            for i in range(10):
                print('nop')
        place = paddle.CPUPlace()
        if core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        exe = base.Executor(place)
        self.__pruning_and_checking(exe, place, get_mask_gen_func, get_mask_check_func, False)

    def run_training_pruning_test(self, get_mask_gen_func, get_mask_check_func):
        if False:
            print('Hello World!')
        with base.program_guard(self.main_program, self.startup_program):
            loss = paddle.mean(paddle.nn.functional.cross_entropy(input=self.predict, label=self.label, reduction='none', use_softmax=False))
            optimizer = paddle.incubate.asp.decorate(paddle.optimizer.SGD(learning_rate=0.01))
            optimizer.minimize(loss, self.startup_program)
        place = paddle.CPUPlace()
        if core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        exe = base.Executor(place)
        self.__pruning_and_checking(exe, place, get_mask_gen_func, get_mask_check_func, True)

    def __pruning_and_checking(self, exe, place, mask_func_name, check_func_name, with_mask):
        if False:
            for i in range(10):
                print('nop')
        exe.run(self.startup_program)
        paddle.incubate.asp.prune_model(self.main_program, mask_algo=mask_func_name, with_mask=with_mask)
        for param in self.main_program.global_block().all_parameters():
            if ASPHelper._is_supported_layer(self.main_program, param.name):
                mat = np.array(base.global_scope().find_var(param.name).get_tensor())
                self.assertTrue(paddle.incubate.asp.check_sparsity(mat.T, func_name=check_func_name, n=2, m=4))