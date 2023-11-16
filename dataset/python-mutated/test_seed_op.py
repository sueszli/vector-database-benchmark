import unittest
import numpy as np
from op_test import OpTest
import paddle
from paddle import static
paddle.enable_static()

class TestSeedOpFixSeed(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'seed'
        self.inputs = {}
        self.attrs = {'seed': 123}
        self.outputs = {'Out': np.array([123]).astype('int')}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output()

class TestSeedOpDiffSeed(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'seed'
        self.inputs = {}
        self.attrs = {'seed': 0}
        self.outputs = {'Out': np.array([123]).astype('int')}

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output(no_check_set=['Out'])

class TestDropoutWithRandomSeedGenerator(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        paddle.framework.random.set_random_seed_generator('seed0', 123)
        paddle.framework.random.set_random_seed_generator('seed1', 123)
        self.rng0 = paddle.framework.random.get_random_seed_generator('seed0')
        self.rng1 = paddle.framework.random.get_random_seed_generator('seed1')
        self.places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def check_static_result(self, place):
        if False:
            i = 10
            return i + 15
        from paddle.distributed.fleet.meta_parallel.parallel_layers import random
        with static.program_guard(static.Program(), static.Program()):
            res1 = random.determinate_seed('seed0')
            exe = static.Executor(place)
            res_list = [res1]
            for i in range(2):
                (out1,) = exe.run(static.default_main_program(), fetch_list=res_list)
                self.assertEqual(out1, np.cast['int32'](self.rng1.random()))

    def test_static(self):
        if False:
            while True:
                i = 10
        for place in self.places:
            self.check_static_result(place=place)
if __name__ == '__main__':
    unittest.main()