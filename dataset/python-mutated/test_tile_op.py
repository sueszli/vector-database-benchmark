import unittest
import gradient_checker
import numpy as np
from decorator_helper import prog_scope
from op_test import OpTest, convert_float_to_uint16
import paddle
from paddle import base
from paddle.base import core
from paddle.pir_utils import test_with_pir_api

class TestTileOpRank1(OpTest):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'tile'
        self.python_api = paddle.tile
        self.prim_op_type = 'prim'
        self.public_python_api = paddle.tile
        self.init_data()
        self.if_enable_cinn()
        self.inputs = {'X': np.random.random(self.ori_shape).astype('float64')}
        self.attrs = {'repeat_times': self.repeat_times}
        output = np.tile(self.inputs['X'], self.repeat_times)
        self.outputs = {'Out': output}

    def if_enable_cinn(self):
        if False:
            i = 10
            return i + 15
        self.check_cinn = True

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.ori_shape = [100]
        self.repeat_times = [2]

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_cinn=self.check_cinn, check_pir=True)

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X'], 'Out', check_prim=True, check_pir=True, check_prim_pir=True)

class TestTileOpRank_ZeroDim1(TestTileOpRank1):

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.ori_shape = []
        self.repeat_times = []

    def if_enable_cinn(self):
        if False:
            return 10
        self.check_cinn = False
        self.enable_cinn = False

class TestTileOpRank_ZeroDim2(TestTileOpRank1):

    def init_data(self):
        if False:
            return 10
        self.ori_shape = []
        self.repeat_times = [2]

    def if_enable_cinn(self):
        if False:
            while True:
                i = 10
        self.check_cinn = False
        self.enable_cinn = False

class TestTileOpRank_ZeroDim3(TestTileOpRank1):

    def init_data(self):
        if False:
            print('Hello World!')
        self.ori_shape = []
        self.repeat_times = [2, 3]

    def if_enable_cinn(self):
        if False:
            while True:
                i = 10
        self.check_cinn = False
        self.enable_cinn = False

class TestTileOpRank2Expanding(TestTileOpRank1):

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.ori_shape = [120]
        self.repeat_times = [2, 2]

    def if_enable_cinn(self):
        if False:
            while True:
                i = 10
        self.check_cinn = True

class TestTileOpRank2(TestTileOpRank1):

    def init_data(self):
        if False:
            print('Hello World!')
        self.ori_shape = [12, 14]
        self.repeat_times = [2, 3]

    def if_enable_cinn(self):
        if False:
            while True:
                i = 10
        self.check_cinn = True

class TestTileOpRank3_Corner(TestTileOpRank1):

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.ori_shape = (2, 10, 5)
        self.repeat_times = (1, 1, 1)

    def if_enable_cinn(self):
        if False:
            i = 10
            return i + 15
        self.check_cinn = True

class TestTileOpRank3_Corner2(TestTileOpRank1):

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.ori_shape = (2, 10, 5)
        self.repeat_times = (2, 2)

    def if_enable_cinn(self):
        if False:
            i = 10
            return i + 15
        self.check_cinn = True

class TestTileOpRank3(TestTileOpRank1):

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.ori_shape = (2, 4, 15)
        self.repeat_times = (2, 1, 4)

    def if_enable_cinn(self):
        if False:
            while True:
                i = 10
        self.check_cinn = True

class TestTileOpRank4(TestTileOpRank1):

    def init_data(self):
        if False:
            while True:
                i = 10
        self.ori_shape = (2, 4, 5, 7)
        self.repeat_times = (3, 2, 1, 2)

    def if_enable_cinn(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_cinn = True

class TestTileOpRank1_tensor_attr(OpTest):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'tile'
        self.python_api = paddle.tile
        self.init_data()
        repeat_times_tensor = []
        for (index, ele) in enumerate(self.repeat_times):
            repeat_times_tensor.append(('x' + str(index), np.ones(1).astype('int32') * ele))
        self.inputs = {'X': np.random.random(self.ori_shape).astype('float64'), 'repeat_times_tensor': repeat_times_tensor}
        self.attrs = {'repeat_times': self.infer_repeat_times}
        output = np.tile(self.inputs['X'], self.repeat_times)
        self.outputs = {'Out': output}

    def init_data(self):
        if False:
            return 10
        self.ori_shape = [100]
        self.repeat_times = [2]
        self.infer_repeat_times = [-1]

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            print('Hello World!')
        self.check_grad(['X'], 'Out')

class TestTileOpRank2_Corner_tensor_attr(TestTileOpRank1_tensor_attr):

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.ori_shape = [12, 14]
        self.repeat_times = [1, 1]
        self.infer_repeat_times = [1, -1]

class TestTileOpRank2_attr_tensor(TestTileOpRank1_tensor_attr):

    def init_data(self):
        if False:
            print('Hello World!')
        self.ori_shape = [12, 14]
        self.repeat_times = [2, 3]
        self.infer_repeat_times = [-1, 3]

class TestTileOpRank1_tensor(OpTest):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'tile'
        self.python_api = paddle.tile
        self.init_data()
        self.inputs = {'X': np.random.random(self.ori_shape).astype('float64'), 'RepeatTimes': np.array(self.repeat_times).astype('int32')}
        self.attrs = {}
        output = np.tile(self.inputs['X'], self.repeat_times)
        self.outputs = {'Out': output}

    def init_data(self):
        if False:
            print('Hello World!')
        self.ori_shape = [100]
        self.repeat_times = [2]

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X'], 'Out')

class TestTileOpRank2_tensor(TestTileOpRank1_tensor):

    def init_data(self):
        if False:
            return 10
        self.ori_shape = [12, 14]
        self.repeat_times = [2, 3]

class TestTileOpInteger(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'tile'
        self.python_api = paddle.tile
        self.inputs = {'X': np.random.randint(10, size=(4, 4, 5)).astype('int32')}
        self.attrs = {'repeat_times': [2, 1, 4]}
        output = np.tile(self.inputs['X'], (2, 1, 4))
        self.outputs = {'Out': output}
        self.if_enable_cinn()

    def if_enable_cinn(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_cinn = True

    def test_check_output(self):
        if False:
            return 10
        self.check_output(check_cinn=self.check_cinn, check_pir=True)

class TestTileFP16OP(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'tile'
        self.dtype = np.float16
        self.python_api = paddle.tile
        self.prim_op_type = 'prim'
        self.public_python_api = paddle.tile
        self.init_data()
        x = np.random.uniform(10, size=self.ori_shape).astype(self.dtype)
        output = np.tile(x, self.repeat_times)
        self.inputs = {'X': x}
        self.attrs = {'repeat_times': self.repeat_times}
        self.outputs = {'Out': output}
        self.if_enable_cinn()

    def if_enable_cinn(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_cinn = True

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.float16
        self.ori_shape = [100, 4, 5]
        self.repeat_times = [2, 1, 4]

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output(check_cinn=self.check_cinn, check_pir=True)

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X'], 'Out', check_prim=True, check_pir=True, check_prim_pir=True)

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not complied with CUDA and not support the bfloat16')
class TestTileBF16OP(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'tile'
        self.__class__.op_type = self.op_type
        self.python_api = paddle.tile
        self.prim_op_type = 'prim'
        self.public_python_api = paddle.tile
        self.init_data()
        x = np.random.uniform(10, size=self.ori_shape).astype(np.float32)
        output = np.tile(x, self.repeat_times)
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.attrs = {'repeat_times': self.repeat_times}
        self.outputs = {'Out': convert_float_to_uint16(output)}
        self.if_enable_cinn()

    def if_enable_cinn(self):
        if False:
            while True:
                i = 10
        self.check_cinn = True

    def test_check_output(self):
        if False:
            print('Hello World!')
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_cinn=self.check_cinn, check_pir=True)

    def init_data(self):
        if False:
            return 10
        self.dtype = np.uint16
        self.ori_shape = [100, 4, 5]
        self.repeat_times = [2, 1, 4]

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, ['X'], 'Out', check_prim=True, check_pir=True, check_prim_pir=True)

class TestTileOpBoolean(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'tile'
        self.python_api = paddle.tile
        self.inputs = {'X': np.random.randint(2, size=(2, 4, 5)).astype('bool')}
        self.attrs = {'repeat_times': [2, 1, 4]}
        output = np.tile(self.inputs['X'], (2, 1, 4))
        self.outputs = {'Out': output}
        self.if_enable_cinn()

    def if_enable_cinn(self):
        if False:
            print('Hello World!')
        self.check_cinn = True

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_cinn=self.check_cinn, check_pir=True)

class TestTileOpInt64_t(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'tile'
        self.python_api = paddle.tile
        self.inputs = {'X': np.random.randint(10, size=(2, 4, 5)).astype('int64')}
        self.attrs = {'repeat_times': [2, 1, 4]}
        output = np.tile(self.inputs['X'], (2, 1, 4))
        self.outputs = {'Out': output}
        self.if_enable_cinn()

    def if_enable_cinn(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_cinn = True

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output(check_cinn=self.check_cinn, check_pir=True)

class TestTileError(unittest.TestCase):

    @test_with_pir_api
    def test_errors(self):
        if False:
            while True:
                i = 10
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            x1 = base.create_lod_tensor(np.array([[-1]]), [[1]], base.CPUPlace())
            repeat_times = [2, 2]
            self.assertRaises(TypeError, paddle.tile, x1, repeat_times)
            x2 = paddle.static.data(name='x2', shape=[-1, 4], dtype='uint8')
            self.assertRaises(TypeError, paddle.tile, x2, repeat_times)
            x3 = paddle.static.data(name='x3', shape=[-1, 4], dtype='bool')
            x3.stop_gradient = False
            self.assertRaises(ValueError, paddle.tile, x3, repeat_times)

class TestTileAPIStatic(unittest.TestCase):

    @test_with_pir_api
    def test_api(self):
        if False:
            while True:
                i = 10
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            repeat_times = [2, 2]
            x1 = paddle.static.data(name='x1', shape=[-1, 4], dtype='int32')
            out = paddle.tile(x1, repeat_times)
            positive_2 = paddle.tensor.fill_constant([1], dtype='int32', value=2)
            out2 = paddle.tile(x1, repeat_times=[positive_2, 2])

class TestTileAPI(unittest.TestCase):

    def test_api(self):
        if False:
            print('Hello World!')
        with base.dygraph.guard():
            np_x = np.random.random([12, 14]).astype('float32')
            x = paddle.to_tensor(np_x)
            positive_2 = np.array([2]).astype('int32')
            positive_2 = paddle.to_tensor(positive_2)
            repeat_times = np.array([2, 3]).astype('int32')
            repeat_times = paddle.to_tensor(repeat_times)
            out_1 = paddle.tile(x, repeat_times=[2, 3])
            out_2 = paddle.tile(x, repeat_times=[positive_2, 3])
            out_3 = paddle.tile(x, repeat_times=repeat_times)
            np.testing.assert_array_equal(out_1.numpy(), np.tile(np_x, (2, 3)))
            np.testing.assert_array_equal(out_2.numpy(), np.tile(np_x, (2, 3)))
            np.testing.assert_array_equal(out_3.numpy(), np.tile(np_x, (2, 3)))

class TestTileDoubleGradCheck(unittest.TestCase):

    def tile_wrapper(self, x):
        if False:
            for i in range(10):
                print('nop')
        return paddle.tile(x[0], [2, 1])

    @test_with_pir_api
    @prog_scope()
    def func(self, place):
        if False:
            for i in range(10):
                print('nop')
        eps = 0.005
        dtype = np.float32
        data = paddle.static.data('data', [1, 2], dtype)
        data.persistable = True
        out = paddle.tile(data, [2, 1])
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)
        gradient_checker.double_grad_check([data], out, x_init=[data_arr], place=place, eps=eps)
        gradient_checker.double_grad_check_for_dygraph(self.tile_wrapper, [data], out, x_init=[data_arr], place=place)

    def test_grad(self):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestTileTripleGradCheck(unittest.TestCase):

    def tile_wrapper(self, x):
        if False:
            i = 10
            return i + 15
        return paddle.tile(x[0], [2, 1])

    @test_with_pir_api
    @prog_scope()
    def func(self, place):
        if False:
            print('Hello World!')
        eps = 0.005
        dtype = np.float32
        data = paddle.static.data('data', [1, 2], dtype)
        data.persistable = True
        out = paddle.tile(data, [2, 1])
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)
        gradient_checker.triple_grad_check([data], out, x_init=[data_arr], place=place, eps=eps)
        gradient_checker.triple_grad_check_for_dygraph(self.tile_wrapper, [data], out, x_init=[data_arr], place=place)

    def test_grad(self):
        if False:
            return 10
        paddle.enable_static()
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestTileAPI_ZeroDim(unittest.TestCase):

    def test_dygraph(self):
        if False:
            return 10
        paddle.disable_static()
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.tile(x, [])
        out.retain_grads()
        out.backward()
        self.assertEqual(out.shape, [])
        self.assertEqual(x.grad.shape, [])
        self.assertEqual(out.grad.shape, [])
        out = paddle.tile(x, [3])
        out.retain_grads()
        out.backward()
        self.assertEqual(out.shape, [3])
        self.assertEqual(x.grad.shape, [])
        self.assertEqual(out.grad.shape, [3])
        out = paddle.tile(x, [2, 3])
        out.retain_grads()
        out.backward()
        self.assertEqual(out.shape, [2, 3])
        self.assertEqual(x.grad.shape, [])
        self.assertEqual(out.grad.shape, [2, 3])
        paddle.enable_static()

class Testfp16TileOp(unittest.TestCase):

    @test_with_pir_api
    def testfp16(self):
        if False:
            print('Hello World!')
        input_x = np.random.random([1, 2, 3]).astype('float16')
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='x', shape=[1, 2, 3], dtype='float16')
            repeat_times = [2, 2]
            out = paddle.tile(x, repeat_times=repeat_times)
            if paddle.is_compiled_with_cuda():
                place = paddle.CUDAPlace(0)
                exe = paddle.static.Executor(place)
                exe.run(paddle.static.default_startup_program())
                out = exe.run(feed={'x': input_x}, fetch_list=[out])
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()