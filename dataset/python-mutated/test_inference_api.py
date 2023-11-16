import unittest
import paddle
paddle.enable_static()
import numpy as np
from paddle import base
from paddle.base.core import PaddleDType, PaddleTensor
from paddle.framework import core
from paddle.inference import Config, create_predictor, get_trt_compile_version, get_trt_runtime_version

class TestInferenceApi(unittest.TestCase):

    def test_inference_api(self):
        if False:
            print('Hello World!')
        tensor32 = np.random.randint(10, 20, size=[20, 2]).astype('int32')
        paddletensor32 = PaddleTensor(tensor32)
        dtype32 = paddletensor32.dtype
        self.assertEqual(dtype32, PaddleDType.INT32)
        self.assertEqual(paddletensor32.data.tolist('int32'), tensor32.ravel().tolist())
        paddletensor32.data.reset(tensor32)
        self.assertEqual(paddletensor32.as_ndarray().ravel().tolist(), tensor32.ravel().tolist())
        tensor64 = np.random.randint(10, 20, size=[20, 2]).astype('int64')
        paddletensor64 = PaddleTensor(tensor64)
        dtype64 = paddletensor64.dtype
        self.assertEqual(dtype64, PaddleDType.INT64)
        self.assertEqual(paddletensor64.data.tolist('int64'), tensor64.ravel().tolist())
        paddletensor64.data.reset(tensor64)
        self.assertEqual(paddletensor64.as_ndarray().ravel().tolist(), tensor64.ravel().tolist())
        tensor_float = np.random.randn(20, 2).astype('float32')
        paddletensor_float = PaddleTensor(tensor_float)
        dtype_float = paddletensor_float.dtype
        self.assertEqual(dtype_float, PaddleDType.FLOAT32)
        self.assertEqual(paddletensor_float.data.tolist('float32'), tensor_float.ravel().tolist())
        paddletensor_float.data.reset(tensor_float)
        self.assertEqual(paddletensor_float.as_ndarray().ravel().tolist(), tensor_float.ravel().tolist())

def get_sample_model():
    if False:
        while True:
            i = 10
    place = base.CPUPlace()
    exe = base.Executor(place)
    main_program = base.Program()
    startup_program = base.Program()
    with base.program_guard(main_program, startup_program):
        data = paddle.static.data(name='data', shape=[-1, 6, 64, 64], dtype='float32')
        conv_out = paddle.static.nn.conv2d(input=data, num_filters=3, filter_size=3, groups=1, padding=0, bias_attr=False, act=None)
    exe.run(startup_program)
    serialized_program = paddle.static.serialize_program(data, conv_out, program=main_program)
    serialized_params = paddle.static.serialize_persistables(data, conv_out, executor=exe, program=main_program)
    return (serialized_program, serialized_params)

def get_sample_model_cuda(data_type):
    if False:
        i = 10
        return i + 15
    place = base.CUDAPlace(0)
    exe = base.Executor(place)
    main_program = base.Program()
    startup_program = base.Program()
    with base.program_guard(main_program, startup_program):
        data = paddle.static.data(name='data', shape=[-1, 6, 64, 64], dtype=data_type)
        data_float = paddle.cast(data, 'bfloat16')
        res = paddle.static.nn.conv2d(input=data_float, num_filters=3, filter_size=3, groups=1, padding=0, bias_attr=False, act=None)
    exe.run(startup_program)
    serialized_program = paddle.static.serialize_program(data, res, program=main_program)
    serialized_params = paddle.static.serialize_persistables(data, res, executor=exe, program=main_program)
    return (serialized_program, serialized_params)

class TestInferenceBaseAPI(unittest.TestCase):

    def get_config(self, model, params):
        if False:
            i = 10
            return i + 15
        config = Config()
        config.set_model_buffer(model, len(model), params, len(params))
        config.enable_use_gpu(100, 0)
        return config

    def test_apis(self):
        if False:
            return 10
        print('trt compile version:', get_trt_compile_version())
        print('trt runtime version:', get_trt_runtime_version())
        (program, params) = get_sample_model()
        config = self.get_config(program, params)
        predictor = create_predictor(config)
        in_names = predictor.get_input_names()
        in_handle = predictor.get_input_handle(in_names[0])
        in_data = np.ones((1, 6, 32, 32)).astype(np.float32)
        in_handle.copy_from_cpu(in_data)
        predictor.run()

    def test_wrong_input(self):
        if False:
            i = 10
            return i + 15
        (program, params) = get_sample_model()
        config = self.get_config(program, params)
        predictor = create_predictor(config)
        in_names = predictor.get_input_names()
        in_handle = predictor.get_input_handle(in_names[0])
        with self.assertRaises(TypeError):
            in_data = np.ones((1, 6, 64, 64)).astype(np.float32)
            in_handle.copy_from_cpu(list(in_data))
            predictor.run()
        with self.assertRaises(TypeError):
            in_handle.share_external_data(paddle.to_tensor(np.full((1, 6, 32, 32), 1.0, 'float32'), place=paddle.CPUPlace()))
            predictor.run()

    def test_share_external_data(self):
        if False:
            while True:
                i = 10
        (program, params) = get_sample_model()

        def test_lod_tensor():
            if False:
                while True:
                    i = 10
            config = Config()
            config.set_model_buffer(program, len(program), params, len(params))
            predictor = create_predictor(config)
            in_names = predictor.get_input_names()
            in_handle = predictor.get_input_handle(in_names[0])
            in_data = paddle.base.create_lod_tensor(np.full((1, 6, 32, 32), 1.0, 'float32'), [[1]], paddle.base.CPUPlace())
            in_handle.share_external_data(in_data)
            predictor.run()

        def test_paddle_tensor():
            if False:
                print('Hello World!')
            paddle.disable_static()
            config = self.get_config(program, params)
            predictor = create_predictor(config)
            in_names = predictor.get_input_names()
            in_handle = predictor.get_input_handle(in_names[0])
            in_data = paddle.Tensor(np.ones((1, 6, 32, 32)).astype(np.float32))
            in_handle.share_external_data(in_data)
            predictor.run()
            paddle.enable_static()
        test_lod_tensor()
        test_paddle_tensor()

@unittest.skipIf(not core.is_compiled_with_cuda() or paddle.get_cudnn_version() < 8100 or paddle.device.cuda.get_device_capability()[0] < 8, 'share_external_data_bf16 requires cudnn >= 8.1 and CUDA_ARCH >= 8')
class TestInferenceShareExternalDataAPI(unittest.TestCase):

    def get_config(self, model, params):
        if False:
            print('Hello World!')
        config = Config()
        config.set_model_buffer(model, len(model), params, len(params))
        config.enable_use_gpu(100, 0)
        return config

    def test_share_external_data_cuda(self):
        if False:
            i = 10
            return i + 15

        def test_paddle_tensor_bf16():
            if False:
                for i in range(10):
                    print('nop')
            paddle.set_default_dtype('bfloat16')
            (program, params) = get_sample_model_cuda('bfloat16')
            paddle.disable_static()
            config = self.get_config(program, params)
            predictor = create_predictor(config)
            in_names = predictor.get_input_names()
            in_handle = predictor.get_input_handle(in_names[0])
            in_data = paddle.to_tensor(np.ones((1, 6, 32, 32)), 'bfloat16')
            in_handle.share_external_data(in_data)
            predictor.run()
            paddle.set_default_dtype('float32')
            paddle.enable_static()

        def test_paddle_tensor_bool():
            if False:
                while True:
                    i = 10
            paddle.set_default_dtype('bfloat16')
            (program, params) = get_sample_model_cuda('bool')
            paddle.disable_static()
            config = self.get_config(program, params)
            predictor = create_predictor(config)
            in_names = predictor.get_input_names()
            in_handle = predictor.get_input_handle(in_names[0])
            in_data = paddle.to_tensor(np.ones((1, 6, 32, 32)), 'bool')
            in_handle.share_external_data(in_data)
            predictor.run()
            paddle.set_default_dtype('float32')
            paddle.enable_static()
        test_paddle_tensor_bf16()
        test_paddle_tensor_bool()
if __name__ == '__main__':
    unittest.main()