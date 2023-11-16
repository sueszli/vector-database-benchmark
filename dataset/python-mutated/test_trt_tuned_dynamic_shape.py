import unittest
import numpy as np
import paddle
paddle.enable_static()
from paddle import base
from paddle.inference import Config, create_predictor

class TRTTunedDynamicShapeTest(unittest.TestCase):

    def get_model(self):
        if False:
            i = 10
            return i + 15
        place = base.CUDAPlace(0)
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

    def get_config(self, model, params, tuned=False):
        if False:
            return 10
        config = Config()
        config.set_model_buffer(model, len(model), params, len(params))
        config.enable_use_gpu(100, 0)
        config.set_optim_cache_dir('tuned_test')
        if tuned:
            config.collect_shape_range_info('shape_range.pbtxt')
        else:
            config.enable_tensorrt_engine(workspace_size=1024, max_batch_size=1, min_subgraph_size=0, precision_mode=paddle.inference.PrecisionType.Float32, use_static=True, use_calib_mode=False)
            config.enable_tuned_tensorrt_dynamic_shape('shape_range.pbtxt', True)
        return config

    def predictor_run(self, config, in_data):
        if False:
            for i in range(10):
                print('nop')
        predictor = create_predictor(config)
        in_names = predictor.get_input_names()
        in_handle = predictor.get_input_handle(in_names[0])
        in_handle.copy_from_cpu(in_data)
        predictor.run()

    def test_tuned_dynamic_shape_run(self):
        if False:
            i = 10
            return i + 15
        (program, params) = self.get_model()
        config = self.get_config(program, params, tuned=True)
        self.predictor_run(config, np.ones((1, 6, 64, 64)).astype(np.float32))
        config2 = self.get_config(program, params, tuned=False)
        self.predictor_run(config2, np.ones((1, 6, 32, 32)).astype(np.float32))
if __name__ == '__main__':
    unittest.main()