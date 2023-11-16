import os
import tempfile
import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, compare_legacy_with_pir, test_ast_only, test_legacy_and_pir
import paddle
from paddle import base
from paddle.autograd import PyLayer
from paddle.jit.api import to_static
from paddle.jit.dy2static.partial_program import partial_program_from
from paddle.jit.translated_layer import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX
SEED = 2020
np.random.seed(SEED)
place = base.CUDAPlace(0) if base.is_compiled_with_cuda() else base.CPUPlace()

class SimpleFcLayer(paddle.nn.Layer):

    def __init__(self, fc_size):
        if False:
            return 10
        super().__init__()
        self._linear = paddle.nn.Linear(fc_size, fc_size)

    @to_static(full_graph=True)
    def forward(self, x):
        if False:
            while True:
                i = 10
        y = self._linear(x)
        z = self._linear(y)
        out = paddle.mean(z)
        return (out, y)

class cus_tanh(PyLayer):

    @staticmethod
    def forward(ctx, x):
        if False:
            return 10
        y = paddle.tanh(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        if False:
            i = 10
            return i + 15
        (y,) = ctx.saved_tensor()
        grad = dy * (1 - paddle.square(y))
        return grad

class SimplePyLayerNet(paddle.nn.Layer):

    def __init__(self, fc_size):
        if False:
            while True:
                i = 10
        super().__init__()
        self._linear = paddle.nn.Linear(fc_size, fc_size)

    @to_static(full_graph=True)
    def forward(self, x):
        if False:
            print('Hello World!')
        y = self._linear(x)
        out = cus_tanh.apply(y)
        loss = paddle.mean(out)
        return (loss, out)

class TestDyToStaticSaveInferenceModel(Dy2StTestBase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.temp_dir.cleanup()

    @test_ast_only
    def test_save_inference_model(self):
        if False:
            for i in range(10):
                print('nop')
        fc_size = 20
        x_data = np.random.random((fc_size, fc_size)).astype('float32')
        with base.dygraph.guard(place):
            base.default_startup_program().random_seed = SEED
            base.default_main_program().random_seed = SEED
            x = base.dygraph.to_variable(x_data)
            layer = SimpleFcLayer(fc_size)
            adam = paddle.optimizer.SGD(learning_rate=0.1, parameters=layer.parameters())
            for i in range(5):
                (loss, pred) = layer(x)
                loss.backward()
                adam.minimize(loss)
                layer.clear_gradients()
            infer_model_prefix = os.path.join(self.temp_dir.name, 'test_dy2stat_inference_in_guard/model')
            infer_model_dir = os.path.join(self.temp_dir.name, 'test_dy2stat_inference_in_guard')
            paddle.jit.save(layer=layer, path=infer_model_prefix, input_spec=[x], output_spec=[pred])
            (dygraph_out, _) = layer(x)
        self.check_save_inference_model(layer, [x_data], dygraph_out.numpy())
        self.check_save_inference_model(layer, [x_data], dygraph_out.numpy(), fetch=[loss])
        self.check_save_inference_model(layer, [x_data], dygraph_out.numpy(), feed=[x])

    @test_ast_only
    def test_save_pylayer_model(self):
        if False:
            return 10
        fc_size = 20
        x_data = np.random.random((fc_size, fc_size)).astype('float32')
        paddle.base.framework._set_expected_place(place)
        base.default_startup_program().random_seed = SEED
        base.default_main_program().random_seed = SEED
        paddle.disable_static()
        x = base.dygraph.to_variable(x_data)
        layer = SimplePyLayerNet(fc_size)
        adam = paddle.optimizer.SGD(learning_rate=0.1, parameters=layer.parameters())
        for i in range(5):
            (loss, pred) = layer(x)
            loss.backward()
            adam.minimize(loss)
            layer.clear_gradients()
        infer_model_prefix = os.path.join(self.temp_dir.name, 'test_dy2stat_inference_in_guard/model_pylayer')
        paddle.jit.save(layer=layer, path=infer_model_prefix, input_spec=[x], output_spec=[pred])
        (loss_out, _) = layer(x)
        loss_out_numpy = float(loss_out)
        self.check_save_inference_model(layer, [x_data], loss_out_numpy, enable_pir=False)
        self.check_save_inference_model(layer, [x_data], loss_out_numpy, fetch=[loss], enable_pir=False)
        self.check_save_inference_model(layer, [x_data], loss_out_numpy, feed=[x], enable_pir=False)

    def check_save_inference_model(self, model, inputs, gt_out, feed=None, fetch=None, enable_pir=True):
        if False:
            return 10
        expected_persistable_vars = {p.name for p in model.parameters()}
        infer_model_prefix = os.path.join(self.temp_dir.name, 'test_dy2stat_inference/model')
        infer_model_dir = os.path.join(self.temp_dir.name, 'test_dy2stat_inference')
        model_filename = 'model' + INFER_MODEL_SUFFIX
        params_filename = 'model' + INFER_PARAMS_SUFFIX
        paddle.jit.save(layer=model, path=infer_model_prefix, input_spec=feed if feed else None, output_spec=fetch if fetch else None)
        if enable_pir:
            wrapped_load_and_run_inference = compare_legacy_with_pir(self.load_and_run_inference)
            infer_out = wrapped_load_and_run_inference(infer_model_dir, model_filename, params_filename, inputs)
        else:
            infer_out = self.load_and_run_inference(infer_model_dir, model_filename, params_filename, inputs)
        np.testing.assert_allclose(gt_out, infer_out, rtol=1e-05)

    def load_and_run_inference(self, model_path, model_filename, params_filename, inputs):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        exe = base.Executor(place)
        [inference_program, feed_target_names, fetch_targets] = paddle.static.io.load_inference_model(path_prefix=model_path, executor=exe, model_filename=model_filename, params_filename=params_filename)
        results = exe.run(inference_program, feed=dict(zip(feed_target_names, inputs)), fetch_list=fetch_targets)
        return np.array(results[0])

class TestPartialProgramRaiseError(Dy2StTestBase):

    @test_ast_only
    @test_legacy_and_pir
    def test_param_type(self):
        if False:
            print('Hello World!')
        paddle.jit.enable_to_static(True)
        x_data = np.random.random((20, 20)).astype('float32')
        with base.dygraph.guard(base.CPUPlace()):
            net = SimpleFcLayer(20)
            x = base.dygraph.to_variable(x_data)
            out = net(x)
            program_cache = net.forward.program_cache
            (_, (concrete_program, _)) = program_cache.last()
            params = concrete_program.parameters
            concrete_program.parameters = params[0]
            with self.assertRaises(TypeError):
                partial_program_from(concrete_program)
            params[0] = 'linear.w.0'
            concrete_program.parameters = params
            with self.assertRaises(TypeError):
                partial_program_from(concrete_program)
if __name__ == '__main__':
    unittest.main()