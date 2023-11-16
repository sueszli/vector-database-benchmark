import os
import tempfile
import unittest
from io import BytesIO
import numpy as np
from test_imperative_base import new_program_scope
import paddle
import paddle.optimizer as opt
from paddle import base, nn
from paddle.base import framework
from paddle.optimizer import Adam
from paddle.optimizer.lr import LRScheduler
BATCH_SIZE = 16
BATCH_NUM = 4
EPOCH_NUM = 4
SEED = 10
IMAGE_SIZE = 784
CLASS_NUM = 10
LARGE_PARAM = 2 ** 26

def random_batch_reader():
    if False:
        i = 10
        return i + 15

    def _get_random_inputs_and_labels():
        if False:
            print('Hello World!')
        np.random.seed(SEED)
        image = np.random.random([BATCH_SIZE, IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, CLASS_NUM - 1, (BATCH_SIZE, 1)).astype('int64')
        return (image, label)

    def __reader__():
        if False:
            for i in range(10):
                print('nop')
        for _ in range(BATCH_NUM):
            (batch_image, batch_label) = _get_random_inputs_and_labels()
            batch_image = paddle.to_tensor(batch_image)
            batch_label = paddle.to_tensor(batch_label)
            yield (batch_image, batch_label)
    return __reader__

class LinearNet(nn.Layer):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        return self._linear(x)

class LayerWithLargeParameters(paddle.nn.Layer):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self._l = paddle.nn.Linear(10, LARGE_PARAM)

    def forward(self, x):
        if False:
            print('Hello World!')
        y = self._l(x)
        return y

def train(layer, loader, loss_fn, opt):
    if False:
        while True:
            i = 10
    for epoch_id in range(EPOCH_NUM):
        for (batch_id, (image, label)) in enumerate(loader()):
            out = layer(image)
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()
            opt.clear_grad()

class TestSaveLoadLargeParameters(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.temp_dir.cleanup()

    def test_large_parameters_paddle_save(self):
        if False:
            return 10
        paddle.disable_static()
        paddle.set_device('cpu')
        layer = LayerWithLargeParameters()
        save_dict = layer.state_dict()
        path = os.path.join(self.temp_dir.name, 'test_paddle_save_load_large_param_save', 'layer.pdparams')
        protocol = 4
        paddle.save(save_dict, path, protocol=protocol)
        dict_load = paddle.load(path, return_numpy=True)
        for (key, value) in save_dict.items():
            np.testing.assert_array_equal(dict_load[key], value.numpy())

class TestSaveLoadPickle(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.temp_dir.cleanup()

    def test_pickle_protocol(self):
        if False:
            return 10
        paddle.disable_static()
        layer = LinearNet()
        save_dict = layer.state_dict()
        path = os.path.join(self.temp_dir.name, 'test_paddle_save_load_pickle_protocol', 'layer.pdparams')
        with self.assertRaises(ValueError):
            paddle.save(save_dict, path, 2.0)
        with self.assertRaises(ValueError):
            paddle.save(save_dict, path, 1)
        with self.assertRaises(ValueError):
            paddle.save(save_dict, path, 5)
        protocols = [2, 3, 4]
        for protocol in protocols:
            paddle.save(save_dict, path, pickle_protocol=protocol)
            dict_load = paddle.load(path)
            for (key, value) in save_dict.items():
                np.testing.assert_array_equal(dict_load[key].numpy(), value.numpy())

class TestSaveLoadAny(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.temp_dir.cleanup()

    def set_zero(self, prog, place, scope=None):
        if False:
            return 10
        if scope is None:
            scope = base.global_scope()
        for var in prog.list_vars():
            if isinstance(var, framework.Parameter) or var.persistable:
                ten = scope.find_var(var.name).get_tensor()
                if ten is not None:
                    ten.set(np.zeros_like(np.array(ten)), place)
                    new_t = np.array(scope.find_var(var.name).get_tensor())
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

    def replace_static_save(self, program, model_path, pickle_protocol=2):
        if False:
            return 10
        with self.assertRaises(TypeError):
            program.state_dict(1)
        with self.assertRaises(TypeError):
            program.state_dict(scope=1)
        with self.assertRaises(ValueError):
            program.state_dict('x')
        state_dict_param = program.state_dict('param')
        paddle.save(state_dict_param, model_path + '.pdparams')
        state_dict_opt = program.state_dict('opt')
        paddle.save(state_dict_opt, model_path + '.pdopt')
        state_dict_all = program.state_dict()
        paddle.save(state_dict_opt, model_path + '.pdall')

    def replace_static_load(self, program, model_path):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(TypeError):
            program.set_state_dict(1)
        state_dict_param = paddle.load(model_path + '.pdparams')
        state_dict_param['fake_var_name.@@'] = np.random.randn(1, 2)
        state_dict_param['static_x'] = 'UserWarning'
        program.set_state_dict(state_dict_param)
        state_dict_param['static_x'] = np.random.randn(1, 2)
        program.set_state_dict(state_dict_param)
        program.set_state_dict(state_dict_param)
        state_dict_opt = paddle.load(model_path + '.pdopt')
        program.set_state_dict(state_dict_opt)

    def test_replace_static_save_load(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        with new_program_scope():
            x = paddle.static.data(name='static_x', shape=[None, IMAGE_SIZE], dtype='float32')
            z = paddle.static.nn.fc(x, 10)
            z = paddle.static.nn.fc(z, 10, bias_attr=False)
            loss = paddle.mean(z)
            opt = Adam(learning_rate=0.001)
            opt.minimize(loss)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            prog = paddle.static.default_main_program()
            fake_inputs = np.random.randn(2, IMAGE_SIZE).astype('float32')
            exe.run(prog, feed={'static_x': fake_inputs}, fetch_list=[loss])
            base_map = {}
            for var in prog.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    t = np.array(base.global_scope().find_var(var.name).get_tensor())
                    base_map[var.name] = t
            path = os.path.join(self.temp_dir.name, 'test_replace_static_save_load', 'model')
            self.replace_static_save(prog, path)
            self.set_zero(prog, place)
            paddle.static.load(prog, path)
            for var in prog.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(base.global_scope().find_var(var.name).get_tensor())
                    base_t = base_map[var.name]
                    np.testing.assert_array_equal(new_t, np.array(base_t))
            paddle.static.save(prog, path)
            self.set_zero(prog, place)
            self.replace_static_load(prog, path)
            for var in prog.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(base.global_scope().find_var(var.name).get_tensor())
                    base_t = base_map[var.name]
                    np.testing.assert_array_equal(new_t, base_t)
            path_vars = 'test_replace_save_load_return_tensor_static/model'
            for var in prog.list_vars():
                if var.persistable:
                    tensor = var.get_value(base.global_scope())
                    paddle.save(tensor, os.path.join(self.temp_dir.name, path_vars, var.name))
            with self.assertRaises(TypeError):
                var.get_value('base.global_scope()')
            with self.assertRaises(ValueError):
                x.get_value()
            with self.assertRaises(TypeError):
                x.set_value('1')
            fake_data = np.zeros([3, 2, 1, 2, 3])
            with self.assertRaises(TypeError):
                x.set_value(fake_data, '1')
            with self.assertRaises(ValueError):
                x.set_value(fake_data)
            with self.assertRaises(ValueError):
                var.set_value(fake_data)
            self.set_zero(prog, place)
            for var in prog.list_vars():
                if var.persistable:
                    tensor = paddle.load(os.path.join(self.temp_dir.name, path_vars, var.name), return_numpy=False)
                    var.set_value(tensor)
                    new_t = np.array(base.global_scope().find_var(var.name).get_tensor())
                    base_t = base_map[var.name]
                    np.testing.assert_array_equal(new_t, base_t)

    def test_paddle_save_load_v2(self):
        if False:
            while True:
                i = 10
        paddle.disable_static()

        class StepDecay(LRScheduler):

            def __init__(self, learning_rate, step_size, gamma=0.1, last_epoch=-1, verbose=False):
                if False:
                    print('Hello World!')
                self.step_size = step_size
                self.gamma = gamma
                super().__init__(learning_rate, last_epoch, verbose)

            def get_lr(self):
                if False:
                    print('Hello World!')
                i = self.last_epoch // self.step_size
                return self.base_lr * self.gamma ** i
        layer = LinearNet()
        inps = paddle.randn([2, IMAGE_SIZE])
        adam = opt.Adam(learning_rate=StepDecay(0.1, 1), parameters=layer.parameters())
        y = layer(inps)
        y.mean().backward()
        adam.step()
        state_dict = adam.state_dict()
        path = os.path.join(self.temp_dir.name, 'paddle_save_load_v2/model.pdparams')
        with self.assertRaises(TypeError):
            paddle.save(state_dict, path, use_binary_format='False')
        paddle.framework.io._legacy_save(state_dict, path)
        load_dict_tensor = paddle.load(path, return_numpy=False)
        paddle.save(state_dict, path)
        load_dict_np = paddle.framework.io._legacy_load(path)
        for (k, v) in state_dict.items():
            if isinstance(v, dict):
                self.assertTrue(v == load_dict_tensor[k])
            else:
                np.testing.assert_array_equal(v.numpy(), load_dict_tensor[k].numpy())
                if not np.array_equal(v.numpy(), load_dict_np[k]):
                    print(v.numpy())
                    print(load_dict_np[k])
                np.testing.assert_array_equal(v.numpy(), load_dict_np[k])

    def test_single_pickle_var_dygraph(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()
        layer = LinearNet()
        path = os.path.join(self.temp_dir.name, 'paddle_save_load_v2/var_dygraph')
        tensor = layer._linear.weight
        with self.assertRaises(ValueError):
            paddle.save(tensor, path, pickle_protocol='3')
        with self.assertRaises(ValueError):
            paddle.save(tensor, path, pickle_protocol=5)
        paddle.save(tensor, path)
        t_dygraph = paddle.load(path)
        np_dygraph = paddle.load(path, return_numpy=True)
        self.assertTrue(isinstance(t_dygraph, paddle.base.core.eager.Tensor))
        np.testing.assert_array_equal(tensor.numpy(), np_dygraph)
        np.testing.assert_array_equal(tensor.numpy(), t_dygraph.numpy())
        paddle.enable_static()
        lod_static = paddle.load(path)
        np_static = paddle.load(path, return_numpy=True)
        self.assertTrue(isinstance(lod_static, paddle.base.core.LoDTensor))
        np.testing.assert_array_equal(tensor.numpy(), np_static)
        np.testing.assert_array_equal(tensor.numpy(), np.array(lod_static))

    def test_single_pickle_var_static(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        with new_program_scope():
            x = paddle.static.data(name='x', shape=[None, IMAGE_SIZE], dtype='float32')
            z = paddle.static.nn.fc(x, 128)
            loss = paddle.mean(z)
            place = base.CPUPlace() if not paddle.base.core.is_compiled_with_cuda() else base.CUDAPlace(0)
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            prog = paddle.static.default_main_program()
            for var in prog.list_vars():
                if list(var.shape) == [IMAGE_SIZE, 128]:
                    tensor = var.get_value()
                    break
            scope = base.global_scope()
        origin_tensor = np.array(tensor)
        path = os.path.join(self.temp_dir.name, 'test_single_pickle_var_static/var')
        paddle.save(tensor, path)
        self.set_zero(prog, place, scope)
        lod_static = paddle.load(path)
        np_static = paddle.load(path, return_numpy=True)
        var.set_value(np_static, scope)
        np.testing.assert_array_equal(origin_tensor, np.array(tensor))
        self.set_zero(prog, place, scope)
        var.set_value(lod_static, scope)
        np.testing.assert_array_equal(origin_tensor, np.array(tensor))
        paddle.disable_static()
        var_dygraph = paddle.load(path)
        np_dygraph = paddle.load(path, return_numpy=True)
        np.testing.assert_array_equal(np.array(tensor), np_dygraph)
        np.testing.assert_array_equal(np.array(tensor), var_dygraph.numpy())

    def test_dygraph_save_static_load(self):
        if False:
            print('Hello World!')
        inps = np.random.randn(1, IMAGE_SIZE).astype('float32')
        path = os.path.join(self.temp_dir.name, 'test_dygraph_save_static_load/dy-static.pdparams')
        paddle.disable_static()
        with paddle.utils.unique_name.guard():
            layer = LinearNet()
            state_dict_dy = layer.state_dict()
            paddle.save(state_dict_dy, path)
        paddle.enable_static()
        with new_program_scope():
            layer = LinearNet()
            data = paddle.static.data(name='x_static_save', shape=(None, IMAGE_SIZE), dtype='float32')
            y_static = layer(data)
            program = paddle.static.default_main_program()
            place = base.CPUPlace() if not paddle.base.core.is_compiled_with_cuda() else base.CUDAPlace(0)
            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(paddle.static.default_startup_program())
            state_dict = paddle.load(path, keep_name_table=True)
            program.set_state_dict(state_dict)
            state_dict_param = program.state_dict('param')
            for (name, tensor) in state_dict_dy.items():
                np.testing.assert_array_equal(tensor.numpy(), np.array(state_dict_param[tensor.name]))

    def test_save_load_complex_object_dygraph_save(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()
        layer = paddle.nn.Linear(3, 4)
        state_dict = layer.state_dict()
        obj1 = [paddle.randn([3, 4], dtype='float32'), np.random.randn(5, 6), ('fake_weight', np.ones([7, 8], dtype='float32'))]
        obj2 = {'k1': obj1, 'k2': state_dict, 'epoch': 123}
        obj3 = (paddle.randn([5, 4], dtype='float32'), np.random.randn(3, 4).astype('float32'), {'state_dict': state_dict, 'opt': state_dict})
        obj4 = (np.random.randn(5, 6), (123,))
        path1 = os.path.join(self.temp_dir.name, 'test_save_load_any_complex_object_dygraph/obj1')
        path2 = os.path.join(self.temp_dir.name, 'test_save_load_any_complex_object_dygraph/obj2')
        path3 = os.path.join(self.temp_dir.name, 'test_save_load_any_complex_object_dygraph/obj3')
        path4 = os.path.join(self.temp_dir.name, 'test_save_load_any_complex_object_dygraph/obj4')
        paddle.save(obj1, path1)
        paddle.save(obj2, path2)
        paddle.save(obj3, path3)
        paddle.save(obj4, path4)
        load_tensor1 = paddle.load(path1, return_numpy=False)
        load_tensor2 = paddle.load(path2, return_numpy=False)
        load_tensor3 = paddle.load(path3, return_numpy=False)
        load_tensor4 = paddle.load(path4, return_numpy=False)
        np.testing.assert_array_equal(load_tensor1[0].numpy(), obj1[0].numpy())
        np.testing.assert_array_equal(load_tensor1[1], obj1[1])
        np.testing.assert_array_equal(load_tensor1[2].numpy(), obj1[2][1])
        for i in range(len(load_tensor1)):
            self.assertTrue(type(load_tensor1[i]) == type(load_tensor2['k1'][i]))
        for (k, v) in state_dict.items():
            np.testing.assert_array_equal(v.numpy(), load_tensor2['k2'][k].numpy())
        self.assertTrue(load_tensor2['epoch'] == 123)
        np.testing.assert_array_equal(load_tensor3[0].numpy(), obj3[0].numpy())
        np.testing.assert_array_equal(np.array(load_tensor3[1]), obj3[1])
        for (k, v) in state_dict.items():
            np.testing.assert_array_equal(load_tensor3[2]['state_dict'][k].numpy(), v.numpy())
        for (k, v) in state_dict.items():
            np.testing.assert_array_equal(load_tensor3[2]['opt'][k].numpy(), v.numpy())
        np.testing.assert_array_equal(load_tensor4[0].numpy(), obj4[0])
        load_array1 = paddle.load(path1, return_numpy=True)
        load_array2 = paddle.load(path2, return_numpy=True)
        load_array3 = paddle.load(path3, return_numpy=True)
        load_array4 = paddle.load(path4, return_numpy=True)
        np.testing.assert_array_equal(load_array1[0], obj1[0].numpy())
        np.testing.assert_array_equal(load_array1[1], obj1[1])
        np.testing.assert_array_equal(load_array1[2], obj1[2][1])
        for i in range(len(load_array1)):
            self.assertTrue(type(load_array1[i]) == type(load_array2['k1'][i]))
        for (k, v) in state_dict.items():
            np.testing.assert_array_equal(v.numpy(), load_array2['k2'][k])
        self.assertTrue(load_array2['epoch'] == 123)
        np.testing.assert_array_equal(load_array3[0], obj3[0].numpy())
        np.testing.assert_array_equal(load_array3[1], obj3[1])
        for (k, v) in state_dict.items():
            np.testing.assert_array_equal(load_array3[2]['state_dict'][k], v.numpy())
        for (k, v) in state_dict.items():
            np.testing.assert_array_equal(load_array3[2]['opt'][k], v.numpy())
        np.testing.assert_array_equal(load_array4[0], obj4[0])
        paddle.enable_static()
        load_tensor1 = paddle.load(path1, return_numpy=False)
        load_tensor2 = paddle.load(path2, return_numpy=False)
        load_tensor3 = paddle.load(path3, return_numpy=False)
        load_tensor4 = paddle.load(path4, return_numpy=False)
        np.testing.assert_array_equal(np.array(load_tensor1[0]), obj1[0].numpy())
        np.testing.assert_array_equal(np.array(load_tensor1[1]), obj1[1])
        np.testing.assert_array_equal(np.array(load_tensor1[2]), obj1[2][1])
        for i in range(len(load_tensor1)):
            self.assertTrue(type(load_tensor1[i]) == type(load_tensor2['k1'][i]))
        for (k, v) in state_dict.items():
            np.testing.assert_array_equal(v.numpy(), np.array(load_tensor2['k2'][k]))
        self.assertTrue(load_tensor2['epoch'] == 123)
        self.assertTrue(isinstance(load_tensor3[0], paddle.base.core.LoDTensor))
        np.testing.assert_array_equal(np.array(load_tensor3[0]), obj3[0].numpy())
        np.testing.assert_array_equal(np.array(load_tensor3[1]), obj3[1])
        for (k, v) in state_dict.items():
            self.assertTrue(isinstance(load_tensor3[2]['state_dict'][k], paddle.base.core.LoDTensor))
            np.testing.assert_array_equal(np.array(load_tensor3[2]['state_dict'][k]), v.numpy())
        for (k, v) in state_dict.items():
            self.assertTrue(isinstance(load_tensor3[2]['opt'][k], paddle.base.core.LoDTensor))
            np.testing.assert_array_equal(np.array(load_tensor3[2]['opt'][k]), v.numpy())
        self.assertTrue(load_tensor4[0], paddle.base.core.LoDTensor)
        np.testing.assert_array_equal(np.array(load_tensor4[0]), obj4[0])
        load_array1 = paddle.load(path1, return_numpy=True)
        load_array2 = paddle.load(path2, return_numpy=True)
        load_array3 = paddle.load(path3, return_numpy=True)
        load_array4 = paddle.load(path4, return_numpy=True)
        np.testing.assert_array_equal(load_array1[0], obj1[0].numpy())
        np.testing.assert_array_equal(load_array1[1], obj1[1])
        np.testing.assert_array_equal(load_array1[2], obj1[2][1])
        for i in range(len(load_array1)):
            self.assertTrue(type(load_array1[i]) == type(load_array2['k1'][i]))
        for (k, v) in state_dict.items():
            np.testing.assert_array_equal(v.numpy(), load_array2['k2'][k])
        self.assertTrue(load_array2['epoch'] == 123)
        self.assertTrue(isinstance(load_array3[0], np.ndarray))
        np.testing.assert_array_equal(load_array3[0], obj3[0].numpy())
        np.testing.assert_array_equal(load_array3[1], obj3[1])
        for (k, v) in state_dict.items():
            np.testing.assert_array_equal(load_array3[2]['state_dict'][k], v.numpy())
        for (k, v) in state_dict.items():
            np.testing.assert_array_equal(load_array3[2]['opt'][k], v.numpy())
        np.testing.assert_array_equal(load_array4[0], obj4[0])

    def test_save_load_complex_object_static_save(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        with new_program_scope():
            x = paddle.static.data(name='x', shape=[None, IMAGE_SIZE], dtype='float32')
            z = paddle.static.nn.fc(x, 10, bias_attr=False)
            z = paddle.static.nn.fc(z, 128, bias_attr=False)
            loss = paddle.mean(z)
            place = base.CPUPlace() if not paddle.base.core.is_compiled_with_cuda() else base.CUDAPlace(0)
            prog = paddle.static.default_main_program()
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            state_dict = prog.state_dict()
            keys = list(state_dict.keys())
            obj1 = [state_dict[keys[0]], np.random.randn(5, 6), ('fake_weight', np.ones([7, 8], dtype='float32'))]
            obj2 = {'k1': obj1, 'k2': state_dict, 'epoch': 123}
            obj3 = (state_dict[keys[0]], np.ndarray([3, 4], dtype='float32'), {'state_dict': state_dict, 'opt': state_dict})
            obj4 = (np.ndarray([3, 4], dtype='float32'),)
            path1 = os.path.join(self.temp_dir.name, 'test_save_load_any_complex_object_static/obj1')
            path2 = os.path.join(self.temp_dir.name, 'test_save_load_any_complex_object_static/obj2')
            path3 = os.path.join(self.temp_dir.name, 'test_save_load_any_complex_object_static/obj3')
            path4 = os.path.join(self.temp_dir.name, 'test_save_load_any_complex_object_static/obj4')
            paddle.save(obj1, path1)
            paddle.save(obj2, path2)
            paddle.save(obj3, path3)
            paddle.save(obj4, path4)
            load_tensor1 = paddle.load(path1, return_numpy=False)
            load_tensor2 = paddle.load(path2, return_numpy=False)
            load_tensor3 = paddle.load(path3, return_numpy=False)
            load_tensor4 = paddle.load(path4, return_numpy=False)
            np.testing.assert_array_equal(np.array(load_tensor1[0]), np.array(obj1[0]))
            np.testing.assert_array_equal(np.array(load_tensor1[1]), obj1[1])
            np.testing.assert_array_equal(np.array(load_tensor1[2]), obj1[2][1])
            for i in range(len(load_tensor1)):
                self.assertTrue(type(load_tensor1[i]) == type(load_tensor2['k1'][i]))
            for (k, v) in state_dict.items():
                np.testing.assert_array_equal(np.array(v), np.array(load_tensor2['k2'][k]))
            self.assertTrue(load_tensor2['epoch'] == 123)
            self.assertTrue(isinstance(load_tensor3[0], base.core.LoDTensor))
            np.testing.assert_array_equal(np.array(load_tensor3[0]), obj3[0])
            self.assertTrue(isinstance(load_tensor3[1], base.core.LoDTensor))
            np.testing.assert_array_equal(np.array(load_tensor3[1]), obj3[1])
            for (k, v) in state_dict.items():
                self.assertTrue(isinstance(load_tensor3[2]['state_dict'][k], base.core.LoDTensor))
                np.testing.assert_array_equal(np.array(load_tensor3[2]['state_dict'][k]), np.array(v))
            for (k, v) in state_dict.items():
                self.assertTrue(isinstance(load_tensor3[2]['opt'][k], base.core.LoDTensor))
                np.testing.assert_array_equal(np.array(load_tensor3[2]['opt'][k]), np.array(v))
            self.assertTrue(isinstance(load_tensor4[0], base.core.LoDTensor))
            np.testing.assert_array_equal(np.array(load_tensor4[0]), obj4[0])
            load_array1 = paddle.load(path1, return_numpy=True)
            load_array2 = paddle.load(path2, return_numpy=True)
            load_array3 = paddle.load(path3, return_numpy=True)
            load_array4 = paddle.load(path4, return_numpy=True)
            np.testing.assert_array_equal(load_array1[0], np.array(obj1[0]))
            np.testing.assert_array_equal(load_array1[1], obj1[1])
            np.testing.assert_array_equal(load_array1[2], obj1[2][1])
            for i in range(len(load_array1)):
                self.assertTrue(type(load_array1[i]) == type(load_array2['k1'][i]))
            for (k, v) in state_dict.items():
                np.testing.assert_array_equal(np.array(v), load_array2['k2'][k])
            self.assertTrue(load_array2['epoch'] == 123)
            np.testing.assert_array_equal(load_array3[0], np.array(obj3[0]))
            np.testing.assert_array_equal(load_array3[1], obj3[1])
            for (k, v) in state_dict.items():
                np.testing.assert_array_equal(load_array3[2]['state_dict'][k], np.array(v))
            for (k, v) in state_dict.items():
                np.testing.assert_array_equal(load_array3[2]['opt'][k], np.array(v))
            np.testing.assert_array_equal(load_array4[0], obj4[0])
            paddle.disable_static()
            load_tensor1 = paddle.load(path1, return_numpy=False)
            load_tensor2 = paddle.load(path2, return_numpy=False)
            load_tensor3 = paddle.load(path3, return_numpy=False)
            load_tensor4 = paddle.load(path4, return_numpy=False)
            np.testing.assert_array_equal(np.array(load_tensor1[0]), np.array(obj1[0]))
            np.testing.assert_array_equal(np.array(load_tensor1[1]), obj1[1])
            np.testing.assert_array_equal(load_tensor1[2].numpy(), obj1[2][1])
            for i in range(len(load_tensor1)):
                self.assertTrue(type(load_tensor1[i]) == type(load_tensor2['k1'][i]))
            for (k, v) in state_dict.items():
                np.testing.assert_array_equal(np.array(v), np.array(load_tensor2['k2'][k]))
            self.assertTrue(load_tensor2['epoch'] == 123)
            self.assertTrue(isinstance(load_tensor3[0], base.core.eager.Tensor))
            np.testing.assert_array_equal(load_tensor3[0].numpy(), obj3[0])
            self.assertTrue(isinstance(load_tensor3[1], base.core.eager.Tensor))
            np.testing.assert_array_equal(load_tensor3[1].numpy(), obj3[1])
            for (k, v) in state_dict.items():
                self.assertTrue(isinstance(load_tensor3[2]['state_dict'][k], base.core.eager.Tensor))
                np.testing.assert_array_equal(load_tensor3[2]['state_dict'][k].numpy(), np.array(v))
            for (k, v) in state_dict.items():
                self.assertTrue(isinstance(load_tensor3[2]['opt'][k], base.core.eager.Tensor))
                np.testing.assert_array_equal(load_tensor3[2]['opt'][k].numpy(), np.array(v))
            self.assertTrue(isinstance(load_tensor4[0], base.core.eager.Tensor))
            np.testing.assert_array_equal(load_tensor4[0].numpy(), obj4[0])
            load_array1 = paddle.load(path1, return_numpy=True)
            load_array2 = paddle.load(path2, return_numpy=True)
            load_array3 = paddle.load(path3, return_numpy=True)
            load_array4 = paddle.load(path4, return_numpy=True)
            np.testing.assert_array_equal(load_array1[0], np.array(obj1[0]))
            np.testing.assert_array_equal(load_array1[1], obj1[1])
            np.testing.assert_array_equal(load_array1[2], obj1[2][1])
            for i in range(len(load_array1)):
                self.assertTrue(type(load_array1[i]) == type(load_array2['k1'][i]))
            for (k, v) in state_dict.items():
                np.testing.assert_array_equal(np.array(v), load_array2['k2'][k])
            self.assertTrue(load_array2['epoch'] == 123)
            np.testing.assert_array_equal(load_array3[0], np.array(obj3[0]))
            np.testing.assert_array_equal(load_array3[1], obj3[1])
            for (k, v) in state_dict.items():
                np.testing.assert_array_equal(load_array3[2]['state_dict'][k], np.array(v))
            for (k, v) in state_dict.items():
                np.testing.assert_array_equal(load_array3[2]['opt'][k], np.array(v))
            self.assertTrue(isinstance(load_array4[0], np.ndarray))
            np.testing.assert_array_equal(load_array4[0], obj4[0])

    def test_varbase_binary_var(self):
        if False:
            while True:
                i = 10
        paddle.disable_static()
        varbase = paddle.randn([3, 2], dtype='float32')
        path = os.path.join(self.temp_dir.name, 'test_paddle_save_load_varbase_binary_var/varbase')
        paddle.save(varbase, path, use_binary_format=True)
        load_array = paddle.load(path, return_numpy=True)
        load_tensor = paddle.load(path, return_numpy=False)
        origin_array = varbase.numpy()
        load_tensor_array = load_tensor.numpy()
        if paddle.base.core.is_compiled_with_cuda():
            base.core._cuda_synchronize(paddle.CUDAPlace(0))
        np.testing.assert_array_equal(origin_array, load_array)
        np.testing.assert_array_equal(origin_array, load_tensor_array)

class TestSaveLoadToMemory(unittest.TestCase):

    def test_dygraph_save_to_memory(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()
        linear = LinearNet()
        state_dict = linear.state_dict()
        byio = BytesIO()
        paddle.save(state_dict, byio)
        tensor = paddle.randn([2, 3], dtype='float32')
        paddle.save(tensor, byio)
        byio.seek(0)
        dict_load = paddle.load(byio, return_numpy=True)
        for (k, v) in state_dict.items():
            np.testing.assert_array_equal(v.numpy(), dict_load[k])
        tensor_load = paddle.load(byio, return_numpy=True)
        np.testing.assert_array_equal(tensor_load, tensor.numpy())
        with self.assertRaises(ValueError):
            paddle.save(4, 3)
        with self.assertRaises(ValueError):
            paddle.save(state_dict, '')
        with self.assertRaises(ValueError):
            paddle.framework.io_utils._open_file_buffer('temp', 'b')

    def test_static_save_to_memory(self):
        if False:
            return 10
        paddle.enable_static()
        with new_program_scope():
            x = paddle.static.data(name='x', shape=[None, IMAGE_SIZE], dtype='float32')
            z = paddle.static.nn.fc(x, 10, bias_attr=False)
            z = paddle.static.nn.fc(z, 128, bias_attr=False)
            loss = paddle.mean(z)
            place = base.CPUPlace() if not paddle.base.core.is_compiled_with_cuda() else base.CUDAPlace(0)
            prog = paddle.static.default_main_program()
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            state_dict = prog.state_dict()
            keys = list(state_dict.keys())
            tensor = state_dict[keys[0]]
            byio = BytesIO()
            byio2 = BytesIO()
            paddle.save(prog, byio2)
            paddle.save(tensor, byio)
            paddle.save(state_dict, byio)
            byio.seek(0)
            byio2.seek(0)
            prog_load = paddle.load(byio2)
            self.assertTrue(prog.desc.serialize_to_string() == prog_load.desc.serialize_to_string())
            tensor_load = paddle.load(byio, return_numpy=True)
            np.testing.assert_array_equal(tensor_load, np.array(tensor))
            state_dict_load = paddle.load(byio, return_numpy=True)
            for (k, v) in state_dict.items():
                np.testing.assert_array_equal(np.array(v), state_dict_load[k])

class TestSaveLoad(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.disable_static()
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.temp_dir.cleanup()

    def build_and_train_model(self):
        if False:
            while True:
                i = 10
        layer = LinearNet()
        loss_fn = nn.CrossEntropyLoss()
        adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())
        loader = random_batch_reader()
        train(layer, loader, loss_fn, adam)
        return (layer, adam)

    def check_load_state_dict(self, orig_dict, load_dict):
        if False:
            return 10
        for (var_name, value) in orig_dict.items():
            load_value = load_dict[var_name].numpy() if hasattr(load_dict[var_name], 'numpy') else np.array(load_dict[var_name])
            np.testing.assert_array_equal(value.numpy(), load_value)

    def test_save_load(self):
        if False:
            i = 10
            return i + 15
        (layer, opt) = self.build_and_train_model()
        layer_save_path = os.path.join(self.temp_dir.name, 'test_paddle_save_load.linear.pdparams')
        opt_save_path = os.path.join(self.temp_dir.name, 'test_paddle_save_load.linear.pdopt')
        layer_state_dict = layer.state_dict()
        opt_state_dict = opt.state_dict()
        paddle.save(layer_state_dict, layer_save_path)
        paddle.save(opt_state_dict, opt_save_path)
        load_layer_state_dict = paddle.load(layer_save_path)
        load_opt_state_dict = paddle.load(opt_save_path)
        self.check_load_state_dict(layer_state_dict, load_layer_state_dict)
        self.check_load_state_dict(opt_state_dict, load_opt_state_dict)
        paddle.enable_static()
        static_save_path = os.path.join(self.temp_dir.name, 'static_mode_test/test_paddle_save_load.linear.pdparams')
        paddle.save(layer_state_dict, static_save_path)
        load_static_state_dict = paddle.load(static_save_path)
        self.check_load_state_dict(layer_state_dict, load_static_state_dict)
        test_list = [1, 2, 3]
        with self.assertRaises(ValueError):
            paddle.save(layer_state_dict, os.path.join(self.temp_dir.name, 'test_paddle_save_load.linear.model/'))
        with self.assertRaises(ValueError):
            paddle.load(os.path.join(self.temp_dir.name, 'test_paddle_save_load.linear.params'))
        with self.assertRaises(ValueError):
            paddle.load(os.path.join(self.temp_dir.name, 'test_paddle_save_load.linear'))

class TestSaveLoadProgram(unittest.TestCase):

    def test_save_load_program(self):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        temp_dir = tempfile.TemporaryDirectory()
        with new_program_scope():
            layer = LinearNet()
            data = paddle.static.data(name='x_static_save', shape=(None, IMAGE_SIZE), dtype='float32')
            y_static = layer(data)
            main_program = paddle.static.default_main_program()
            startup_program = paddle.static.default_startup_program()
            origin_main = main_program.desc.serialize_to_string()
            origin_startup = startup_program.desc.serialize_to_string()
            path1 = os.path.join(temp_dir.name, 'test_paddle_save_load_program/main_program.pdmodel')
            path2 = os.path.join(temp_dir.name, 'test_paddle_save_load_program/startup_program.pdmodel')
            paddle.save(main_program, path1)
            paddle.save(startup_program, path2)
        with new_program_scope():
            load_main = paddle.load(path1).desc.serialize_to_string()
            load_startup = paddle.load(path2).desc.serialize_to_string()
            self.assertTrue(origin_main == load_main)
            self.assertTrue(origin_startup == load_startup)
        temp_dir.cleanup()

class TestSaveLoadLayer(unittest.TestCase):

    def test_save_load_layer(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.disable_static()
        temp_dir = tempfile.TemporaryDirectory()
        inps = paddle.randn([1, IMAGE_SIZE], dtype='float32')
        layer1 = LinearNet()
        layer2 = LinearNet()
        layer1.eval()
        layer2.eval()
        origin_layer = (layer1, layer2)
        origin = (layer1(inps), layer2(inps))
        path = os.path.join(temp_dir.name, 'test_save_load_layer_/layer.pdmodel')
        with self.assertRaises(ValueError):
            paddle.save(origin_layer, path)
        temp_dir.cleanup()
if __name__ == '__main__':
    unittest.main()