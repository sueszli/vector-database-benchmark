import os
import pickle
import shutil
import tempfile
import unittest
import numpy as np
import paddle
from paddle import base
from paddle.base import unique_name
from paddle.jit.api import to_static
from paddle.jit.translated_layer import INFER_PARAMS_INFO_SUFFIX
from paddle.nn import Linear
from paddle.static import InputSpec
BATCH_SIZE = 32
BATCH_NUM = 10
SEED = 10

def random_batch_reader(input_size, label_size):
    if False:
        for i in range(10):
            print('nop')

    def _get_random_inputs_and_labels(input_size, label_size):
        if False:
            while True:
                i = 10
        np.random.seed(SEED)
        input = np.random.random(size=input_size).astype('float32')
        label = np.random.random(size=label_size).astype('int64')
        return (input, label)

    def __reader__():
        if False:
            i = 10
            return i + 15
        for _ in range(BATCH_NUM):
            (batch_input, batch_label) = _get_random_inputs_and_labels([BATCH_SIZE, input_size], [BATCH_SIZE, label_size])
            yield (batch_input, batch_label)
    return __reader__

class LinearNet(paddle.nn.Layer):

    def __init__(self, in_size, out_size):
        if False:
            return 10
        super().__init__()
        self._linear = Linear(in_size, out_size)

    @to_static
    def forward(self, x):
        if False:
            print('Hello World!')
        return self._linear(x)

class LinearNetWithInputSpec(paddle.nn.Layer):

    def __init__(self, in_size, out_size):
        if False:
            print('Hello World!')
        super().__init__()
        self._linear = Linear(in_size, out_size)

    @to_static(input_spec=[InputSpec(shape=[None, 784], dtype='float32')])
    def forward(self, x):
        if False:
            i = 10
            return i + 15
        return self._linear(x)

class LinearNetNotDeclarative(paddle.nn.Layer):

    def __init__(self, in_size, out_size):
        if False:
            while True:
                i = 10
        super().__init__()
        self._linear = Linear(in_size, out_size)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        return self._linear(x)

class LinerNetWithLabel(paddle.nn.Layer):

    def __init__(self, in_size, out_size):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._linear = Linear(in_size, out_size)

    @to_static(input_spec=[InputSpec(shape=[None, 784], dtype='float32', name='image'), InputSpec(shape=[None, 1], dtype='int64', name='label')])
    def forward(self, x, label):
        if False:
            for i in range(10):
                print('nop')
        out = self._linear(x)
        loss = paddle.nn.functional.cross_entropy(out, label, reduction='none', use_softmax=False)
        avg_loss = paddle.mean(loss)
        return (out, avg_loss)

class LinerNetWithPruneInput(paddle.nn.Layer):

    def __init__(self, in_size, out_size):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._linear = Linear(in_size, out_size)

    @to_static(input_spec=[InputSpec(shape=[None, 784], dtype='float32', name='image'), InputSpec(shape=[None, 1], dtype='int64', name='label')])
    def forward(self, x, label):
        if False:
            return 10
        out = self._linear(x)
        loss = paddle.nn.functional.cross_entropy(out, label, reduction='none', use_softmax=False)
        avg_loss = paddle.mean(loss)
        return out

class LinerNetWithUselessInput(paddle.nn.Layer):

    def __init__(self, in_size, out_size):
        if False:
            return 10
        super().__init__()
        self._linear = Linear(in_size, out_size)

    @to_static(input_spec=[InputSpec(shape=[None, 784], dtype='float32', name='image'), InputSpec(shape=[None, 1], dtype='int64', name='label')])
    def forward(self, x, label):
        if False:
            for i in range(10):
                print('nop')
        out = self._linear(x)
        return out

class LinearNetReturnLoss(paddle.nn.Layer):

    def __init__(self, in_size, out_size):
        if False:
            print('Hello World!')
        super().__init__()
        self._linear = Linear(in_size, out_size)

    @to_static
    def forward(self, x):
        if False:
            i = 10
            return i + 15
        y = self._linear(x)
        z = self._linear(y)
        loss = paddle.mean(z)
        return (z, loss)

class LinearNetMultiInput(paddle.nn.Layer):

    def __init__(self, in_size, out_size):
        if False:
            while True:
                i = 10
        super().__init__()
        self._linear1 = Linear(in_size, out_size)
        self._linear2 = Linear(in_size, out_size)

    @to_static(input_spec=[InputSpec([None, 8], dtype='float32'), InputSpec([None, 8], dtype='float32')])
    def forward(self, x, y):
        if False:
            i = 10
            return i + 15
        x_out = self._linear1(x)
        y_out = self._linear2(y)
        loss = paddle.mean(x_out + y_out)
        return (x_out, y_out, loss)

class LinearNetMultiInput1(paddle.nn.Layer):

    def __init__(self, in_size, out_size):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._linear1 = Linear(in_size, out_size)
        self._linear2 = Linear(in_size, out_size)

    @to_static(input_spec=(InputSpec([None, 8], dtype='float32'), InputSpec([None, 8], dtype='float32')))
    def forward(self, x, y):
        if False:
            print('Hello World!')
        x_out = self._linear1(x)
        y_out = self._linear2(y)
        loss = paddle.mean(x_out + y_out)
        return (x_out, y_out, loss)

class MultiLoadingLinearNet(paddle.nn.Layer):

    def __init__(self, size, model_path):
        if False:
            print('Hello World!')
        super().__init__()
        self._linear = Linear(size, size)
        self._load_linear1 = paddle.jit.load(model_path)
        self._load_linear2 = paddle.jit.load(model_path)

    @to_static
    def forward(self, x):
        if False:
            print('Hello World!')
        tmp1 = self._linear(x)
        tmp2 = self._load_linear1(tmp1)
        tmp3 = self._load_linear2(tmp2)
        y = self._linear(tmp3)
        return y

class LinearNetReturnHidden(paddle.nn.Layer):

    def __init__(self, in_size, out_size):
        if False:
            while True:
                i = 10
        super().__init__()
        self._linear_1 = Linear(in_size, out_size)
        self._linear_2 = Linear(in_size, out_size)

    @to_static
    def forward(self, x):
        if False:
            while True:
                i = 10
        y = self._linear_1(x)
        z = self._linear_2(y)
        loss = paddle.mean(z)
        return (y, loss)

class LinearNetWithNestOut(paddle.nn.Layer):

    def __init__(self, in_size, out_size):
        if False:
            return 10
        super().__init__()
        self._linear_1 = Linear(in_size, out_size)
        self._linear_2 = Linear(in_size, out_size)

    @to_static
    def forward(self, x):
        if False:
            return 10
        y = self._linear_1(x)
        z = self._linear_2(y)
        out = y + z
        loss = paddle.mean(out)
        return (y, [(z, loss), out])

class LinearNetWithDictInput(paddle.nn.Layer):

    def __init__(self, in_size, out_size):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._linear = Linear(in_size, out_size)

    @paddle.jit.to_static(input_spec=[{'img': InputSpec(shape=[None, 8], dtype='float32', name='img')}, {'label': InputSpec(shape=[None, 1], dtype='int64', name='label')}])
    def forward(self, img, label):
        if False:
            i = 10
            return i + 15
        out = self._linear(img['img'])
        loss = paddle.nn.functional.cross_entropy(out, label['label'])
        return out

class LinearNetWithDictInputNoPrune(paddle.nn.Layer):

    def __init__(self, in_size, out_size):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._linear = Linear(in_size, out_size)

    def forward(self, img):
        if False:
            i = 10
            return i + 15
        out = self._linear(img['img'] + img['img2'])
        return out

class EmptyLayer(paddle.nn.Layer):

    def __init__(self):
        if False:
            return 10
        super().__init__()

    @paddle.jit.to_static
    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        return x

class NoParamLayer(paddle.nn.Layer):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()

    @paddle.jit.to_static
    def forward(self, x, y):
        if False:
            return 10
        return x + y

class LinearNetWithMultiStaticFunc(paddle.nn.Layer):

    def __init__(self, in_size, out_size):
        if False:
            print('Hello World!')
        super().__init__()
        self._linear_0 = Linear(in_size, out_size)
        self._linear_1 = Linear(in_size, out_size)
        self._scale = paddle.to_tensor([9.9])

    @paddle.jit.to_static
    def forward(self, x):
        if False:
            print('Hello World!')
        return self._linear_0(x)

    @paddle.jit.to_static
    def forward_no_param(self, x):
        if False:
            print('Hello World!')
        return x

    @paddle.jit.to_static
    def forward_general(self, x):
        if False:
            i = 10
            return i + 15
        return self._linear_0(x) + self._linear_1(x) * self._scale

def train(layer, input_size=784, label_size=1):
    if False:
        print('Hello World!')
    sgd = paddle.optimizer.SGD(learning_rate=0.01, parameters=layer.parameters())
    train_loader = base.io.DataLoader.from_generator(capacity=5)
    train_loader.set_batch_generator(random_batch_reader(input_size, label_size))
    for data in train_loader():
        (img, label) = data
        label.stop_gradient = True
        cost = layer(img)
        loss = paddle.nn.functional.cross_entropy(cost, label, reduction='none', use_softmax=True)
        avg_loss = paddle.mean(loss)
        avg_loss.backward()
        sgd.minimize(avg_loss)
        layer.clear_gradients()
    return ([img], layer, avg_loss)

def train_with_label(layer, input_size=784, label_size=1):
    if False:
        return 10
    sgd = paddle.optimizer.SGD(learning_rate=0.01, parameters=layer.parameters())
    train_loader = base.io.DataLoader.from_generator(capacity=5)
    train_loader.set_batch_generator(random_batch_reader(input_size, label_size))
    for data in train_loader():
        (img, label) = data
        label.stop_gradient = True
        (out, avg_loss) = layer(img, label)
        avg_loss.backward()
        sgd.minimize(avg_loss)
        layer.clear_gradients()
    return out

class TestJitSaveLoad(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, 'test_jit_save_load/model')
        base.enable_dygraph()
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

    def tearDown(self):
        if False:
            return 10
        self.temp_dir.cleanup()

    def train_and_save_model(self, model_path=None):
        if False:
            i = 10
            return i + 15
        layer = LinearNet(784, 1)
        (example_inputs, layer, _) = train(layer)
        final_model_path = model_path if model_path else self.model_path
        orig_input_types = [type(x) for x in example_inputs]
        paddle.jit.save(layer=layer, path=final_model_path, input_spec=example_inputs)
        new_input_types = [type(x) for x in example_inputs]
        self.assertEqual(orig_input_types, new_input_types)
        return layer

    def test_save_load(self):
        if False:
            return 10
        train_layer = self.train_and_save_model()
        loaded_layer = paddle.jit.load(self.model_path)
        self.load_and_inference(train_layer, loaded_layer)
        self.load_dygraph_state_dict(train_layer)
        self.load_and_finetune(train_layer, loaded_layer)

    def load_and_inference(self, train_layer, infer_layer):
        if False:
            i = 10
            return i + 15
        train_layer.eval()
        infer_layer.eval()
        x = base.dygraph.to_variable(np.random.random((1, 784)).astype('float32'))
        np.testing.assert_array_equal(train_layer(x).numpy(), infer_layer(x).numpy())

    def load_and_finetune(self, train_layer, load_train_layer):
        if False:
            for i in range(10):
                print('nop')
        train_layer.train()
        load_train_layer.train()
        (img0, _, train_loss) = train(train_layer)
        (img1, _, load_train_loss) = train(load_train_layer)
        np.testing.assert_array_equal(train_loss.numpy(), load_train_loss.numpy())

    def load_dygraph_state_dict(self, train_layer):
        if False:
            for i in range(10):
                print('nop')
        train_layer.eval()
        new_layer = LinearNet(784, 1)
        orig_state_dict = new_layer.state_dict()
        load_state_dict = paddle.load(self.model_path)
        for structured_name in orig_state_dict:
            self.assertTrue(structured_name in load_state_dict)
        new_layer.set_state_dict(load_state_dict)
        new_layer.eval()
        x = base.dygraph.to_variable(np.random.random((1, 784)).astype('float32'))
        np.testing.assert_array_equal(train_layer(x).numpy(), new_layer(x).numpy())

    def test_load_dygraph_no_path(self):
        if False:
            while True:
                i = 10
        model_path = os.path.join(self.temp_dir.name, 'test_jit_save_load.no_path/model_path')
        with self.assertRaises(ValueError):
            model_dict = paddle.load(model_path)

    def test_jit_load_no_path(self):
        if False:
            i = 10
            return i + 15
        path = os.path.join(self.temp_dir.name, 'test_jit_save_load.no_path/model_path')
        with self.assertRaises(ValueError):
            loaded_layer = paddle.jit.load(path)

class TestSaveLoadWithNestOut(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        base.enable_dygraph()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            return 10
        self.temp_dir.cleanup()

    def test_nest_output(self):
        if False:
            while True:
                i = 10
        x = base.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
        net = LinearNetWithNestOut(8, 8)
        dy_outs = paddle.utils.flatten(net(x))
        net = to_static(net, input_spec=[InputSpec([None, 8], name='x')])
        model_path = os.path.join(self.temp_dir.name, 'net_with_nest_out/model')
        paddle.jit.save(net, model_path)
        load_net = paddle.jit.load(model_path)
        load_outs = paddle.utils.flatten(load_net(x))
        self.assertTrue(len(dy_outs) == 4)
        for (dy_out, load_out) in zip(dy_outs, load_outs):
            np.testing.assert_allclose(dy_out.numpy(), load_out.numpy(), rtol=1e-05)

class TestSaveLoadWithDictInput(unittest.TestCase):

    def test_dict_input(self):
        if False:
            while True:
                i = 10
        net = LinearNetWithDictInput(8, 8)
        self.assertEqual(len(net.forward.concrete_program.inputs), 3)
        temp_dir = tempfile.TemporaryDirectory()
        path = os.path.join(temp_dir.name, 'test_jit_save_load_with_dict_input/model')
        paddle.jit.save(layer=net, path=path, input_spec=[{'img': InputSpec(shape=[None, 8], dtype='float32', name='img')}])
        img = paddle.randn(shape=[4, 8], dtype='float32')
        loaded_net = paddle.jit.load(path)
        loaded_out = loaded_net(img)
        self.assertEqual(len(loaded_net._input_spec()), 1)
        temp_dir.cleanup()

class TestSaveLoadWithDictInputNoPrune(unittest.TestCase):

    def test_dict_input(self):
        if False:
            while True:
                i = 10
        net = LinearNetWithDictInputNoPrune(8, 8)
        temp_dir = tempfile.TemporaryDirectory()
        path = os.path.join(temp_dir.name, 'test_jit_save_load_with_dict_input_no_prune/model')
        paddle.jit.save(layer=net, path=path, input_spec=[{'img': InputSpec(shape=[None, 8], dtype='float32', name='img'), 'img2': InputSpec(shape=[None, 8], dtype='float32', name='img2')}])
        img = paddle.randn(shape=[4, 8], dtype='float32')
        img2 = paddle.randn(shape=[4, 8], dtype='float32')
        loaded_net = paddle.jit.load(path)
        loaded_out = loaded_net(img, img2)
        self.assertEqual(len(loaded_net._input_spec()), 2)
        temp_dir.cleanup()

class TestSaveLoadWithInputSpec(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        base.enable_dygraph()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.temp_dir.cleanup()

    def test_with_input_spec(self):
        if False:
            return 10
        net = LinearNetReturnLoss(8, 8)
        net.forward = to_static(net.forward, input_spec=[InputSpec([None, 8], name='x')])
        model_path = os.path.join(self.temp_dir.name, 'input_spec.output_spec/model')
        self.assertTrue(len(net.forward.inputs) == 1)
        input_x = net.forward.inputs[0]
        self.assertTrue(input_x.shape == (-1, 8))
        self.assertTrue(input_x.name == 'x')
        output_spec = net.forward.outputs[:1]
        paddle.jit.save(net, model_path, output_spec=output_spec)
        infer_layer = paddle.jit.load(model_path)
        x = base.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
        pred = infer_layer(x)

    def test_multi_in_out(self):
        if False:
            print('Hello World!')
        net = LinearNetMultiInput(8, 8)
        model_path = os.path.join(self.temp_dir.name, 'multi_inout.output_spec1/model')
        self.assertTrue(len(net.forward.inputs) == 2)
        input_x = net.forward.inputs[0]
        input_y = net.forward.inputs[1]
        self.assertTrue(input_x.shape == (-1, 8))
        self.assertTrue(input_y.shape == (-1, 8))
        output_spec = net.forward.outputs[:2]
        paddle.jit.save(net, model_path, output_spec=output_spec)
        infer_layer = paddle.jit.load(model_path)
        x = base.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
        y = base.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
        (pred_x, pred_y) = infer_layer(x, y)
        model_path = os.path.join(self.temp_dir.name, 'multi_inout.output_spec2/model')
        output_spec = net.forward.outputs[:1]
        paddle.jit.save(net, model_path, [input_x], output_spec=output_spec)
        infer_layer2 = paddle.jit.load(model_path)
        pred_xx = infer_layer2(x)
        np.testing.assert_allclose(pred_x.numpy(), pred_xx.numpy(), rtol=1e-05)

    def test_multi_in_out1(self):
        if False:
            return 10
        net = LinearNetMultiInput1(8, 8)
        model_path = os.path.join(self.temp_dir.name, 'multi_inout1.output_spec1/model')
        self.assertTrue(len(net.forward.inputs) == 2)
        input_x = net.forward.inputs[0]
        input_y = net.forward.inputs[1]
        self.assertTrue(input_x.shape == (-1, 8))
        self.assertTrue(input_y.shape == (-1, 8))
        output_spec = net.forward.outputs[:2]
        paddle.jit.save(net, model_path, output_spec=output_spec)
        infer_layer = paddle.jit.load(model_path)
        x = base.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
        y = base.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
        (pred_x, pred_y) = infer_layer(x, y)
        model_path = os.path.join(self.temp_dir.name, 'multi_inout1.output_spec2/model')
        output_spec = net.forward.outputs[:1]
        paddle.jit.save(net, model_path, net.forward.inputs, output_spec=output_spec, input_names_after_prune=[input_x.name])
        infer_layer2 = paddle.jit.load(model_path)
        pred_xx = infer_layer2(x)
        np.testing.assert_allclose(pred_x.numpy(), pred_xx.numpy(), rtol=1e-05)

class TestJitSaveLoadConfig(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        base.enable_dygraph()
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.temp_dir.cleanup()

    def test_output_spec(self):
        if False:
            for i in range(10):
                print('nop')
        train_layer = LinearNetReturnLoss(8, 8)
        adam = paddle.optimizer.Adam(learning_rate=0.1, parameters=train_layer.parameters())
        x = base.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
        for i in range(10):
            (out, loss) = train_layer(x)
            loss.backward()
            adam.minimize(loss)
            train_layer.clear_gradients()
        model_path = os.path.join(self.temp_dir.name, 'save_load_config.output_spec')
        output_spec = [out]
        paddle.jit.save(layer=train_layer, path=model_path, input_spec=[x], output_spec=output_spec)
        train_layer.eval()
        infer_layer = paddle.jit.load(model_path)
        x = base.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
        np.testing.assert_array_equal(train_layer(x)[0].numpy(), infer_layer(x).numpy())

    def test_save_no_support_config_error(self):
        if False:
            return 10
        layer = LinearNet(784, 1)
        path = os.path.join(self.temp_dir.name, 'no_support_config_test')
        with self.assertRaises(ValueError):
            paddle.jit.save(layer=layer, path=path, model_filename='')

    def test_load_empty_model_filename_error(self):
        if False:
            for i in range(10):
                print('nop')
        path = os.path.join(self.temp_dir.name, 'error_model_filename_test')
        with self.assertRaises(ValueError):
            paddle.jit.load(path, model_filename='')

    def test_load_empty_params_filename_error(self):
        if False:
            while True:
                i = 10
        path = os.path.join(self.temp_dir.name, 'error_params_filename_test')
        with self.assertRaises(ValueError):
            paddle.jit.load(path, params_filename='')

    def test_load_with_no_support_config(self):
        if False:
            while True:
                i = 10
        path = os.path.join(self.temp_dir.name, 'no_support_config_test')
        with self.assertRaises(ValueError):
            paddle.jit.load(path, separate_params=True)

class TestJitMultipleLoading(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.linear_size = 4
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, 'jit_multi_load/model')
        base.enable_dygraph()
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)
        self.train_and_save_orig_model()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.temp_dir.cleanup()

    def train_and_save_orig_model(self):
        if False:
            for i in range(10):
                print('nop')
        layer = LinearNet(self.linear_size, self.linear_size)
        (example_inputs, layer, _) = train(layer, self.linear_size, 1)
        paddle.jit.save(layer=layer, path=self.model_path, input_spec=example_inputs)

    def test_load_model_retransform_inference(self):
        if False:
            for i in range(10):
                print('nop')
        multi_loaded_layer = MultiLoadingLinearNet(self.linear_size, self.model_path)
        state_dict = multi_loaded_layer.state_dict()
        name_set = set()
        for (_, var) in state_dict.items():
            self.assertTrue(var.name not in name_set)
            name_set.add(var.name)

class TestJitPruneModelAndLoad(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.linear_size = 4
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, 'jit_prune_model_and_load/model')
        base.enable_dygraph()
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.temp_dir.cleanup()

    def train_and_save(self):
        if False:
            for i in range(10):
                print('nop')
        train_layer = LinearNetReturnHidden(8, 8)
        adam = paddle.optimizer.Adam(learning_rate=0.1, parameters=train_layer.parameters())
        x = base.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
        for i in range(10):
            (hidden, loss) = train_layer(x)
            loss.backward()
            adam.minimize(loss)
            train_layer.clear_gradients()
        output_spec = [hidden]
        paddle.jit.save(layer=train_layer, path=self.model_path, input_spec=[x], output_spec=output_spec)
        return train_layer

    def test_load_pruned_model(self):
        if False:
            return 10
        train_layer = self.train_and_save()
        train_layer.eval()
        infer_layer = paddle.jit.load(self.model_path)
        x = base.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
        np.testing.assert_array_equal(train_layer(x)[0].numpy(), infer_layer(x).numpy())

    def test_load_var_not_in_extra_var_info(self):
        if False:
            print('Hello World!')
        self.train_and_save()
        var_info_path = self.model_path + INFER_PARAMS_INFO_SUFFIX
        with open(var_info_path, 'rb') as f:
            extra_var_info = pickle.load(f)
            extra_var_info.clear()
        with open(var_info_path, 'wb') as f:
            pickle.dump(extra_var_info, f, protocol=2)
        with self.assertRaises(RuntimeError):
            paddle.jit.load(self.model_path)

class TestJitSaveMultiCases(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        base.enable_dygraph()
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.temp_dir.cleanup()

    def verify_inference_correctness(self, layer, model_path, with_label_and_loss=False, with_label=False):
        if False:
            for i in range(10):
                print('nop')
        layer.eval()
        loaded_layer = paddle.jit.load(model_path)
        loaded_layer.eval()
        x = paddle.to_tensor(np.random.random((1, 784)).astype('float32'))
        if with_label_and_loss:
            y = paddle.to_tensor(np.random.random((1, 1)).astype('int64'))
            (pred, _) = layer(x, y)
            pred = pred.numpy()
        elif with_label:
            y = paddle.to_tensor(np.random.random((1, 1)).astype('int64'))
            pred = layer(x, y)
            pred = pred.numpy()
        else:
            pred = layer(x).numpy()
        loaded_pred = loaded_layer(x).numpy()
        np.testing.assert_array_equal(pred, loaded_pred, err_msg='Result diff when load and inference:\nlayer result:\n{}\nloaded layer result:\n{}'.format(pred, loaded_pred))

    def test_no_prune_to_static_after_train(self):
        if False:
            print('Hello World!')
        layer = LinearNet(784, 1)
        train(layer)
        model_path = os.path.join(self.temp_dir.name, 'test_no_prune_to_static_after_train/model')
        paddle.jit.save(layer, model_path)
        self.verify_inference_correctness(layer, model_path)

    def test_no_prune_to_static_no_train(self):
        if False:
            i = 10
            return i + 15
        layer = LinearNetWithInputSpec(784, 1)
        model_path = os.path.join(self.temp_dir.name, 'test_no_prune_to_static_no_train/model')
        paddle.jit.save(layer, model_path)
        self.verify_inference_correctness(layer, model_path)

    def test_no_prune_no_to_static_after_train(self):
        if False:
            return 10
        layer = LinearNetNotDeclarative(784, 1)
        train(layer)
        model_path = os.path.join(self.temp_dir.name, 'test_no_prune_no_to_static_after_train/model')
        paddle.jit.save(layer, model_path, input_spec=[InputSpec(shape=[None, 784], dtype='float32')])
        self.verify_inference_correctness(layer, model_path)

    def test_no_prune_no_to_static_after_train_with_examples(self):
        if False:
            i = 10
            return i + 15
        layer = LinearNetNotDeclarative(784, 1)
        (example_inputs, _, _) = train(layer)
        model_path = os.path.join(self.temp_dir.name, 'test_no_prune_no_to_static_after_train_with_examples/model')
        paddle.jit.save(layer=layer, path=model_path, input_spec=example_inputs)
        self.verify_inference_correctness(layer, model_path)

    def test_no_prune_no_to_static_no_train(self):
        if False:
            i = 10
            return i + 15
        layer = LinearNetNotDeclarative(784, 1)
        model_path = os.path.join(self.temp_dir.name, 'test_no_prune_no_to_static_no_train/model')
        paddle.jit.save(layer, model_path, input_spec=[InputSpec(shape=[None, 784], dtype='float32')])
        self.verify_inference_correctness(layer, model_path)

    def test_prune_to_static_after_train(self):
        if False:
            return 10
        layer = LinerNetWithLabel(784, 1)
        out = train_with_label(layer)
        model_path = os.path.join(self.temp_dir.name, 'test_prune_to_static_after_train/model')
        paddle.jit.save(layer, model_path, input_spec=[InputSpec(shape=[None, 784], dtype='float32', name='image'), True], output_spec=[out], input_names_after_prune=['image'])
        self.verify_inference_correctness(layer, model_path, with_label_and_loss=True)

    def test_prune_to_static_no_train(self):
        if False:
            print('Hello World!')
        layer = LinerNetWithLabel(784, 1)
        model_path = os.path.join(self.temp_dir.name, 'test_prune_to_static_no_train/model')
        output_spec = layer.forward.outputs[:1]
        paddle.jit.save(layer, model_path, input_spec=[InputSpec(shape=[None, 784], dtype='float32', name='image'), True], output_spec=output_spec, input_names_after_prune=['image'])
        self.verify_inference_correctness(layer, model_path, with_label_and_loss=True)

    def test_prune_input_to_static_no_train(self):
        if False:
            while True:
                i = 10
        layer = LinerNetWithPruneInput(784, 1)
        model_path = os.path.join(self.temp_dir.name, 'test_prune_input_to_static_no_train/model')
        paddle.jit.save(layer, model_path, input_spec=[InputSpec(shape=[None, 784], dtype='float32', name='image')])
        self.verify_inference_correctness(layer, model_path, with_label=True)

    def test_prune_useless_input_to_static_no_train(self):
        if False:
            while True:
                i = 10
        layer = LinerNetWithUselessInput(784, 1)
        model_path = os.path.join(self.temp_dir.name, 'test_prune_useless_input_to_static_no_train/model')
        paddle.jit.save(layer, model_path, input_spec=[InputSpec(shape=[None, 784], dtype='float32', name='image')])
        self.verify_inference_correctness(layer, model_path, with_label=True)

    def test_no_prune_input_spec_name_warning(self):
        if False:
            print('Hello World!')
        layer = LinearNetWithInputSpec(784, 1)
        train(layer)
        model_path = os.path.join(self.temp_dir.name, 'test_no_prune_input_spec_name_warning/model')
        paddle.jit.save(layer, model_path, input_spec=[InputSpec(shape=[None, 784], dtype='float32')])
        paddle.jit.save(layer, model_path, input_spec=[InputSpec(shape=[None, 784], dtype='float32', name='feed_input')])
        self.verify_inference_correctness(layer, model_path)

    def test_not_prune_output_spec_name_warning(self):
        if False:
            for i in range(10):
                print('nop')
        layer = LinearNet(784, 1)
        train(layer)
        model_path = os.path.join(self.temp_dir.name, 'test_not_prune_output_spec_name_warning/model')
        out = paddle.to_tensor(np.random.random((1, 1)).astype('float'))
        paddle.jit.save(layer, model_path, output_spec=[out])
        self.verify_inference_correctness(layer, model_path)

    def test_prune_input_spec_name_error(self):
        if False:
            for i in range(10):
                print('nop')
        layer = LinerNetWithLabel(784, 1)
        model_path = os.path.join(self.temp_dir.name, 'test_prune_input_spec_name_error/model')
        with self.assertRaises(ValueError):
            paddle.jit.save(layer, model_path, input_spec=[InputSpec(shape=[None, 784], dtype='float32')])
        with self.assertRaises(ValueError):
            paddle.jit.save(layer, model_path, input_spec=[InputSpec(shape=[None, 784], dtype='float32', name='feed_input')])

    def test_prune_output_spec_name_error(self):
        if False:
            return 10
        layer = LinerNetWithLabel(784, 1)
        train_with_label(layer)
        model_path = os.path.join(self.temp_dir.name, 'test_prune_to_static_after_train/model')
        out = paddle.to_tensor(np.random.random((1, 1)).astype('float'))
        with self.assertRaises(ValueError):
            paddle.jit.save(layer, model_path, input_spec=[InputSpec(shape=[None, 784], dtype='float32', name='image'), True], output_spec=[out], input_names_after_prune=['image'])

class TestJitSaveLoadEmptyLayer(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, 'jit_save_load_empty_layer/model')
        paddle.disable_static()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.temp_dir.cleanup()

    def test_save_load_empty_layer(self):
        if False:
            for i in range(10):
                print('nop')
        layer = EmptyLayer()
        x = paddle.to_tensor(np.random.random(10).astype('float32'))
        out = layer(x)
        paddle.jit.save(layer, self.model_path)
        load_layer = paddle.jit.load(self.model_path)
        load_out = load_layer(x)
        np.testing.assert_array_equal(out, load_out)

class TestJitSaveLoadNoParamLayer(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, 'jit_save_load_no_param_layer/model')
        paddle.disable_static()

    def tearDown(self):
        if False:
            return 10
        self.temp_dir.cleanup()

    def test_save_load_no_param_layer(self):
        if False:
            while True:
                i = 10
        layer = NoParamLayer()
        x = paddle.to_tensor(np.random.random(5).astype('float32'))
        y = paddle.to_tensor(np.random.random(5).astype('float32'))
        out = layer(x, y)
        paddle.jit.save(layer, self.model_path)
        load_layer = paddle.jit.load(self.model_path)
        load_out = load_layer(x, y)
        np.testing.assert_array_equal(out, load_out)

class TestJitSaveLoadMultiMethods(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        paddle.disable_static()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            return 10
        self.temp_dir.cleanup()

    def test_jit_save_load_inference(self):
        if False:
            print('Hello World!')
        model_path_inference = os.path.join(self.temp_dir.name, 'jit_save_load_multi_methods/model')
        IMAGE_SIZE = 224
        layer = LinearNetWithMultiStaticFunc(IMAGE_SIZE, 10)
        inps = paddle.randn([1, IMAGE_SIZE])
        result_origin = {}
        for func in dir(layer):
            if func.startswith('forward'):
                result_origin[func] = getattr(layer, func, None)(inps)
        paddle.jit.save(layer, model_path_inference)
        load_net = paddle.jit.load(model_path_inference)
        for (func, result) in result_origin.items():
            self.assertTrue(float((result - getattr(load_net, func, None)(inps)).abs().max()) < 1e-05)

    def test_jit_save_load_multi_methods_inputspec(self):
        if False:
            i = 10
            return i + 15
        model_path = os.path.join(self.temp_dir.name, 'jit_save_load_multi_methods/model')
        layer = LinearNetWithMultiStaticFunc(784, 1)
        with self.assertRaises(ValueError):
            paddle.jit.save(layer, model_path, input_spec=[InputSpec(shape=[None, 784])])

    def test_parse_name(self):
        if False:
            i = 10
            return i + 15
        model_path_inference = os.path.join(self.temp_dir.name, 'jit_save_load_parse_name/model')
        IMAGE_SIZE = 224
        layer = LinearNet(IMAGE_SIZE, 1)
        inps = paddle.randn([1, IMAGE_SIZE])
        layer(inps)
        paddle.jit.save(layer, model_path_inference)
        paddle.jit.save(layer, model_path_inference + '_v2')
        load_net = paddle.jit.load(model_path_inference)
        self.assertFalse(hasattr(load_net, 'v2'))

class LayerSaved(paddle.nn.Layer):

    def __init__(self, in_size, out_size):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.hidden = 100
        self._linear_0 = Linear(in_size, self.hidden)
        self._linear_1_0 = Linear(self.hidden, self.hidden)
        self._linear_1_1 = Linear(self.hidden, self.hidden)
        self._linear_2 = Linear(self.hidden, out_size)
        self._scale = paddle.to_tensor([9.9])

    @paddle.jit.to_static
    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        y = self._linear_0(x)
        if paddle.shape(x)[0] == 1:
            y = self._linear_1_0(y)
        else:
            y += self._linear_1_1(y + self._scale)
        return self._linear_2(y)

class Net(paddle.nn.Layer):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.fc1 = paddle.nn.Linear(4, 4)
        self.fc2 = paddle.nn.Linear(4, 4)
        self.bias = 0.4
        self.flag = paddle.ones([2], dtype='int32')

    @paddle.jit.to_static(input_spec=[InputSpec([None, 4], dtype='float32')])
    def log_softmax(self, input):
        if False:
            while True:
                i = 10
        return paddle.nn.functional.log_softmax(input, axis=-1)

    @paddle.jit.to_static(input_spec=[InputSpec([None, 4], dtype='float32')])
    def forward(self, x):
        if False:
            print('Hello World!')
        out = self.fc1(x)
        out = paddle.nn.functional.relu(out)
        out = paddle.mean(out)
        return out

    @paddle.jit.to_static(input_spec=[InputSpec([None, 4], dtype='float32')])
    def infer(self, input):
        if False:
            print('Hello World!')
        out = self.fc2(input)
        out = out + self.bias
        out = paddle.mean(out)
        return out

    @paddle.jit.to_static(property=True)
    def fbias(self):
        if False:
            for i in range(10):
                print('nop')
        return self.bias + 1

    @paddle.jit.to_static(property=True)
    def down_sampling(self):
        if False:
            for i in range(10):
                print('nop')
        return 4

    @paddle.jit.to_static(property=True)
    def fstr(self):
        if False:
            print('Hello World!')
        return 'save str property'

    @paddle.jit.to_static(property=True)
    def ints(self):
        if False:
            for i in range(10):
                print('nop')
        return [10, 20]

    @paddle.jit.to_static(property=True)
    def floats(self):
        if False:
            while True:
                i = 10
        return [1.1, 2.2]

    @paddle.jit.to_static(property=True)
    def strs(self):
        if False:
            while True:
                i = 10
        return ['hello', 'world']

class NetTensor(paddle.nn.Layer):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.fc1 = paddle.nn.Linear(4, 4)
        self.fc2 = paddle.nn.Linear(4, 4)
        self.bias = 0.4
        self.flag = paddle.ones([2], dtype='int32')

    @paddle.jit.to_static(input_spec=[InputSpec([None, 4], dtype='float32')])
    def forward(self, x):
        if False:
            while True:
                i = 10
        out = self.fc1(x)
        out = paddle.nn.functional.relu(out)
        out = paddle.mean(out)
        return out

    @paddle.jit.to_static(property=True)
    def fflag(self):
        if False:
            return 10
        return True

class TestJitSaveCombineProperty(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.disable_static()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.temp_dir.cleanup()

    def test_jit_save_combine_property(self):
        if False:
            for i in range(10):
                print('nop')
        model_path = os.path.join(self.temp_dir.name, 'test_jit_save_combine/model')
        with unique_name.guard():
            net = Net()
        paddle.jit.save(net, model_path, combine_params=True)

    def test_jit_save_tensor_property(self):
        if False:
            while True:
                i = 10
        model_path = os.path.join(self.temp_dir.name, 'test_jit_save_combine/model')
        with unique_name.guard():
            net = NetTensor()
        paddle.jit.save(net, model_path, combine_params=True)

class LayerLoadFinetune(paddle.nn.Layer):

    def __init__(self, in_size, out_size, load_path):
        if False:
            while True:
                i = 10
        super().__init__()
        self._linear_0 = Linear(in_size, in_size)
        self._linear_1_0 = Linear(out_size, in_size)
        self._linear_1_1 = Linear(out_size, in_size)
        self._linear_2 = Linear(out_size, out_size)
        self._scale = paddle.to_tensor([9.9])
        self._load_l1 = paddle.jit.load(load_path)
        self._load_l2 = paddle.jit.load(load_path)

    @paddle.jit.to_static
    def forward(self, x):
        if False:
            while True:
                i = 10
        y = self._linear_0(x)
        y = self._load_l1(y)
        if paddle.shape(x)[0] == 1:
            y = self._linear_1_0(y)
            y = self._load_l1(y)
        else:
            y += self._linear_1_1(x + self._scale)
            y = self._load_l2(y)
        y = self._linear_1_0(y)
        y = self._load_l1(y)
        y = self._linear_1_0(y)
        y = self._load_l1(y)
        return y

class TestJitSaveLoadSaveWithoutRunning(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        paddle.disable_static()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            return 10
        self.temp_dir.cleanup()

    def test_save_load_finetune_load(self):
        if False:
            while True:
                i = 10
        model_path = os.path.join(self.temp_dir.name, 'test_jit_save_load_save_without_running/model')
        IMAGE_SIZE = 224
        inps0 = paddle.randn([1, IMAGE_SIZE])
        inps1 = paddle.randn([2, IMAGE_SIZE])
        with unique_name.guard():
            layer_save = LayerSaved(IMAGE_SIZE, IMAGE_SIZE)
        paddle.jit.save(layer_save, model_path, input_spec=[paddle.static.InputSpec(shape=[None, IMAGE_SIZE], dtype='float32')])
        result_00 = layer_save(inps0)
        result_01 = layer_save(inps1)
        with unique_name.guard():
            layer_load = paddle.jit.load(model_path)
            paddle.jit.save(layer_load, model_path, input_spec=[paddle.static.InputSpec(shape=[None, IMAGE_SIZE], dtype='float32')])
        layer_reload = paddle.jit.load(model_path)
        result_10 = layer_reload(inps0)
        result_11 = layer_reload(inps1)
        self.assertTrue(float((result_00 - result_10).abs().max()) < 1e-05)
        self.assertTrue(float((result_01 - result_11).abs().max()) < 1e-05)

class TestJitSaveLoadFinetuneLoad(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            return 10
        self.temp_dir.cleanup()

    def test_save_load_finetune_load(self):
        if False:
            print('Hello World!')
        model_path = os.path.join(self.temp_dir.name, 'test_jit_save_load_finetune_load/model')
        IMAGE_SIZE = 224
        inps0 = paddle.randn([1, IMAGE_SIZE])
        inps1 = paddle.randn([2, IMAGE_SIZE])
        with unique_name.guard():
            layer_save = LayerSaved(IMAGE_SIZE, IMAGE_SIZE)
        layer_save(inps0)
        paddle.jit.save(layer_save, model_path)
        with unique_name.guard():
            layer_load = LayerLoadFinetune(IMAGE_SIZE, IMAGE_SIZE, model_path)
        train(layer_load, input_size=IMAGE_SIZE)
        result_00 = layer_load(inps0)
        result_01 = layer_load(inps1)
        paddle.jit.save(layer_load, model_path)
        layer_finetune = paddle.jit.load(model_path)
        result_10 = layer_finetune(inps0)
        result_11 = layer_finetune(inps1)
        self.assertTrue(float((result_00 - result_10).abs().max()) < 1e-05)
        self.assertTrue(float((result_01 - result_11).abs().max()) < 1e-05)

class TestJitSaveLoadFunctionCase1(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        paddle.disable_static()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.temp_dir.cleanup()

    def test_jit_save_load_static_function(self):
        if False:
            i = 10
            return i + 15

        @paddle.jit.to_static
        def fun(inputs):
            if False:
                for i in range(10):
                    print('nop')
            return paddle.tanh(inputs)
        path = os.path.join(self.temp_dir.name, 'test_jit_save_load_function_1/func')
        inps = paddle.rand([3, 6])
        origin = fun(inps)
        paddle.jit.save(fun, path)
        load_func = paddle.jit.load(path)
        load_result = load_func(inps)
        self.assertTrue((load_result - origin).abs().max() < 1e-10)

class TestJitSaveLoadFunctionCase2(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        paddle.disable_static()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.temp_dir.cleanup()

    def test_jit_save_load_function_input_spec(self):
        if False:
            print('Hello World!')

        @paddle.jit.to_static(input_spec=[InputSpec(shape=[None, 6], dtype='float32', name='x')])
        def fun(inputs):
            if False:
                print('Hello World!')
            return paddle.nn.functional.relu(inputs)
        path = os.path.join(self.temp_dir.name, 'test_jit_save_load_function_2/func')
        inps = paddle.rand([3, 6])
        origin = fun(inps)
        paddle.jit.save(fun, path)
        load_func = paddle.jit.load(path)
        load_result = load_func(inps)
        self.assertTrue((load_result - origin).abs().max() < 1e-10)

class TestJitSaveLoadFunctionCase3(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        paddle.disable_static()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.temp_dir.cleanup()

    def test_jit_save_load_function_function(self):
        if False:
            while True:
                i = 10

        def fun(inputs):
            if False:
                return 10
            return paddle.tanh(inputs)
        path = os.path.join(self.temp_dir.name, 'test_jit_save_load_function_3/func')
        inps = paddle.rand([3, 6])
        origin = fun(inps)
        paddle.jit.save(fun, path, input_spec=[InputSpec(shape=[None, 6], dtype='float32', name='x')])
        load_func = paddle.jit.load(path)
        load_result = load_func(inps)
        self.assertTrue((load_result - origin).abs().max() < 1e-10)

class TestJitSaveLoadFunctionWithParamCase1(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.temp_dir.cleanup()

    def test_jit_save_load_function(self):
        if False:
            for i in range(10):
                print('nop')

        class LinearNet(paddle.nn.Layer):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self._linear = paddle.nn.Linear(5, 6)

            def forward(self, x):
                if False:
                    print('Hello World!')
                return paddle.tanh(x)

            def anothor_forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self._linear(x)
        layer = LinearNet()
        inps = paddle.rand([3, 5])
        origin = layer.anothor_forward(inps)
        func = paddle.jit.to_static(layer.anothor_forward, [paddle.static.InputSpec(shape=[-1, 5])])
        path = os.path.join(self.temp_dir.name, 'test_jit_save_load_function_with_params_case1/func')
        paddle.jit.save(func, path)
        load_func = paddle.jit.load(path)
        load_result = load_func(inps)
        np.testing.assert_array_equal(load_result.numpy(), origin.numpy())

class TestJitSaveLoadFunctionWithParamCase2(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.temp_dir.cleanup()

    def test_jit_save_load_function(self):
        if False:
            while True:
                i = 10

        class LinearNet(paddle.nn.Layer):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self._linear = paddle.nn.Linear(5, 6)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return paddle.tanh(x)

            @paddle.jit.to_static(input_spec=[InputSpec(shape=[-1, 5])])
            def anothor_forward(self, x):
                if False:
                    print('Hello World!')
                return self._linear(x)
        layer = LinearNet()
        inps = paddle.rand([3, 5])
        path = os.path.join(self.temp_dir.name, 'test_jit_save_load_function_with_params_case2/func')
        paddle.jit.save(layer.anothor_forward, path)
        origin_result = layer.anothor_forward(inps)
        load_func = paddle.jit.load(path)
        load_result = load_func(inps)
        np.testing.assert_array_equal(origin_result.numpy(), load_result.numpy())

class TestJitSaveLoadFunctionWithParamCase3(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        paddle.disable_static()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.temp_dir.cleanup()

    def test_jit_save_load_function(self):
        if False:
            while True:
                i = 10

        class LinearNet(paddle.nn.Layer):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self._linear = paddle.nn.Linear(5, 6)

            def forward(self, x):
                if False:
                    return 10
                return paddle.tanh(x)

            @paddle.jit.to_static
            def anothor_forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self._linear(x)
        layer = LinearNet()
        inps = paddle.rand([3, 5])
        origin = layer.anothor_forward(inps)
        path = os.path.join(self.temp_dir.name, 'test_jit_save_load_function_with_params_case3/func')
        paddle.jit.save(layer.anothor_forward, path)
        load_func = paddle.jit.load(path)
        load_result = load_func(inps)
        np.testing.assert_array_equal(load_result.numpy(), origin.numpy())

class TestJitSaveLoadDataParallel(unittest.TestCase):

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

    def verify_inference_correctness(self, layer, path):
        if False:
            i = 10
            return i + 15
        layer.eval()
        loaded_layer = paddle.jit.load(path)
        loaded_layer.eval()
        x = paddle.to_tensor(np.random.random((1, 784)).astype('float32'))
        pred = layer(x).numpy()
        loaded_pred = loaded_layer(x).numpy()
        np.testing.assert_array_equal(pred, loaded_pred, err_msg='Result diff when load and inference:\nlayer result:\n{}\nloaded layer result:\n{}'.format(pred, loaded_pred))

    def test_jit_save_data_parallel_with_inputspec(self):
        if False:
            i = 10
            return i + 15
        layer = LinearNetNotDeclarative(784, 1)
        layer = paddle.DataParallel(layer)
        path = os.path.join(self.temp_dir.name, 'jit_save_data_parallel_with_inputspec/model')
        paddle.jit.save(layer=layer, path=path, input_spec=[InputSpec(shape=[None, 784])])
        self.verify_inference_correctness(layer, path)

    def test_jit_save_data_parallel_with_to_static(self):
        if False:
            print('Hello World!')
        layer = LinearNetWithInputSpec(784, 1)
        layer = paddle.DataParallel(layer)
        path = os.path.join(self.temp_dir.name, 'jit_save_data_parallel_with_to_static/model')
        paddle.jit.save(layer, path)
        self.verify_inference_correctness(layer, path)

class InputSepcLayer(paddle.nn.Layer):
    """
    A layer with InputSpec to test InputSpec compatibility
    """

    @paddle.jit.to_static(input_spec=[InputSpec(shape=[None, 8], dtype='float32', name='x'), InputSpec(shape=[None, 1], dtype='float64', name='y')])
    def forward(self, x, y):
        if False:
            print('Hello World!')
        return (x, y)

class TestInputSpecCompatibility(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            return 10
        self.temp_dir.cleanup()

    def _assert_input_spec_layer_return(self, expect_layer, test_layer):
        if False:
            while True:
                i = 10
        input_x = paddle.uniform([8, 8], dtype='float32')
        input_y = paddle.uniform([8, 1], dtype='float64')
        expected_result = expect_layer(input_x, input_y)
        test_result = test_layer(input_x, input_y)
        np.testing.assert_allclose(expected_result[0].numpy(), test_result[0].numpy())
        np.testing.assert_allclose(expected_result[1].numpy(), test_result[1].numpy())

    def test_jit_save_compatible_input_sepc(self):
        if False:
            i = 10
            return i + 15
        layer = InputSepcLayer()
        save_dir = os.path.join(self.temp_dir.name, 'jit_save_compatible_input_spec')
        path = save_dir + '/model'
        paddle.jit.save(layer=layer, path=path)
        no_input_spec_layer = paddle.jit.load(path)
        self._assert_input_spec_layer_return(layer, no_input_spec_layer)
        shutil.rmtree(save_dir)
        paddle.jit.save(layer=layer, path=path, input_spec=[InputSpec(shape=[None, 8], dtype='float32', name='x'), InputSpec(shape=[None, 1], dtype='float64', name='y')])
        same_input_spec_layer = paddle.jit.load(path)
        self._assert_input_spec_layer_return(layer, same_input_spec_layer)
        shutil.rmtree(save_dir)
        paddle.jit.save(layer=layer, path=path, input_spec=[InputSpec(shape=[8, 8], dtype='float32'), InputSpec(shape=[8, -1], dtype='float64')])
        compatible_input_spec_layer = paddle.jit.load(path)
        self._assert_input_spec_layer_return(layer, compatible_input_spec_layer)
        shutil.rmtree(save_dir)

    def test_jit_save_incompatible_input_sepc(self):
        if False:
            for i in range(10):
                print('nop')
        layer = InputSepcLayer()
        save_dir = os.path.join(self.temp_dir.name, 'jit_save_compatible_input_spec')
        path = save_dir + '/model'
        with self.assertRaises(ValueError):
            paddle.jit.save(layer=layer, path=path, input_spec=[InputSpec(shape=[None, 8], dtype='float64'), InputSpec(shape=[None, 1], dtype='float64')])
        with self.assertRaises(ValueError):
            paddle.jit.save(layer=layer, path=path, input_spec=[InputSpec(shape=[None, 8, 1], dtype='float32'), InputSpec(shape=[None, 1], dtype='float64')])
        with self.assertRaises(ValueError):
            paddle.jit.save(layer=layer, path=path, input_spec=[InputSpec(shape=[None, 8], dtype='float32'), InputSpec(shape=[None, 2], dtype='float64')])
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

class NotJitForward(paddle.nn.Layer):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()

    def forward(self, x, y):
        if False:
            return 10
        return x + y

class TestNotJitForward(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.temp_dir.cleanup()

    def test_jit_not_save_forward(self):
        if False:
            i = 10
            return i + 15
        layer = NotJitForward()
        save_dir = os.path.join(self.temp_dir.name, 'jit_not_save_forward')
        path = save_dir + '/model'
        paddle.jit.save(layer=layer, path=path, skip_forward=True)
        self.assertTrue(not os.path.exists(path + '.pdmodel'))
        self.assertTrue(not os.path.exists(path + '.pdparam'))
        with self.assertRaises(ValueError):
            paddle.jit.load(path=path)
        shutil.rmtree(save_dir)
if __name__ == '__main__':
    unittest.main()