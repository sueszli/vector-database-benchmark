import itertools
import unittest
import numpy as np
from test_imperative_base import new_program_scope
import paddle
from paddle import base
from paddle.base import core
from paddle.distributed.fleet.meta_optimizers import DGCMomentumOptimizer

class MLP(paddle.nn.Layer):

    def __init__(self, param_attr=None, bias_attr=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._fc1 = paddle.nn.Linear(784, 10)
        self._fc2 = paddle.nn.Linear(10, 10)

    def forward(self, inputs):
        if False:
            return 10
        y = self._fc1(inputs)
        y = self._fc2(y)
        return y

class TestImperativeOptimizerBase(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.batch_num = 20

    def get_optimizer_dygraph(self, parameter_list):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def get_optimizer(self):
        if False:
            return 10
        raise NotImplementedError()

    def reader_decorator(self, reader):
        if False:
            return 10

        def _reader_imple():
            if False:
                for i in range(10):
                    print('nop')
            for item in reader():
                image = np.array(item[0]).reshape(1, 784)
                label = np.array(item[1]).astype('int64').reshape(1)
                yield (image, label)
        return _reader_imple

    def _check_exception(self, exception_message, place=None):
        if False:
            print('Hello World!')
        seed = 90
        batch_size = 128
        if place is None:
            place = base.CUDAPlace(0) if core.is_compiled_with_cuda() else base.CPUPlace()
        try:
            paddle.disable_static()
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)
            mlp = MLP()
            optimizer = self.get_optimizer_dygraph(parameter_list=mlp.parameters())
        except Exception as e:
            assert str(e) == exception_message
        finally:
            paddle.enable_static()

    def _check_mlp(self, place=None):
        if False:
            print('Hello World!')
        seed = 90
        batch_size = 128
        if place is None:
            place = base.CPUPlace() if not core.is_compiled_with_cuda() else base.CUDAPlace(0)
        paddle.disable_static(place)
        paddle.seed(seed)
        paddle.framework.random._manual_program_seed(seed)
        mlp = MLP()
        optimizer = self.get_optimizer_dygraph(parameter_list=mlp.parameters())
        batch_py_reader = base.io.PyReader(capacity=1)
        batch_py_reader.decorate_sample_list_generator(paddle.batch(self.reader_decorator(paddle.dataset.mnist.train()), batch_size=batch_size, drop_last=True), places=base.CPUPlace())
        dy_param_init_value = {}
        for (batch_id, data) in enumerate(batch_py_reader()):
            if batch_id >= self.batch_num:
                break
            img = data[0]
            label = data[1]
            label.stop_gradient = True
            img = paddle.reshape(img, shape=[batch_size, -1])
            cost = mlp(img)
            avg_loss = paddle.mean(cost)
            dy_out = avg_loss.numpy()
            if batch_id == 0:
                for param in mlp.parameters():
                    dy_param_init_value[param.name] = param.numpy()
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            if isinstance(optimizer._learning_rate, paddle.optimizer.lr.LRScheduler):
                if isinstance(optimizer._learning_rate, paddle.optimizer.lr.ReduceOnPlateau):
                    optimizer._learning_rate.step(avg_loss)
                else:
                    optimizer._learning_rate.step()
            mlp.clear_gradients()
            dy_param_value = {}
            for param in mlp.parameters():
                dy_param_value[param.name] = param.numpy()
        paddle.enable_static()
        with new_program_scope():
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)
            if place is None:
                place = base.CPUPlace() if not core.is_compiled_with_cuda() else base.CUDAPlace(0)
            exe = base.Executor(place)
            mlp = MLP()
            optimizer = self.get_optimizer()
            train_reader = paddle.batch(paddle.dataset.mnist.train(), batch_size=128, drop_last=True)
            img = paddle.static.data(name='pixel', shape=[-1, 1, 28, 28], dtype='float32')
            label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
            img = paddle.reshape(img, shape=[batch_size, 784])
            cost = mlp(img)
            avg_loss = paddle.mean(cost)
            optimizer.minimize(avg_loss)
            static_param_init_value = {}
            static_param_name_list = []
            for param in mlp.parameters():
                static_param_name_list.append(param.name)
            out = exe.run(base.default_startup_program(), fetch_list=static_param_name_list)
            for i in range(len(static_param_name_list)):
                static_param_init_value[static_param_name_list[i]] = out[i]
            for (batch_id, data) in enumerate(train_reader()):
                if batch_id >= self.batch_num:
                    break
                static_x_data = np.array([x[0].reshape(1, 28, 28) for x in data]).astype('float32')
                y_data = np.array([x[1] for x in data]).astype('int64').reshape([128, 1])
                fetch_list = [avg_loss.name]
                fetch_list.extend(static_param_name_list)
                out = exe.run(base.default_main_program(), feed={'pixel': static_x_data, 'label': y_data}, fetch_list=fetch_list)
                if isinstance(optimizer._learning_rate, paddle.optimizer.lr.LRScheduler):
                    if isinstance(optimizer._learning_rate, paddle.optimizer.lr.ReduceOnPlateau):
                        optimizer._learning_rate.step(out[0])
                    else:
                        optimizer._learning_rate.step()
                static_param_value = {}
                static_out = out[0]
                for i in range(1, len(out)):
                    static_param_value[static_param_name_list[i - 1]] = out[i]
        for (key, value) in static_param_init_value.items():
            np.testing.assert_allclose(value, dy_param_init_value[key], rtol=1e-05)
        if core.is_compiled_with_rocm():
            np.testing.assert_allclose(static_out, dy_out, rtol=1e-05, atol=0.001)
        else:
            np.testing.assert_allclose(static_out, dy_out, rtol=1e-05)
        for (key, value) in static_param_value.items():
            if core.is_compiled_with_rocm():
                np.testing.assert_allclose(value, dy_param_value[key], rtol=1e-05, atol=0.001)
            else:
                np.testing.assert_allclose(value, dy_param_value[key], rtol=1e-05)

class TestImperativeOptimizerPiecewiseDecay(TestImperativeOptimizerBase):

    def get_optimizer_dygraph(self, parameter_list):
        if False:
            i = 10
            return i + 15
        bd = [3, 6, 9]
        optimizer = paddle.optimizer.SGD(learning_rate=paddle.optimizer.lr.PiecewiseDecay(boundaries=bd, values=[0.1 * 0.1 ** i for i in range(len(bd) + 1)]), parameters=parameter_list)
        return optimizer

    def get_optimizer(self):
        if False:
            for i in range(10):
                print('nop')
        bd = [3, 6, 9]
        optimizer = paddle.optimizer.SGD(learning_rate=paddle.optimizer.lr.PiecewiseDecay(boundaries=bd, values=[0.1 * 0.1 ** i for i in range(len(bd) + 1)]))
        return optimizer

    def test_sgd(self):
        if False:
            while True:
                i = 10
        self._check_mlp()

class TestImperativeOptimizerNaturalExpDecay(TestImperativeOptimizerBase):

    def get_optimizer_dygraph(self, parameter_list):
        if False:
            for i in range(10):
                print('nop')
        optimizer = paddle.optimizer.SGD(learning_rate=paddle.optimizer.lr.NaturalExpDecay(learning_rate=0.5, gamma=0.9), parameters=parameter_list)
        return optimizer

    def get_optimizer(self):
        if False:
            for i in range(10):
                print('nop')
        optimizer = paddle.optimizer.SGD(learning_rate=paddle.optimizer.lr.NaturalExpDecay(learning_rate=0.5, gamma=0.9))
        return optimizer

    def test_sgd(self):
        if False:
            print('Hello World!')
        self._check_mlp()

class TestImperativeOptimizerExponentialDecay(TestImperativeOptimizerBase):

    def get_optimizer_dygraph(self, parameter_list):
        if False:
            while True:
                i = 10
        optimizer = paddle.optimizer.SGD(learning_rate=paddle.optimizer.lr.ExponentialDecay(learning_rate=0.5, gamma=0.9), parameters=parameter_list)
        return optimizer

    def get_optimizer(self):
        if False:
            return 10
        optimizer = paddle.optimizer.SGD(learning_rate=paddle.optimizer.lr.ExponentialDecay(learning_rate=0.5, gamma=0.9))
        return optimizer

    def test_sgd(self):
        if False:
            for i in range(10):
                print('nop')
        self._check_mlp()

class TestImperativeOptimizerInverseTimeDecay(TestImperativeOptimizerBase):

    def get_optimizer_dygraph(self, parameter_list):
        if False:
            print('Hello World!')
        optimizer = paddle.optimizer.Adam(learning_rate=paddle.optimizer.lr.InverseTimeDecay(learning_rate=0.5, gamma=0.9), parameters=parameter_list)
        return optimizer

    def get_optimizer(self):
        if False:
            print('Hello World!')
        optimizer = paddle.optimizer.Adam(learning_rate=paddle.optimizer.lr.InverseTimeDecay(learning_rate=0.5, gamma=0.9))
        return optimizer

    def test_adam(self):
        if False:
            print('Hello World!')
        self._check_mlp()

class TestImperativeOptimizerPolynomialDecay(TestImperativeOptimizerBase):

    def get_optimizer_dygraph(self, parameter_list):
        if False:
            return 10
        optimizer = paddle.optimizer.SGD(learning_rate=paddle.optimizer.lr.PolynomialDecay(learning_rate=0.5, decay_steps=5, cycle=self.cycle), parameters=parameter_list)
        return optimizer

    def get_optimizer(self):
        if False:
            for i in range(10):
                print('nop')
        optimizer = paddle.optimizer.SGD(learning_rate=paddle.optimizer.lr.PolynomialDecay(learning_rate=0.5, decay_steps=5, cycle=self.cycle))
        return optimizer

    def test_sgd_cycle(self):
        if False:
            return 10
        self.cycle = True
        self._check_mlp()

    def test_sgd(self):
        if False:
            while True:
                i = 10
        self.cycle = False
        self._check_mlp()

class TestImperativeOptimizerCosineAnnealingDecay(TestImperativeOptimizerBase):

    def get_optimizer_dygraph(self, parameter_list):
        if False:
            while True:
                i = 10
        optimizer = paddle.optimizer.SGD(learning_rate=paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.5, T_max=5), parameters=parameter_list)
        return optimizer

    def get_optimizer(self):
        if False:
            for i in range(10):
                print('nop')
        optimizer = paddle.optimizer.SGD(learning_rate=paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.5, T_max=5))
        return optimizer

    def test_sgd(self):
        if False:
            while True:
                i = 10
        self._check_mlp()

class TestImperativeOptimizerNoamDecay(TestImperativeOptimizerBase):

    def get_optimizer_dygraph(self, parameter_list):
        if False:
            return 10
        optimizer = paddle.optimizer.SGD(learning_rate=paddle.optimizer.lr.NoamDecay(d_model=0.01, warmup_steps=100, verbose=True), parameters=parameter_list)
        return optimizer

    def get_optimizer(self):
        if False:
            return 10
        optimizer = paddle.optimizer.SGD(learning_rate=paddle.optimizer.lr.NoamDecay(d_model=0.01, warmup_steps=100))
        return optimizer

    def test_sgd(self):
        if False:
            while True:
                i = 10
        self._check_mlp()

class TestImperativeOptimizerLambdaDecay(TestImperativeOptimizerBase):

    def get_optimizer_dygraph(self, parameter_list):
        if False:
            while True:
                i = 10
        optimizer = paddle.optimizer.SGD(learning_rate=paddle.optimizer.lr.LambdaDecay(learning_rate=0.5, lr_lambda=lambda epoch: 0.9 ** epoch), parameters=parameter_list)
        return optimizer

    def get_optimizer(self):
        if False:
            for i in range(10):
                print('nop')
        optimizer = paddle.optimizer.SGD(learning_rate=paddle.optimizer.lr.LambdaDecay(learning_rate=0.5, lr_lambda=lambda epoch: 0.9 ** epoch))
        return optimizer

    def test_sgd(self):
        if False:
            i = 10
            return i + 15
        self._check_mlp()

class TestImperativeOptimizerLinearWarmup(TestImperativeOptimizerBase):

    def get_optimizer_dygraph(self, parameter_list):
        if False:
            print('Hello World!')
        optimizer = paddle.optimizer.SGD(learning_rate=paddle.optimizer.lr.LinearWarmup(learning_rate=0.5, warmup_steps=20, start_lr=0, end_lr=0.5), parameters=parameter_list)
        return optimizer

    def get_optimizer(self):
        if False:
            return 10
        optimizer = paddle.optimizer.SGD(learning_rate=paddle.optimizer.lr.LinearWarmup(learning_rate=0.5, warmup_steps=20, start_lr=0, end_lr=0.5, verbose=True))
        return optimizer

    def test_sgd(self):
        if False:
            while True:
                i = 10
        self._check_mlp()

class TestImperativeOptimizerMultiStepDecay(TestImperativeOptimizerBase):

    def get_optimizer_dygraph(self, parameter_list):
        if False:
            return 10
        optimizer = paddle.optimizer.SGD(learning_rate=paddle.optimizer.lr.MultiStepDecay(learning_rate=0.5, milestones=[2, 4, 6], gamma=0.8), parameters=parameter_list)
        return optimizer

    def get_optimizer(self):
        if False:
            i = 10
            return i + 15
        optimizer = paddle.optimizer.SGD(learning_rate=paddle.optimizer.lr.MultiStepDecay(learning_rate=0.5, milestones=[2, 4, 6], gamma=0.8))
        return optimizer

    def test_sgd(self):
        if False:
            return 10
        self._check_mlp()

class TestImperativeOptimizerStepLR(TestImperativeOptimizerBase):

    def get_optimizer_dygraph(self, parameter_list):
        if False:
            print('Hello World!')
        optimizer = paddle.optimizer.SGD(learning_rate=paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=5, gamma=0.8), parameters=parameter_list)
        return optimizer

    def get_optimizer(self):
        if False:
            for i in range(10):
                print('nop')
        optimizer = paddle.optimizer.SGD(learning_rate=paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=5, gamma=0.8))
        return optimizer

    def test_sgd(self):
        if False:
            while True:
                i = 10
        self._check_mlp()

class TestImperativeOptimizerReduceOnPlateau(TestImperativeOptimizerBase):

    def get_optimizer_dygraph(self, parameter_list):
        if False:
            for i in range(10):
                print('nop')
        optimizer = paddle.optimizer.SGD(learning_rate=paddle.optimizer.lr.ReduceOnPlateau(learning_rate=0.5), parameters=parameter_list)
        return optimizer

    def get_optimizer(self):
        if False:
            print('Hello World!')
        optimizer = paddle.optimizer.SGD(learning_rate=paddle.optimizer.lr.ReduceOnPlateau(learning_rate=0.5))
        return optimizer

    def test_sgd(self):
        if False:
            i = 10
            return i + 15
        self._check_mlp()

class TestOptimizerLearningRate(unittest.TestCase):

    def test_constant_lr(self):
        if False:
            print('Hello World!')
        with base.dygraph.guard():
            a = np.random.uniform(-0.1, 0.1, [10, 10]).astype('float32')
            linear = paddle.nn.Linear(10, 10)
            a = base.dygraph.to_variable(a)
            b = linear(a)
            loss = paddle.mean(b)
            adam = paddle.optimizer.Adam(0.001, parameters=linear.parameters())
            np.testing.assert_allclose(adam.get_lr(), 0.001, rtol=1e-06, atol=0.0)
            for i in range(10):
                adam.minimize(loss)
                lr = adam.get_lr()
                np.testing.assert_allclose(lr, 0.001, rtol=1e-06, atol=0.0)

    def test_lr_decay(self):
        if False:
            for i in range(10):
                print('nop')
        with base.dygraph.guard():
            a = np.random.uniform(-0.1, 0.1, [10, 10]).astype('float32')
            linear = paddle.nn.Linear(10, 10)
            a = base.dygraph.to_variable(a)
            b = linear(a)
            loss = paddle.mean(b)
            bd = [2, 4, 6, 8]
            value = [0.2, 0.4, 0.6, 0.8, 1.0]
            scheduler = paddle.optimizer.lr.PiecewiseDecay(bd, value)
            adam = paddle.optimizer.Adam(scheduler, parameters=linear.parameters())
            np.testing.assert_allclose(adam.get_lr(), 0.2, rtol=1e-06, atol=0.0)
            ret = [0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0]
            for i in range(12):
                adam.minimize(loss)
                lr = adam.get_lr()
                np.testing.assert_allclose(lr, ret[i], rtol=1e-06, atol=0.0)
                scheduler.step()

    def test_lr_scheduler_natural_exp(self):
        if False:
            for i in range(10):
                print('nop')
        with base.dygraph.guard():
            a = np.random.uniform(-0.1, 0.1, [10, 10]).astype('float32')
            linear = paddle.nn.Linear(10, 10)
            a = base.dygraph.to_variable(a)
            b = linear(a)
            loss = paddle.mean(b)
            base_lr = 1.0
            scheduler = paddle.optimizer.lr.NaturalExpDecay(1.0, gamma=0.5)
            adam = paddle.optimizer.Adam(scheduler, parameters=linear.parameters())
            np.testing.assert_allclose(adam.get_lr(), 1.0, rtol=1e-06, atol=0.0)
            ret = [1.0, np.exp(-0.5), np.exp(-1)]
            for i in range(3):
                adam.minimize(loss)
                lr = adam.get_lr()
                np.testing.assert_allclose(lr, ret[i], rtol=1e-06, atol=0.0)
                scheduler.step()

    def test_set_lr(self):
        if False:
            print('Hello World!')
        with base.dygraph.guard():
            a = np.random.uniform(-0.1, 0.1, [10, 10]).astype('float32')
            linear = paddle.nn.Linear(10, 10)
            a = base.dygraph.to_variable(a)
            b = linear(a)
            loss = paddle.mean(b)
            adam = paddle.optimizer.Adam(0.1, parameters=linear.parameters())
            lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
            for i in range(5):
                adam.set_lr(lr_list[i])
                adam.minimize(loss)
                lr = adam.get_lr()
                np.testing.assert_allclose(lr, lr_list[i], rtol=1e-06, atol=0.0)
            with self.assertRaises(TypeError):
                lr_var = paddle.static.create_global_var(shape=[1], value=0.7, dtype='float32')
                adam.set_lr(lr_var)
            with self.assertRaises(RuntimeError):
                adam = paddle.optimizer.Adam(paddle.optimizer.lr.NaturalExpDecay(learning_rate=0.1, gamma=0.5), parameters=linear.parameters())
                adam.set_lr(0.01)

    def test_set_lr_scheduler(self):
        if False:
            for i in range(10):
                print('nop')
        with base.dygraph.guard():
            a = np.random.uniform(-0.1, 0.1, [10, 10]).astype('float32')
            linear = paddle.nn.Linear(10, 10)
            a = base.dygraph.to_variable(a)
            b = linear(a)
            loss = paddle.mean(b)
            adam = paddle.optimizer.Adam(0.1, parameters=linear.parameters())
            scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.2, step_size=5, gamma=0.6)
            adam.set_lr_scheduler(scheduler)
            adam.minimize(loss)
            lr = adam.get_lr()
            np.testing.assert_allclose(lr, 0.2, rtol=1e-06, atol=0.0)
            scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=0.5, milestones=[2, 4, 6], gamma=0.8)
            adam.set_lr_scheduler(scheduler)
            adam.minimize(loss)
            lr = adam.get_lr()
            np.testing.assert_allclose(lr, 0.5, rtol=1e-06, atol=0.0)

class TestImperativeMomentumOptimizer(TestImperativeOptimizerBase):

    def get_optimizer_dygraph(self, parameter_list):
        if False:
            for i in range(10):
                print('nop')
        optimizer = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=parameter_list)
        return optimizer

    def get_optimizer(self):
        if False:
            return 10
        optimizer = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9)
        return optimizer

    def test_momentum(self):
        if False:
            print('Hello World!')
        self._check_mlp()

class TestImperativeLarsMomentumOptimizer(TestImperativeOptimizerBase):

    def get_optimizer_dygraph(self, parameter_list):
        if False:
            while True:
                i = 10
        optimizer = paddle.incubate.optimizer.LarsMomentumOptimizer(learning_rate=0.001, momentum=0.9, parameter_list=parameter_list)
        return optimizer

    def get_optimizer(self):
        if False:
            return 10
        optimizer = paddle.incubate.optimizer.LarsMomentumOptimizer(learning_rate=0.001, momentum=0.9)
        return optimizer

    def test_larsmomentum(self):
        if False:
            for i in range(10):
                print('nop')
        self._check_mlp()

class TestImperativeAdagradOptimizer(TestImperativeOptimizerBase):

    def get_optimizer_dygraph(self, parameter_list):
        if False:
            while True:
                i = 10
        optimizer = paddle.optimizer.Adagrad(learning_rate=0.2, parameters=parameter_list)
        return optimizer

    def get_optimizer(self):
        if False:
            return 10
        optimizer = paddle.optimizer.Adagrad(learning_rate=0.2)
        return optimizer

    def test_adagrad(self):
        if False:
            i = 10
            return i + 15
        self._check_mlp()

class TestImperativeAdamaxOptimizer(TestImperativeOptimizerBase):

    def get_optimizer_dygraph(self, parameter_list):
        if False:
            i = 10
            return i + 15
        optimizer = paddle.optimizer.Adamax(learning_rate=0.2, parameters=parameter_list)
        return optimizer

    def get_optimizer(self):
        if False:
            return 10
        optimizer = paddle.optimizer.Adamax(learning_rate=0.2)
        return optimizer

    def test_adamax(self):
        if False:
            i = 10
            return i + 15
        self._check_mlp()

class TestImperativeAdadeltaOptimizer(TestImperativeOptimizerBase):

    def get_optimizer_dygraph(self, parameter_list):
        if False:
            i = 10
            return i + 15
        optimizer = paddle.optimizer.Adadelta(learning_rate=0.0003, epsilon=1e-06, rho=0.95, parameters=parameter_list)
        return optimizer

    def get_optimizer(self):
        if False:
            i = 10
            return i + 15
        optimizer = paddle.optimizer.Adadelta(learning_rate=0.0003, epsilon=1e-06, rho=0.95)
        return optimizer

    def test_adadelta(self):
        if False:
            for i in range(10):
                print('nop')
        self._check_mlp()

class TestImperativeRMSPropOptimizer(TestImperativeOptimizerBase):

    def get_optimizer_dygraph(self, parameter_list):
        if False:
            print('Hello World!')
        optimizer = paddle.optimizer.RMSProp(learning_rate=0.1, parameters=parameter_list)
        return optimizer

    def get_optimizer(self):
        if False:
            while True:
                i = 10
        optimizer = paddle.optimizer.RMSProp(learning_rate=0.1)
        return optimizer

    def test_rmsprop(self):
        if False:
            i = 10
            return i + 15
        self._check_mlp()

def exclude_fn(param):
    if False:
        for i in range(10):
            print('nop')
    return param.name.endswith('.b_0')

class TestImperativeLambOptimizer(TestImperativeOptimizerBase):

    def get_optimizer_dygraph(self, parameter_list):
        if False:
            while True:
                i = 10
        optimizer = paddle.optimizer.Lamb(learning_rate=0.002, exclude_from_weight_decay_fn=exclude_fn, parameters=parameter_list)
        return optimizer

    def get_optimizer(self):
        if False:
            print('Hello World!')
        optimizer = paddle.optimizer.Lamb(learning_rate=0.002, exclude_from_weight_decay_fn=exclude_fn)
        return optimizer

    def _test_lamb(self):
        if False:
            i = 10
            return i + 15
        self._check_mlp()

class TestImperativeDGCMomentumOptimizer(TestImperativeOptimizerBase):

    def get_optimizer_dygraph(self, parameter_list):
        if False:
            while True:
                i = 10
        optimizer = DGCMomentumOptimizer(learning_rate=0.0001, momentum=0.9, rampup_step=1000, rampup_begin_step=1252, sparsity=[0.999, 0.999])
        return optimizer

    def test_dgcmomentum(self):
        if False:
            i = 10
            return i + 15
        exception_message = "In dygraph, don't support DGCMomentumOptimizer."
        self._check_exception(exception_message)

class TestImperativeExponentialMovingAverage(TestImperativeOptimizerBase):

    def get_optimizer_dygraph(self, parameter_list):
        if False:
            for i in range(10):
                print('nop')
        optimizer = paddle.static.ExponentialMovingAverage(0.999)
        return optimizer

    def test_exponentialmoving(self):
        if False:
            while True:
                i = 10
        exception_message = "In dygraph, don't support ExponentialMovingAverage."
        self._check_exception(exception_message)

class TestImperativePipelineOptimizer(TestImperativeOptimizerBase):

    def get_optimizer_dygraph(self, parameter_list):
        if False:
            print('Hello World!')
        optimizer = paddle.optimizer.SGD(learning_rate=0.5, parameters=parameter_list)
        optimizer = paddle.incubate.optimizer.PipelineOptimizer(optimizer)
        return optimizer

    def test_pipline(self):
        if False:
            print('Hello World!')
        exception_message = "In dygraph, don't support PipelineOptimizer."
        self._check_exception(exception_message)

class TestImperativeLookaheadOptimizer(TestImperativeOptimizerBase):

    def get_optimizer_dygraph(self, parameter_list):
        if False:
            print('Hello World!')
        optimizer = paddle.optimizer.SGD(learning_rate=0.5, parameters=parameter_list)
        optimizer = paddle.incubate.optimizer.LookAhead(optimizer, alpha=0.5, k=5)
        return optimizer

    def test_lookahead(self):
        if False:
            i = 10
            return i + 15
        exception_message = "In dygraph, don't support LookaheadOptimizer."
        self._check_exception(exception_message)

class TestImperativeRecomputeOptimizer(TestImperativeOptimizerBase):

    def get_optimizer_dygraph(self, parameter_list):
        if False:
            print('Hello World!')
        optimizer = paddle.optimizer.SGD(learning_rate=0.5, parameters=parameter_list)
        optimizer = paddle.incubate.optimizer.RecomputeOptimizer(optimizer)
        return optimizer

    def test_recompute(self):
        if False:
            return 10
        exception_message = "In dygraph, don't support RecomputeOptimizer."
        self._check_exception(exception_message)

class TestImperativeOptimizerList(unittest.TestCase):

    def test_parameter_list(self):
        if False:
            while True:
                i = 10
        with base.dygraph.guard():
            linear_1 = paddle.nn.Linear(10, 10)
            linear_2 = paddle.nn.Linear(10, 10)
            sgd = paddle.optimizer.SGD(1.0, parameters=itertools.chain(linear_1.parameters(), linear_2.parameters()))
            in_np = np.random.uniform(-0.1, 0.1, [10, 10]).astype('float32')
            in_data = base.dygraph.to_variable(in_np)
            y = linear_1(in_data)
            y = linear_2(y)
            loss = paddle.mean(y)
            loss.backward()
            sgd.minimize(loss)
            self.assertTrue(len(sgd._parameter_list) == len(linear_1.parameters() + linear_2.parameters()))
if __name__ == '__main__':
    unittest.main()