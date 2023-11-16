import contextlib
import unittest
import numpy as np
import seresnext_net
from simple_nets import fc_with_batchnorm, init_data, simple_fc_net
from test_parallel_executor_transformer import DeviceType, get_feed_data_reader, transformer
import paddle
from paddle import base
from paddle.base import core

def simple_fc_net_with_accuracy(use_feed):
    if False:
        for i in range(10):
            print('nop')
    img = paddle.static.data(name='image', shape=[-1, 784], dtype='float32')
    label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
    hidden = img
    for _ in range(4):
        hidden = paddle.static.nn.fc(hidden, size=200, activation='relu', bias_attr=base.ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0)))
    prediction = paddle.static.nn.fc(hidden, size=10, activation='softmax')
    loss = paddle.nn.functional.cross_entropy(input=prediction, label=label, reduction='none', use_softmax=False)
    loss = paddle.mean(loss)
    accuracy_out = paddle.static.accuracy(input=prediction, label=label, k=5)
    return loss

def cond_net(use_feed=None):
    if False:
        i = 10
        return i + 15
    x = paddle.static.data(name='x', shape=[-1, 4], dtype='float32')
    label = paddle.static.data('label', shape=[-1, 1], dtype='int64')
    prediction = paddle.static.nn.fc(x, size=1, activation=None)

    def loss1(pred, label):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.static.data(name='x', shape=[-1, 4], dtype='float32')
        loss = paddle.nn.functional.cross_entropy(input=pred, label=label, reduction='none', use_softmax=False)
        avg_loss = paddle.mean(loss, name='mean_cross_entropy_loss')
        return avg_loss

    def loss2(pred, label):
        if False:
            print('Hello World!')
        loss = paddle.nn.functional.softmax_with_cross_entropy(logits=pred, label=label)
        avg_loss = paddle.mean(loss, name='mean_softmax_loss')
        return avg_loss
    two = paddle.tensor.fill_constant([1], 'int32', 2)
    pred = two == 0
    avg_loss = paddle.static.nn.case([(pred, lambda : loss1(prediction, label))], lambda : loss2(prediction, label))
    return avg_loss

def pylayer_net(use_feed=None):
    if False:
        return 10
    x = paddle.static.data(name='x', shape=[-1, 4], dtype='float32')
    label = paddle.static.data('label', shape=[-1, 1], dtype='int64')

    def forward_fn(x):
        if False:
            while True:
                i = 10
        y = 3 * x
        return y

    def backward_fn(dy):
        if False:
            print('Hello World!')
        grad = paddle.exp(dy)
        return grad
    y = paddle.static.nn.static_pylayer(forward_fn, [x], backward_fn)
    hidden = paddle.static.nn.fc(x=[y], size=4, activation='softmax')
    loss = paddle.nn.functional.cross_entropy(input=hidden, label=label, reduction='none', use_softmax=False)
    loss = paddle.mean(loss, name='mean_softmax_loss')
    return loss

def optimization_in_cond_net(with_optimize=False):
    if False:
        print('Hello World!')
    x = paddle.static.data(name='x', shape=[-1, 4], dtype='float32')
    label = paddle.static.data('label', shape=[-1, 1], dtype='int64')
    prediction = paddle.static.nn.fc(x, size=1, activation=None)

    def loss1(opt, pred, label, with_optimize):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.static.data(name='x', shape=[-1, 4], dtype='float32')
        loss = paddle.nn.functional.cross_entropy(input=pred, label=label, reduction='none', use_softmax=False)
        avg_loss = paddle.mean(loss, name='mean_cross_entropy_loss')
        if with_optimize:
            opt.minimize(avg_loss)
        return avg_loss

    def loss2(opt, pred, label, with_optimize):
        if False:
            print('Hello World!')
        loss = paddle.nn.functional.softmax_with_cross_entropy(logits=pred, label=label)
        avg_loss = paddle.mean(loss, name='mean_softmax_loss')
        if with_optimize:
            opt.minimize(avg_loss)
        return avg_loss
    sgd = paddle.optimizer.SGD(learning_rate=0.1)
    two = paddle.tensor.fill_constant([1], 'int32', 2)
    pred = two == 0
    avg_loss = paddle.static.nn.case([(pred, lambda : loss1(sgd, prediction, label, with_optimize))], lambda : loss2(sgd, prediction, label, with_optimize))
    return avg_loss

def optimization_in_pylayer_net(with_optimize=False):
    if False:
        i = 10
        return i + 15
    x = paddle.static.data(name='x', shape=[-1, 4], dtype='float32')
    label = paddle.static.data('label', shape=[-1, 1], dtype='int64')

    def forward_fn(x):
        if False:
            print('Hello World!')
        y = 3 * x
        return y

    def backward_fn(dy):
        if False:
            for i in range(10):
                print('nop')
        grad = paddle.exp(dy)
        return grad
    y = paddle.static.nn.static_pylayer(forward_fn, [x], backward_fn)
    hidden = 3 * y
    loss = paddle.nn.functional.softmax_with_cross_entropy(logits=hidden, label=label)
    loss = paddle.mean(loss, name='mean_softmax_loss')
    sgd = paddle.optimizer.SGD(learning_rate=0.1)
    if with_optimize:
        sgd.minimize(loss)
    return loss

class TestProgramPruneBackward(unittest.TestCase):

    def program_compare(self, program_a, program_b):
        if False:
            print('Hello World!')
        assert isinstance(program_a, base.framework.Program), 'The first argument should be base.framework.Program.'
        assert isinstance(program_b, base.framework.Program), 'The second argument should be base.framework Program.'
        self.assertEqual(len(program_a.blocks), len(program_b.blocks))
        for idx in range(len(program_a.blocks)):
            block_a = program_a.blocks[idx]
            block_b = program_b.blocks[idx]
            self.assertEqual(len(block_a.ops), len(block_b.ops))
            self.assertEqual(len(block_a.vars), len(block_b.vars))
            for op_idx in range(len(block_a.ops)):
                self.assertEqual(block_a.ops[op_idx].type, block_b.ops[op_idx].type)
            for var_key in list(block_a.vars.keys()):
                self.assertTrue(block_b.has_var(var_key))

    def check_prune_correctness(self, method, feed_dict, optimizer):
        if False:
            print('Hello World!')
        loss = method(use_feed=False)
        main_program = base.default_main_program()
        test_prog_orig = main_program.clone(for_test=True)
        optimizer().minimize(loss)
        test_prog_prune = main_program.clone(for_test=True)
        self.program_compare(test_prog_orig, test_prog_prune)
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            exe = base.Executor(place)
            exe.run(base.default_startup_program())
            (loss_data_prune,) = exe.run(test_prog_prune, feed=feed_dict, fetch_list=[loss.name])
            (loss_data_orig,) = exe.run(test_prog_orig, feed=feed_dict, fetch_list=[loss.name])
            self.assertEqual(loss_data_orig, loss_data_prune)

    def test_simple_fc_net(self):
        if False:
            return 10

        def optimizer():
            if False:
                i = 10
                return i + 15
            optimizer = paddle.optimizer.SGD(learning_rate=0.001, weight_decay=paddle.regularizer.L2Decay(0.0001))
            return optimizer
        with self.program_scope_guard():
            (img, label) = init_data()
            self.check_prune_correctness(method=simple_fc_net, feed_dict={'image': img, 'label': label}, optimizer=optimizer)

    def test_simple_fc_net_with_accuracy(self):
        if False:
            while True:
                i = 10

        def optimizer():
            if False:
                i = 10
                return i + 15
            optimizer = paddle.optimizer.SGD(learning_rate=0.001, weight_decay=paddle.regularizer.L2Decay(0.0001))
            return optimizer
        with self.program_scope_guard():
            (img, label) = init_data()
            self.check_prune_correctness(method=simple_fc_net_with_accuracy, feed_dict={'image': img, 'label': label}, optimizer=optimizer)

    def test_batchnorm_fc(self):
        if False:
            for i in range(10):
                print('nop')

        def optimizer():
            if False:
                for i in range(10):
                    print('nop')
            optimizer = paddle.optimizer.SGD(learning_rate=0.001, weight_decay=paddle.regularizer.L2Decay(0.0001))
            return optimizer
        with self.program_scope_guard():
            (img, label) = init_data()
            self.check_prune_correctness(method=fc_with_batchnorm, feed_dict={'image': img, 'label': label}, optimizer=optimizer)

    def test_seresnet(self):
        if False:
            return 10
        with self.program_scope_guard():
            self.check_prune_correctness(method=seresnext_net.model, feed_dict=seresnext_net.feed_dict(use_device=DeviceType.CPU), optimizer=seresnext_net.optimizer)

    def test_transformer(self):
        if False:
            while True:
                i = 10

        def optimizer():
            if False:
                i = 10
                return i + 15
            optimizer = paddle.optimizer.Adam(learning_rate=0.001, weight_decay=paddle.regularizer.L2Decay(0.0001))
            return optimizer
        with self.program_scope_guard():
            feed_dict = get_feed_data_reader().get_next(base.Executor(core.CPUPlace()), base.default_main_program())
            self.check_prune_correctness(method=transformer, feed_dict=feed_dict, optimizer=optimizer)

    def test_cond(self):
        if False:
            i = 10
            return i + 15

        def optimizer():
            if False:
                i = 10
                return i + 15
            optimizer = paddle.optimizer.SGD(learning_rate=0.01)
            return optimizer
        with self.program_scope_guard():
            x_in = np.random.random(size=(10, 4)).astype('float32')
            label_in = np.random.randint(1, size=(10, 1)).astype('int64')
            feed_dict = {'x': x_in, 'label': label_in}
            self.check_prune_correctness(method=cond_net, feed_dict=feed_dict, optimizer=optimizer)

    def test_pylayer(self):
        if False:
            i = 10
            return i + 15

        def optimizer():
            if False:
                return 10
            optimizer = paddle.optimizer.SGD(learning_rate=0.01)
            return optimizer
        with self.program_scope_guard():
            x_in = np.random.random(size=(10, 4)).astype('float32')
            label_in = np.random.randint(1, size=(10, 1)).astype('int64')
            feed_dict = {'x': x_in, 'label': label_in}
            self.check_prune_correctness(method=pylayer_net, feed_dict=feed_dict, optimizer=optimizer)

    def test_optimization_in_cond(self):
        if False:
            i = 10
            return i + 15
        x_in = np.random.random(size=(10, 4)).astype('float32')
        label_in = np.random.randint(1, size=(10, 1)).astype('int64')
        feed_dict = {'x': x_in, 'label': label_in}
        with self.program_scope_guard():
            loss = optimization_in_cond_net(False)
            main_program = base.default_main_program()
            test_prog_orig = main_program.clone(for_test=True)
            place = core.CPUPlace()
            exe = base.Executor(place)
            exe.run(base.default_startup_program())
            (loss_data_orig,) = exe.run(test_prog_orig, feed=feed_dict, fetch_list=[loss.name])
        with self.program_scope_guard():
            loss = optimization_in_cond_net(True)
            main_program = base.default_main_program()
            test_prog_prune = main_program.clone(for_test=True)
            place = core.CPUPlace()
            exe = base.Executor(place)
            exe.run(base.default_startup_program())
            (loss_data_prune,) = exe.run(test_prog_prune, feed=feed_dict, fetch_list=[loss.name])
        self.program_compare(test_prog_orig, test_prog_prune)
        self.assertEqual(loss_data_orig, loss_data_prune)

    def test_optimization_in_pylayer(self):
        if False:
            return 10
        x_in = np.random.random(size=(10, 4)).astype('float32')
        label_in = np.random.randint(1, size=(10, 1)).astype('int64')
        feed_dict = {'x': x_in, 'label': label_in}
        with self.program_scope_guard():
            loss = optimization_in_pylayer_net(False)
            main_program = base.default_main_program()
            test_prog_orig = main_program.clone(for_test=True)
            place = core.CPUPlace()
            exe = base.Executor(place)
            exe.run(base.default_startup_program())
            (loss_data_orig,) = exe.run(test_prog_orig, feed=feed_dict, fetch_list=[loss.name])
        with self.program_scope_guard():
            loss = optimization_in_pylayer_net(True)
            main_program = base.default_main_program()
            test_prog_prune = main_program.clone(for_test=True)
            place = core.CPUPlace()
            exe = base.Executor(place)
            exe.run(base.default_startup_program())
            (loss_data_prune,) = exe.run(test_prog_prune, feed=feed_dict, fetch_list=[loss.name])
        self.program_compare(test_prog_orig, test_prog_prune)
        self.assertEqual(loss_data_orig, loss_data_prune)

    @contextlib.contextmanager
    def program_scope_guard(self):
        if False:
            i = 10
            return i + 15
        prog = base.Program()
        startup_prog = base.Program()
        scope = base.core.Scope()
        with base.scope_guard(scope):
            with base.program_guard(prog, startup_prog):
                with base.unique_name.guard():
                    yield
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()