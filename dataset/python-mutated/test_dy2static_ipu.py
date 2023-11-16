import tempfile
import unittest
from functools import partial
import numpy as np
from op_test_ipu import IPUD2STest
import paddle
from paddle.jit import to_static
from paddle.jit.dy2static.program_translator import ProgramCache
from paddle.optimizer.lr import LRScheduler

class SimpleLayer(paddle.nn.Layer):

    def __init__(self, loss_op=None, use_softmax=True, use_reduction=True, use_identity_loss=True):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.loss_op = loss_op
        self.conv = paddle.nn.Conv2D(in_channels=3, out_channels=1, kernel_size=2, stride=1)
        self.use_softmax = use_softmax
        self.use_reduction = use_reduction
        self.use_identity_loss = use_identity_loss

    @to_static()
    def forward(self, x, target=None):
        if False:
            return 10
        x = self.conv(x)
        x = paddle.flatten(x, 1, -1)
        if target is not None:
            if self.use_softmax:
                x = paddle.nn.functional.softmax(x)
            loss = paddle.paddle.nn.functional.cross_entropy(x, target, reduction='none', use_softmax=False)
            if self.use_reduction:
                loss = paddle.mean(loss)
            if self.use_identity_loss:
                loss = paddle.incubate.identity_loss(loss, 1)
            return (x, loss)
        return x

class TestBase(IPUD2STest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_op_attrs()
        self.set_data_feed()

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        pass

    def set_data_feed(self):
        if False:
            return 10
        self.data = paddle.uniform((8, 3, 10, 10), dtype='float32')
        self.label = paddle.randint(0, 10, shape=[8], dtype='int64')

    def create_model(self, use_ipu=False):
        if False:
            i = 10
            return i + 15
        return SimpleLayer(loss_op=self.loss_op, use_softmax=True, use_reduction=not use_ipu, use_identity_loss=use_ipu)

    def _test(self, use_ipu=False):
        if False:
            return 10
        paddle.seed(self.SEED)
        np.random.seed(self.SEED)
        model = self.create_model(use_ipu)
        optim = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())
        if use_ipu:
            paddle.set_device('ipu')
            ipu_strategy = paddle.static.IpuStrategy()
            ipu_strategy.set_graph_config(num_ipus=1, is_training=True, micro_batch_size=1, enable_manual_shard=False)
            ipu_strategy.set_optimizer(optim)
        epochs = 100
        result = []
        for _ in range(epochs):
            (pred, loss) = model(self.data, self.label)
            if not use_ipu:
                loss.backward()
                optim.step()
                optim.clear_grad()
            result.append(loss)
        if use_ipu:
            ipu_strategy.release_patch()
        return np.array(result)

    def test_training(self):
        if False:
            print('Hello World!')
        ipu_loss = self._test(True).flatten()
        cpu_loss = self._test(False).flatten()
        np.testing.assert_allclose(ipu_loss, cpu_loss, rtol=1e-05, atol=0.0001)

class TestSaveLoad(TestBase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.save_path = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            while True:
                i = 10
        super().tearDown()
        self.save_path.cleanup()

    def _test(self, use_ipu=False):
        if False:
            i = 10
            return i + 15
        paddle.seed(self.SEED)
        np.random.seed(self.SEED)
        model = self.create_model(use_ipu)
        optim = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())
        model_path = '{}/model_state_dict_{}.pdparams'.format(self.save_path, 'ipu' if use_ipu else 'cpu')
        optim_path = '{}/optim_state_dict_{}.pdopt'.format(self.save_path, 'ipu' if use_ipu else 'cpu')
        if use_ipu:
            paddle.set_device('ipu')
            ipu_strategy = paddle.static.IpuStrategy()
            ipu_strategy.set_graph_config(num_ipus=1, is_training=True, micro_batch_size=1, enable_manual_shard=False)
            ipu_strategy.set_optimizer(optim)
        epochs = 100
        result = []
        for _ in range(epochs):
            (pred, loss) = model(self.data, self.label)
            if not use_ipu:
                loss.backward()
                optim.step()
                optim.clear_grad()
            result.append(loss)
        if use_ipu:
            paddle.base.core.IpuBackend.get_instance().weights_to_host()
        paddle.save(model.state_dict(), model_path)
        paddle.save(optim.state_dict(), optim_path)
        model.set_state_dict(paddle.load(model_path))
        optim.set_state_dict(paddle.load(optim_path))
        for _ in range(epochs):
            (pred, loss) = model(self.data, self.label)
            if not use_ipu:
                loss.backward()
                optim.step()
                optim.clear_grad()
            result.append(loss)
        if use_ipu:
            ipu_strategy.release_patch()
        return np.array(result)

class TestPatch(IPUD2STest):

    def setUp(cls):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()

    def test(self, use_ipu=False):
        if False:
            return 10
        old_getter = ProgramCache.__getitem__
        old_step = LRScheduler.step
        ipu_strategy = paddle.static.IpuStrategy()
        ipu_strategy.release_patch()
        reset_getter = ProgramCache.__getitem__
        reset_step = LRScheduler.step
        self.assertTrue(reset_getter is old_getter)
        self.assertTrue(reset_step is old_step)

class TestWithoutIdentityLoss1(TestBase):

    def create_model(self, use_ipu=False):
        if False:
            while True:
                i = 10
        return SimpleLayer(loss_op=self.loss_op, use_softmax=True, use_reduction=True, use_identity_loss=False)

class TestWithoutIdentityLoss2(TestBase):

    def set_op_attrs(self):
        if False:
            return 10
        self.loss_op = paddle.paddle.nn.functional.softmax_with_cross_entropy

    def set_data_feed(self):
        if False:
            return 10
        self.data = paddle.uniform((8, 3, 10, 10), dtype='float32')
        self.label = paddle.randint(0, 10, shape=[8, 1], dtype='int64')

    def create_model(self, use_ipu=False):
        if False:
            print('Hello World!')
        return SimpleLayer(loss_op=self.loss_op, use_softmax=False, use_reduction=True, use_identity_loss=False)

class TestWithoutIdentityLoss3(TestBase):

    def set_op_attrs(self):
        if False:
            return 10
        self.loss_op = partial(paddle.nn.functional.kl_div, reduction='none')

    def set_data_feed(self):
        if False:
            return 10
        self.data = paddle.uniform((8, 3, 10, 10), dtype='float32')
        self.label = paddle.rand(shape=[8, 81], dtype='float32')

    def create_model(self, use_ipu=False):
        if False:
            for i in range(10):
                print('nop')
        return SimpleLayer(loss_op=self.loss_op, use_softmax=True, use_reduction=True, use_identity_loss=False)

class TestWithoutIdentityLoss4(TestBase):

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.loss_op = paddle.nn.functional.binary_cross_entropy

    def set_data_feed(self):
        if False:
            i = 10
            return i + 15
        self.data = paddle.uniform((8, 3, 10, 10), dtype='float32')
        self.label = paddle.rand(shape=[8, 81], dtype='float32')

    def create_model(self, use_ipu=False):
        if False:
            print('Hello World!')
        return SimpleLayer(loss_op=self.loss_op, use_softmax=True, use_reduction=False, use_identity_loss=False)

class TestWithoutIdentityLoss5(TestBase):

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        self.loss_op = paddle.nn.functional.binary_cross_entropy_with_logits

    def set_data_feed(self):
        if False:
            for i in range(10):
                print('nop')
        self.data = paddle.uniform((8, 3, 10, 10), dtype='float32')
        self.label = paddle.randint(0, 10, shape=[8, 81], dtype='int64').astype('float32')

    def create_model(self, use_ipu=False):
        if False:
            i = 10
            return i + 15
        return SimpleLayer(loss_op=self.loss_op, use_softmax=True, use_reduction=True, use_identity_loss=False)
if __name__ == '__main__':
    unittest.main()