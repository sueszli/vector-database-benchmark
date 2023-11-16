import tempfile
import unittest
import numpy as np
from op_test_ipu import IPUD2STest
import paddle

class SimpleLayer(paddle.nn.Layer):

    def __init__(self, use_ipu=False):
        if False:
            while True:
                i = 10
        super().__init__()
        self.use_ipu = use_ipu
        self.conv = paddle.nn.Conv2D(in_channels=3, out_channels=1, kernel_size=2, stride=1)

    def forward(self, x, target=None):
        if False:
            while True:
                i = 10
        x = self.conv(x)
        x = paddle.flatten(x, 1, -1)
        if target is not None:
            x = paddle.nn.functional.softmax(x)
            loss = paddle.paddle.nn.functional.cross_entropy(x, target, reduction='none', use_softmax=False)
            if self.use_ipu:
                loss = paddle.incubate.identity_loss(loss, 1)
            else:
                loss = paddle.mean(loss)
            return (x, loss)
        return x

class TestBase(IPUD2STest):

    def setUp(self):
        if False:
            return 10
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
            print('Hello World!')
        paddle.seed(self.SEED)
        np.random.seed(self.SEED)
        model = SimpleLayer(use_ipu)
        specs = [paddle.static.InputSpec(name='x', shape=[32, 3, 10, 10], dtype='float32'), paddle.static.InputSpec(name='target', shape=[32], dtype='int64')]
        model = paddle.jit.to_static(model, input_spec=specs)
        optim = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())
        data = paddle.uniform((32, 3, 10, 10), dtype='float32')
        label = paddle.randint(0, 10, shape=[32], dtype='int64')
        model_path = '{}/model_state_dict_{}.pdparams'.format(self.save_path, 'ipu' if use_ipu else 'cpu')
        optim_path = '{}/optim_state_dict_{}.pdopt'.format(self.save_path, 'ipu' if use_ipu else 'cpu')
        if use_ipu:
            paddle.set_device('ipu')
            ipu_strategy = paddle.static.IpuStrategy()
            ipu_strategy.set_graph_config(num_ipus=1, is_training=True, micro_batch_size=1, enable_manual_shard=False)
            ipu_strategy.set_precision_config(enable_fp16=True)
            ipu_strategy.set_optimizer(optim)
            data = data.astype(np.float16)
        epochs = 100
        result = []
        for _ in range(epochs):
            (pred, loss) = model(data, label)
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
            (pred, loss) = model(data, label)
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
        cpu_loss = self._test(False).flatten()
        ipu_loss = self._test(True).flatten()
        np.testing.assert_allclose(ipu_loss, cpu_loss, rtol=1e-05, atol=0.01)
if __name__ == '__main__':
    unittest.main()