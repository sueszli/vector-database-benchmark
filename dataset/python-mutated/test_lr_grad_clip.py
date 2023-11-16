import unittest
from test_to_static import MLPLayer, MyDataset
import paddle
from paddle.distributed.fleet import auto
paddle.enable_static()

class TestEngineBase(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.batch_size = 4
        self.batch_num = 5
        self.hidden_size = 1024
        self.init_model()
        self.init_optimizer()
        self.init_dataset()
        self.init_engine()

    def init_model(self):
        if False:
            i = 10
            return i + 15
        self.mlp = MLPLayer(hidden_size=self.hidden_size, intermediate_size=4 * self.hidden_size, dropout_ratio=0.1, initializer_range=0.02)
        self.loss = paddle.nn.CrossEntropyLoss()

    def init_optimizer(self):
        if False:
            i = 10
            return i + 15
        self.optimizer = paddle.optimizer.SGD(learning_rate=1e-05, parameters=self.mlp.parameters())

    def init_dataset(self):
        if False:
            for i in range(10):
                print('nop')
        self.dataset = MyDataset(self.batch_num * self.batch_size)

    def init_engine(self):
        if False:
            i = 10
            return i + 15
        self.engine = auto.Engine(model=self.mlp, loss=self.loss, optimizer=self.optimizer, metrics=paddle.metric.Accuracy())

class TestLRScheduler(TestEngineBase):

    def init_optimizer(self):
        if False:
            i = 10
            return i + 15
        scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=1e-05, T_max=10)
        self.optimizer = paddle.optimizer.SGD(learning_rate=scheduler)

    def test_lr_scheduler(self):
        if False:
            return 10
        self.init_engine()
        self.engine.fit(self.dataset, batch_size=self.batch_size)
        lr = self.engine._optimizer._learning_rate
        assert isinstance(lr, paddle.optimizer.lr.LRScheduler)

class TestGradClipByGlobalNorm(TestEngineBase):

    def init_optimizer(self):
        if False:
            for i in range(10):
                print('nop')
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
        self.optimizer = paddle.optimizer.SGD(learning_rate=1e-05, grad_clip=clip)

    def test_grad_clip(self):
        if False:
            i = 10
            return i + 15
        self.engine.fit(self.dataset, batch_size=self.batch_size)
        self.check_program()

    def check_program(self):
        if False:
            return 10
        ops = self.engine.main_program.global_block().ops
        has_grad_clip = False
        for op in ops:
            if op.desc.has_attr('op_namescope') and op.desc.attr('op_namescope').startswith('/gradient_clip'):
                has_grad_clip = True
                break
        assert has_grad_clip is True

class TestGradClipByNorm(TestGradClipByGlobalNorm):

    def init_optimizer(self):
        if False:
            i = 10
            return i + 15
        clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)
        self.optimizer = paddle.optimizer.SGD(learning_rate=1e-05, grad_clip=clip)
if __name__ == '__main__':
    unittest.main()