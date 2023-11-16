import random
import shutil
import tempfile
import time
import unittest
import paddle
import paddle.vision.transforms as T
from paddle.distributed.auto_parallel.static.callbacks import config_callbacks
from paddle.distributed.fleet import auto
from paddle.static import InputSpec
from paddle.vision.datasets import MNIST
from paddle.vision.models import LeNet
paddle.enable_static()

class TestCallbacks(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.save_dir = tempfile.mkdtemp()

    def tearDown(self):
        if False:
            return 10
        shutil.rmtree(self.save_dir)

    def run_callback(self):
        if False:
            print('Hello World!')
        epochs = 2
        steps = 5
        freq = 2
        eval_steps = 2
        inputs_spec = [InputSpec([None, 1, 28, 28], 'float32', 'image')]
        strategy = auto.Strategy()
        strategy.auto_mode = 'semi'
        engine = auto.Engine(LeNet(), strategy=strategy)
        engine.prepare(inputs_spec, mode='predict')
        cbks = config_callbacks(engine=engine, batch_size=128, epochs=epochs, steps=steps, log_freq=freq, verbose=self.verbose, metrics=['loss', 'acc'], save_dir=self.save_dir)
        cbks.on_begin('train')
        logs = {'loss': 50.341673, 'acc': 0.00256}
        for epoch in range(epochs):
            cbks.on_epoch_begin(epoch)
            for step in range(steps):
                cbks.on_batch_begin('train', step, logs)
                logs['loss'] -= random.random() * 0.1
                logs['acc'] += random.random() * 0.1
                time.sleep(0.005)
                cbks.on_batch_end('train', step, logs)
            cbks.on_epoch_end(epoch, logs)
            eval_logs = {'eval_loss': 20.341673, 'eval_acc': 0.256}
            params = {'steps': eval_steps, 'metrics': ['eval_loss', 'eval_acc']}
            cbks.on_begin('eval', params)
            for step in range(eval_steps):
                cbks.on_batch_begin('eval', step, eval_logs)
                eval_logs['eval_loss'] -= random.random() * 0.1
                eval_logs['eval_acc'] += random.random() * 0.1
                eval_logs['batch_size'] = 2
                time.sleep(0.005)
                cbks.on_batch_end('eval', step, eval_logs)
            cbks.on_end('eval', eval_logs)
            test_logs = {}
            params = {'steps': eval_steps}
            cbks.on_begin('predict', params)
            for step in range(eval_steps):
                cbks.on_batch_begin('predict', step, test_logs)
                test_logs['batch_size'] = 2
                time.sleep(0.005)
                cbks.on_batch_end('predict', step, test_logs)
            cbks.on_end('predict', test_logs)
        cbks.on_end('train')
        print(engine.history.history)

    def test_callback_verbose_0(self):
        if False:
            i = 10
            return i + 15
        self.verbose = 0
        self.run_callback()

    def test_callback_verbose_1(self):
        if False:
            for i in range(10):
                print('nop')
        self.verbose = 1
        self.run_callback()

    def test_callback_verbose_2(self):
        if False:
            i = 10
            return i + 15
        self.verbose = 2
        self.run_callback()

    def test_callback_verbose_3(self):
        if False:
            for i in range(10):
                print('nop')
        self.verbose = 3
        self.run_callback()

class TestCallbacksEngine(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.save_dir = tempfile.mkdtemp()
        transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
        self.train_dataset = MNIST(mode='train', transform=transform)
        self.test_dataset = MNIST(mode='test', transform=transform)
        self.prepare_engine()

    def tearDown(self):
        if False:
            print('Hello World!')
        shutil.rmtree(self.save_dir)

    def prepare_engine(self):
        if False:
            i = 10
            return i + 15
        model = paddle.vision.models.LeNet()
        loss = paddle.nn.CrossEntropyLoss()
        base_lr = 0.001
        boundaries = [5, 8]
        values = [base_lr * 0.1 ** i for i in range(len(boundaries) + 1)]
        lr = paddle.optimizer.lr.PiecewiseDecay(boundaries=boundaries, values=values, verbose=False)
        optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters())
        auto.fetch(model.parameters()[0], 'param0', logging=True)
        metrics = paddle.metric.Accuracy(topk=(1, 2))
        self.engine = auto.Engine(model, loss, optimizer, metrics)

    def test_fit_eval(self):
        if False:
            while True:
                i = 10
        history = self.engine.fit(train_data=self.train_dataset, valid_data=self.test_dataset, batch_size=128, steps_per_epoch=60, valid_steps=40, log_freq=20, save_dir=self.save_dir, save_freq=1)
        print(history.history)

    def test_eval(self):
        if False:
            for i in range(10):
                print('nop')
        self.engine.evaluate(valid_data=self.test_dataset, batch_size=128, steps=40, log_freq=10)

    def test_predict(self):
        if False:
            return 10
        logger_cbks = paddle.callbacks.ProgBarLogger()
        self.engine.predict(test_data=self.test_dataset, batch_size=128, callbacks=[logger_cbks])
if __name__ == '__main__':
    unittest.main()