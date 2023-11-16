import shutil
import tempfile
import unittest
import paddle
import paddle.vision.transforms as T
from paddle.static import InputSpec
from paddle.vision.datasets import MNIST

class MnistDataset(MNIST):

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return 512

class TestCallbacks(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.save_dir = tempfile.mkdtemp()

    def tearDown(self):
        if False:
            print('Hello World!')
        shutil.rmtree(self.save_dir)

    def test_visualdl_callback(self):
        if False:
            i = 10
            return i + 15
        inputs = [InputSpec([-1, 1, 28, 28], 'float32', 'image')]
        labels = [InputSpec([None, 1], 'int64', 'label')]
        transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
        train_dataset = MnistDataset(mode='train', transform=transform)
        eval_dataset = MnistDataset(mode='test', transform=transform)
        net = paddle.vision.models.LeNet()
        model = paddle.Model(net, inputs, labels)
        optim = paddle.optimizer.Adam(0.001, parameters=net.parameters())
        model.prepare(optimizer=optim, loss=paddle.nn.CrossEntropyLoss(), metrics=paddle.metric.Accuracy())
        callback = paddle.callbacks.VisualDL(log_dir='visualdl_log_dir')
        model.fit(train_dataset, eval_dataset, batch_size=64, callbacks=callback)
if __name__ == '__main__':
    unittest.main()