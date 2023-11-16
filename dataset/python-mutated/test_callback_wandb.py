import tempfile
import unittest
import paddle
import paddle.vision.transforms as T
from paddle.static import InputSpec
from paddle.vision.datasets import MNIST

class MnistDataset(MNIST):

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return 512

class TestWandbCallbacks(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.save_dir = tempfile.mkdtemp()

    def test_wandb_callback(self):
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
        callback = paddle.callbacks.WandbCallback(project='random', dir=self.save_dir, anonymous='must', mode='offline')
        model.fit(train_dataset, eval_dataset, batch_size=64, callbacks=callback)
if __name__ == '__main__':
    unittest.main()