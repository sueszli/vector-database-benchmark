import unittest
import numpy as np
import paddle
from paddle import Model, base, set_device
from paddle.metric import Accuracy
from paddle.nn.layer.loss import CrossEntropyLoss
from paddle.static import InputSpec as Input
from paddle.vision.datasets import MNIST
from paddle.vision.models import LeNet

class MnistDataset(MNIST):

    def __init__(self, mode, return_label=True):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(mode=mode)
        self.return_label = return_label

    def __getitem__(self, idx):
        if False:
            return 10
        img = np.reshape(self.images[idx], [1, 28, 28])
        if self.return_label:
            return (img, np.array(self.labels[idx]).astype('int64'))
        return (img,)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.images)

def compute_accuracy(pred, gt):
    if False:
        i = 10
        return i + 15
    pred = np.argmax(pred, -1)
    gt = np.array(gt)
    correct = pred[:, np.newaxis] == gt
    return np.sum(correct) / correct.shape[0]

@unittest.skipIf(not base.is_compiled_with_cuda(), 'CPU testing is not supported')
class TestDistTraining(unittest.TestCase):

    def test_static_multiple_gpus(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        device = set_device('gpu')
        im_shape = (-1, 1, 28, 28)
        batch_size = 128
        inputs = [Input(im_shape, 'float32', 'image')]
        labels = [Input([None, 1], 'int64', 'label')]
        model = Model(LeNet(), inputs, labels)
        optim = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters())
        model.prepare(optim, CrossEntropyLoss(), Accuracy())
        train_dataset = MnistDataset(mode='train')
        val_dataset = MnistDataset(mode='test')
        test_dataset = MnistDataset(mode='test', return_label=False)
        cbk = paddle.callbacks.ProgBarLogger(50)
        model.fit(train_dataset, val_dataset, epochs=2, batch_size=batch_size, callbacks=cbk)
        eval_result = model.evaluate(val_dataset, batch_size=batch_size)
        output = model.predict(test_dataset, batch_size=batch_size, stack_outputs=True)
        np.testing.assert_equal(output[0].shape[0], len(test_dataset))
        acc = compute_accuracy(output[0], val_dataset.labels)
        np.testing.assert_allclose(acc, eval_result['acc'])
if __name__ == '__main__':
    unittest.main()