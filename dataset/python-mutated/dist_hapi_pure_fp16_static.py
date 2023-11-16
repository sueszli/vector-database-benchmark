import unittest
import numpy as np
import paddle
from paddle import Model, base
from paddle.nn.layer.loss import CrossEntropyLoss
from paddle.static import InputSpec
from paddle.vision.models import LeNet

@unittest.skipIf(not base.is_compiled_with_cuda(), 'CPU testing is not supported')
class TestDistTrainingWithPureFP16(unittest.TestCase):

    def test_amp_training_purefp16(self):
        if False:
            print('Hello World!')
        if not base.is_compiled_with_cuda():
            self.skipTest('module not tested when ONLY_CPU compiling')
        data = np.random.random(size=(4, 1, 28, 28)).astype(np.float32)
        label = np.random.randint(0, 10, size=(4, 1)).astype(np.int64)
        paddle.enable_static()
        paddle.set_device('gpu')
        net = LeNet()
        amp_level = 'O2'
        inputs = InputSpec([None, 1, 28, 28], 'float32', 'x')
        labels = InputSpec([None, 1], 'int64', 'y')
        model = Model(net, inputs, labels)
        optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters(), multi_precision=True)
        amp_configs = {'level': amp_level, 'use_fp16_guard': False}
        model.prepare(optimizer=optim, loss=CrossEntropyLoss(reduction='sum'), amp_configs=amp_configs)
        model.train_batch([data], [label])
if __name__ == '__main__':
    unittest.main()