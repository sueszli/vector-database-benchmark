import platform
import unittest
from test_train_step import TestTrainStepTinyModel, loss_fn_tiny_model, train_step_tiny_model
import paddle
from paddle.vision.models import resnet18

class TestTrainStepResNet18Sgd(TestTrainStepTinyModel):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.input = paddle.randn([64, 3, 224, 224])
        self.net_creator = resnet18
        self.lr_creator = lambda : 0.001
        self.optimizer_creator = paddle.optimizer.SGD
        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 0.0001
        if platform.system() == 'Windows':
            self.rtol = 0.001
if __name__ == '__main__':
    unittest.main()