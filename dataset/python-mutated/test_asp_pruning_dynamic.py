import unittest
import numpy as np
import paddle
from paddle.base import core
from paddle.incubate.asp import ASPHelper

class MyLayer(paddle.nn.Layer):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=3, out_channels=2, kernel_size=3, padding=2)
        self.linear1 = paddle.nn.Linear(1352, 32)
        self.linear2 = paddle.nn.Linear(32, 10)

    def forward(self, img):
        if False:
            print('Hello World!')
        hidden = self.conv1(img)
        hidden = paddle.flatten(hidden, start_axis=1)
        hidden = self.linear1(hidden)
        prediction = self.linear2(hidden)
        return prediction

class TestASPDynamicPruningBase(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.layer = MyLayer()
        place = paddle.CPUPlace()
        if core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        self.img = paddle.to_tensor(np.random.uniform(low=-0.5, high=0.5, size=(32, 3, 24, 24)), dtype=np.float32, place=place, stop_gradient=False)
        self.set_config()

    def set_config(self):
        if False:
            print('Hello World!')
        self.mask_gen_func = 'mask_1d'
        self.mask_check_func = paddle.incubate.asp.CheckMethod.CHECK_1D

    def test_inference_pruning(self):
        if False:
            return 10
        self.__pruning_and_checking(False)

    def test_training_pruning(self):
        if False:
            for i in range(10):
                print('nop')
        optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=self.layer.parameters())
        optimizer = paddle.incubate.asp.decorate(optimizer)
        self.__pruning_and_checking(True)

    def __pruning_and_checking(self, with_mask):
        if False:
            for i in range(10):
                print('nop')
        paddle.incubate.asp.prune_model(self.layer, mask_algo=self.mask_gen_func, with_mask=with_mask)
        for param in self.layer.parameters():
            if ASPHelper._is_supported_layer(paddle.static.default_main_program(), param.name):
                mat = param.numpy()
                if len(param.shape) == 4 and param.shape[1] < 4 or (len(param.shape) == 2 and param.shape[0] < 4):
                    self.assertFalse(paddle.incubate.asp.check_sparsity(mat.T, n=2, m=4))
                else:
                    self.assertTrue(paddle.incubate.asp.check_sparsity(mat.T, func_name=self.mask_check_func, n=2, m=4))

class TestASPDynamicPruning1D(TestASPDynamicPruningBase):

    def set_config(self):
        if False:
            i = 10
            return i + 15
        self.mask_gen_func = 'mask_1d'
        self.mask_check_func = paddle.incubate.asp.CheckMethod.CHECK_1D

class TestASPDynamicPruning2DBest(TestASPDynamicPruningBase):

    def set_config(self):
        if False:
            while True:
                i = 10
        self.mask_gen_func = 'mask_2d_best'
        self.mask_check_func = paddle.incubate.asp.CheckMethod.CHECK_2D

class TestASPDynamicPruning2DGreedy(TestASPDynamicPruningBase):

    def set_config(self):
        if False:
            for i in range(10):
                print('nop')
        self.mask_gen_func = 'mask_2d_greedy'
        self.mask_check_func = paddle.incubate.asp.CheckMethod.CHECK_2D
if __name__ == '__main__':
    unittest.main()