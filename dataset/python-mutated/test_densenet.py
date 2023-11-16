import jittor as jt
from jittor import nn, Module
from jittor.models import densenet
import numpy as np
import sys, os
import random
import math
import unittest
from jittor.test.test_reorder_tuner import simple_parser
from jittor.test.test_log import find_log_with_re
from jittor.dataset.mnist import MNIST
import jittor.transform as trans
import time
skip_this_test = True

class MnistNet(Module):

    def __init__(self):
        if False:
            print('Hello World!')
        self.model = densenet.densenet169()
        self.layer = nn.Linear(1000, 10)

    def execute(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = self.model(x)
        x = self.layer(x)
        return x

@unittest.skipIf(skip_this_test, 'skip_this_test')
class TestDensenet(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if False:
            i = 10
            return i + 15
        self.batch_size = 100
        self.weight_decay = 0.0001
        self.momentum = 0.9
        self.learning_rate = 0.1
        self.train_loader = MNIST(train=True, transform=trans.Resize(224)).set_attrs(batch_size=self.batch_size, shuffle=True)
        self.train_loader.num_workers = 4

    def setup_seed(self, seed):
        if False:
            i = 10
            return i + 15
        np.random.seed(seed)
        random.seed(seed)
        jt.seed(seed)

    @unittest.skipIf(not jt.has_cuda, 'Cuda not found')
    @jt.flag_scope(use_cuda=1, use_stat_allocator=1)
    def test_densenet(self):
        if False:
            while True:
                i = 10
        self.setup_seed(1)
        loss_list = []
        acc_list = []
        mnist_net = MnistNet()
        global prev
        prev = time.time()
        SGD = nn.SGD(mnist_net.parameters(), self.learning_rate, self.momentum, self.weight_decay)
        for (batch_idx, (data, target)) in enumerate(self.train_loader):
            output = mnist_net(data)
            loss = nn.cross_entropy_loss(output, target)
            SGD.step(loss)

            def callback(batch_idx, loss, output, target):
                if False:
                    print('Hello World!')
                global prev
                pred = np.argmax(output, axis=1)
                acc = np.mean(target == pred)
                loss_list.append(loss[0])
                acc_list.append(acc)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f} \tTime:{:.3f}'.format(0, batch_idx, 600, 1.0 * batch_idx / 6.0, loss[0], acc, time.time() - prev))
            jt.fetch(batch_idx, loss, output, target, callback)
        jt.sync_all(True)
        assert np.mean(loss_list[-50:]) < 0.3
        assert np.mean(acc_list[-50:]) > 0.9
if __name__ == '__main__':
    unittest.main()