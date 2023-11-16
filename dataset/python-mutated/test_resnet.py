import jittor as jt
from jittor import nn, Module
from jittor.models import resnet
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
skip_this_test = False
if os.name == 'nt':
    skip_this_test = True

class MnistNet(Module):

    def __init__(self):
        if False:
            print('Hello World!')
        self.model = resnet.Resnet18()
        self.layer = nn.Linear(1000, 10)

    def execute(self, x):
        if False:
            return 10
        x = self.model(x)
        x = self.layer(x)
        return x

@unittest.skipIf(skip_this_test, 'skip_this_test')
class TestResnetFp32(unittest.TestCase):

    def setup_seed(self, seed):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(seed)
        random.seed(seed)
        jt.seed(seed)

    @unittest.skipIf(not jt.has_cuda, 'Cuda not found')
    @jt.flag_scope(use_cuda=1, use_stat_allocator=1)
    def test_resnet(self):
        if False:
            while True:
                i = 10
        self.setup_seed(1)
        self.batch_size = int(os.environ.get('TEST_BATCH_SIZE', '100'))
        self.weight_decay = 0.0001
        self.momentum = 0.9
        self.learning_rate = 0.1
        if jt.flags.amp_reg:
            self.learning_rate = 0.01
        self.train_loader = MNIST(train=True, transform=trans.Resize(224)).set_attrs(batch_size=self.batch_size, shuffle=True)
        self.train_loader.num_workers = 4
        loss_list = []
        acc_list = []
        mnist_net = MnistNet()
        global prev
        prev = time.time()
        SGD = nn.SGD(mnist_net.parameters(), self.learning_rate, self.momentum, self.weight_decay)
        self.train_loader.endless = True
        for (data, target) in self.train_loader:
            batch_id = self.train_loader.batch_id
            epoch_id = self.train_loader.epoch_id
            data = data.float_auto()
            output = mnist_net(data)
            loss = nn.cross_entropy_loss(output, target)
            SGD.step(loss)

            def callback(epoch_id, batch_id, loss, output, target):
                if False:
                    return 10
                global prev
                pred = np.argmax(output, axis=1)
                acc = np.mean(target == pred)
                loss_list.append(loss[0])
                acc_list.append(acc)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f} \tTime:{:.3f}'.format(epoch_id, batch_id, 600, 1.0 * batch_id / 6.0, loss[0], acc, time.time() - prev))
            jt.fetch(epoch_id, batch_id, loss, output, target, callback)
            mem_used = jt.flags.stat_allocator_total_alloc_byte - jt.flags.stat_allocator_total_free_byte
            assert mem_used < 5600000000.0, mem_used
            if jt.flags.amp_reg:
                continue
            if jt.in_mpi:
                assert jt.core.number_of_lived_vars() < 8100, jt.core.number_of_lived_vars()
            else:
                assert jt.core.number_of_lived_vars() < 7000, jt.core.number_of_lived_vars()
            if self.train_loader.epoch_id >= 2:
                break
        jt.sync_all(True)
        assert np.mean(loss_list[-50:]) < 0.5
        assert np.mean(acc_list[-50:]) > 0.8

@unittest.skipIf(skip_this_test, 'skip_this_test')
class TestResnetFp16(TestResnetFp32):

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        jt.flags.auto_mixed_precision_level = 5

    def tearDown(self):
        if False:
            return 10
        jt.flags.auto_mixed_precision_level = 0
if __name__ == '__main__':
    unittest.main()