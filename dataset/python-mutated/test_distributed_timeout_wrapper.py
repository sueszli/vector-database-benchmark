import logging
import signal
import time
import unittest
import torch
from torch import nn
from fairseq.distributed import DistributedTimeoutWrapper

class ModuleWithDelay(nn.Module):

    def __init__(self, delay):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.delay = delay

    def forward(self, x):
        if False:
            while True:
                i = 10
        time.sleep(self.delay)
        return x

class TestDistributedTimeoutWrapper(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        logging.disable(logging.NOTSET)

    def test_no_timeout(self):
        if False:
            return 10
        module = DistributedTimeoutWrapper(ModuleWithDelay(1), 0, signal.SIGINT)
        module(torch.rand(5))
        module.stop_timeout()

    def test_timeout_safe(self):
        if False:
            for i in range(10):
                print('nop')
        module = DistributedTimeoutWrapper(ModuleWithDelay(1), 10, signal.SIGINT)
        module(torch.rand(5))
        module.stop_timeout()

    def test_timeout_killed(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(KeyboardInterrupt):
            module = DistributedTimeoutWrapper(ModuleWithDelay(5), 1, signal.SIGINT)
            module(torch.rand(5))
            module.stop_timeout()
if __name__ == '__main__':
    unittest.main()