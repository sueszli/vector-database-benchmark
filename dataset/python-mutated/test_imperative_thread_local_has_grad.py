import threading
import time
import unittest
import numpy as np
import paddle
from paddle import nn

class SimpleNet(nn.Layer):

    def __init__(self, in_dim, out_dim):
        if False:
            print('Hello World!')
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        return self.fc(x)

class TestCases(unittest.TestCase):

    @paddle.no_grad()
    def thread_1_main(self):
        if False:
            while True:
                i = 10
        time.sleep(8)

    def thread_2_main(self):
        if False:
            i = 10
            return i + 15
        in_dim = 10
        out_dim = 3
        net = SimpleNet(in_dim, out_dim)
        for _ in range(1000):
            x = paddle.to_tensor(np.random.rand(32, in_dim).astype('float32'))
            self.assertTrue(x.stop_gradient)
            x = net(x)
            self.assertFalse(x.stop_gradient)

    def test_main(self):
        if False:
            print('Hello World!')
        threads = []
        for _ in range(10):
            threads.append(threading.Thread(target=self.thread_1_main))
        threads.append(threading.Thread(target=self.thread_2_main))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
if __name__ == '__main__':
    unittest.main()