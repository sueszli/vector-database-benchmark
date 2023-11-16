import jittor as jt
import unittest
import numpy as np
import random
from .test_core import expect_error
from jittor.dataset.mnist import MNIST
import jittor.transform as trans
from tqdm import tqdm

class BBox:

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.x = x

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return bool((self.x == other.x).all())

def test_ring_buffer():
    if False:
        return 10
    buffer = jt.RingBuffer(2000)

    def test_send_recv(data):
        if False:
            i = 10
            return i + 15
        print('test send recv', type(data))
        buffer.push(data)
        recv = buffer.pop()
        if isinstance(data, (np.ndarray, jt.Var)):
            assert (recv == data).all()
        else:
            assert data == recv
    n_byte = 0
    test_send_recv(1)
    n_byte += 1 + 8
    assert n_byte == buffer.total_pop() and n_byte == buffer.total_push()
    test_send_recv(100000000000)
    n_byte += 1 + 8
    assert n_byte == buffer.total_pop() and n_byte == buffer.total_push()
    test_send_recv(1e-05)
    n_byte += 1 + 8
    assert n_byte == buffer.total_pop() and n_byte == buffer.total_push()
    test_send_recv(100000000000.0)
    n_byte += 1 + 8
    assert n_byte == buffer.total_pop() and n_byte == buffer.total_push()
    test_send_recv('float32')
    n_byte += 1 + 8 + 7
    assert n_byte == buffer.total_pop() and n_byte == buffer.total_push()
    test_send_recv('')
    n_byte += 1 + 8 + 0
    assert n_byte == buffer.total_pop() and n_byte == buffer.total_push()
    test_send_recv('xxxxxxxxxx')
    n_byte += 1 + 8 + 10
    assert n_byte == buffer.total_pop() and n_byte == buffer.total_push()
    test_send_recv([1, 0.2])
    n_byte += 1 + 8 + 1 + 8 + 1 + 8
    assert n_byte == buffer.total_pop() and n_byte == buffer.total_push()
    test_send_recv({'asd': 1})
    n_byte += 1 + 8 + 1 + 8 + 3 + 1 + 8
    assert n_byte == buffer.total_pop() and n_byte == buffer.total_push()
    test_send_recv(np.random.rand(10, 10))
    n_byte += 1 + 16 + 4 + 10 * 10 * 8
    assert n_byte == buffer.total_pop() and n_byte == buffer.total_push(), (n_byte, buffer.total_pop(), n_byte, buffer.total_push())
    test_send_recv(test_ring_buffer)
    test_send_recv(jt.array(np.random.rand(10, 10)))
    bbox = BBox(jt.array(np.random.rand(10, 10)))
    test_send_recv(bbox)
    expect_error(lambda : test_send_recv(np.random.rand(10, 1000)))

class TestRingBuffer(unittest.TestCase):

    def test_ring_buffer(self):
        if False:
            return 10
        test_ring_buffer()

    def test_dataset(self):
        if False:
            while True:
                i = 10
        return
        self.train_loader = MNIST(train=True, transform=trans.Resize(224)).set_attrs(batch_size=300, shuffle=True)
        self.train_loader.num_workers = 1
        import time
        for (batch_idx, (data, target)) in tqdm(enumerate(self.train_loader)):
            if batch_idx > 30:
                break
            pass
        for (batch_idx, (data, target)) in tqdm(enumerate(self.train_loader)):
            if batch_idx > 300:
                break
            pass
if __name__ == '__main__':
    unittest.main()