"""
Topic: 定义上下文管理器的简单方法
Desc : 
"""
import time
from contextlib import contextmanager

@contextmanager
def timethis(label):
    if False:
        for i in range(10):
            print('nop')
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print('{}: {}'.format(label, end - start))
with timethis('counting'):
    n = 10000000
    while n > 0:
        n -= 1

@contextmanager
def list_transaction(orig_list):
    if False:
        print('Hello World!')
    working = list(orig_list)
    yield working
    orig_list[:] = working
import time

class timethis:

    def __init__(self, label):
        if False:
            i = 10
            return i + 15
        self.label = label

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self.start = time.time()

    def __exit__(self, exc_ty, exc_val, exc_tb):
        if False:
            print('Hello World!')
        end = time.time()
        print('{}: {}'.format(self.label, end - self.start))