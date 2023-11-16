"""
Topic: 内联回调函数
Desc : 
"""
from queue import Queue
from functools import wraps

def apply_async(func, args, *, callback):
    if False:
        while True:
            i = 10
    result = func(*args)
    callback(result)

class Async:

    def __init__(self, func, args):
        if False:
            i = 10
            return i + 15
        self.func = func
        self.args = args

def inlined_async(func):
    if False:
        while True:
            i = 10

    @wraps(func)
    def wrapper(*args):
        if False:
            i = 10
            return i + 15
        f = func(*args)
        result_queue = Queue()
        result_queue.put(None)
        while True:
            print('1' * 15)
            result = result_queue.get()
            print('2' * 15)
            try:
                print('3' * 15)
                print('result={}'.format(result))
                a = f.send(result)
                print('4' * 15)
                apply_async(a.func, a.args, callback=result_queue.put)
                print('5' * 15)
            except StopIteration:
                break
    return wrapper

def add(x, y):
    if False:
        return 10
    return x + y

@inlined_async
def test():
    if False:
        return 10
    print('start'.center(20, '='))
    r = (yield Async(add, (2, 3)))
    print('last={}'.format(r))
    r = (yield Async(add, ('hello', 'world')))
    print('last={}'.format(r))
    print('end'.center(20, '='))
if __name__ == '__main__':
    test()