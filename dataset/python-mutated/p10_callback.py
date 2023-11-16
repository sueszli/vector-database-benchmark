"""
Topic: 带状态值的回调函数
Desc : 
"""

def apply_async(func, args, *, callback):
    if False:
        return 10
    result = func(*args)
    callback(result)

def add(x, y):
    if False:
        i = 10
        return i + 15
    return x + y

class ResultHandler:

    def __init__(self):
        if False:
            return 10
        self.sequence = 0

    def handler(self, result):
        if False:
            i = 10
            return i + 15
        self.sequence += 1
        print('[{}] Got: {}'.format(self.sequence, result))
r = ResultHandler()
apply_async(add, (2, 3), callback=r.handler)
apply_async(add, ('hello', 'world'), callback=r.handler)

def make_handler():
    if False:
        return 10
    sequence = 0

    def handler(result):
        if False:
            i = 10
            return i + 15
        nonlocal sequence
        sequence += 1
        print('[{}] Got: {}'.format(sequence, result))
    return handler
handler = make_handler()
apply_async(add, (2, 3), callback=handler)
apply_async(add, ('hello', 'world'), callback=handler)

def make_handler():
    if False:
        i = 10
        return i + 15
    sequence = 0
    while True:
        result = (yield)
        sequence += 1
        print('[{}] Got: {}'.format(sequence, result))
handler = make_handler()
next(handler)
apply_async(add, (2, 3), callback=handler.send)
apply_async(add, ('hello', 'world'), callback=handler.send)