"""
Topic: 闭包访问函数内部变量
Desc : 
"""

def sample():
    if False:
        return 10
    n = 0

    def func():
        if False:
            return 10
        print('n=', n)

    def get_n():
        if False:
            while True:
                i = 10
        return n

    def set_n(value):
        if False:
            for i in range(10):
                print('nop')
        nonlocal n
        n = value
    func.get_n = get_n
    func.set_n = set_n
    return func
f = sample()
f()
f.set_n(10)
f()
import sys

class ClosureInstance:

    def __init__(self, locals=None):
        if False:
            print('Hello World!')
        if locals is None:
            locals = sys._getframe(1).f_locals
        self.__dict__.update(((key, value) for (key, value) in locals.items() if callable(value)))

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__dict__['__len__']()

def Stack():
    if False:
        while True:
            i = 10
    items = []

    def push(item):
        if False:
            return 10
        items.append(item)

    def pop():
        if False:
            return 10
        return items.pop()

    def __len__():
        if False:
            for i in range(10):
                print('nop')
        return len(items)
    return ClosureInstance()
s = Stack()
print(s)
s.push(10)
s.push(20)
print(len(s))
print(s.pop())
print(s.pop())