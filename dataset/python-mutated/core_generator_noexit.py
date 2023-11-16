"""
categories: Core,Generator
description: Context manager __exit__() not called in a generator which does not run to completion
cause: Unknown
workaround: Unknown
"""

class foo(object):

    def __enter__(self):
        if False:
            return 10
        print('Enter')

    def __exit__(self, *args):
        if False:
            i = 10
            return i + 15
        print('Exit')

def bar(x):
    if False:
        while True:
            i = 10
    with foo():
        while True:
            x += 1
            yield x

def func():
    if False:
        i = 10
        return i + 15
    g = bar(0)
    for _ in range(3):
        print(next(g))
func()