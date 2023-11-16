from __future__ import print_function
import sys

def displayDict(d):
    if False:
        for i in range(10):
            print('nop')
    if '__loader__' in d:
        d = dict(d)
        d['__loader__'] = '<__loader__ removed>'
    if '__file__' in d:
        d = dict(d)
        d['__file__'] = '<__file__ removed>'
    import pprint
    return pprint.pformat(d)
counter = 1

class X:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        global counter
        self.counter = counter
        counter += 1

    def __del__(self):
        if False:
            return 10
        print('X.__del__ occurred', self.counter)

def raising(doit):
    if False:
        while True:
            i = 10
    _x = X()

    def nested():
        if False:
            while True:
                i = 10
        if doit:
            1 / 0
    try:
        return nested()
    except ZeroDivisionError:
        print('Changing closure variable value.')
        doit = 5
        raise
raising(False)

def catcher():
    if False:
        print('Hello World!')
    try:
        raising(True)
    except ZeroDivisionError:
        print('Caught.')
        print("Top traceback code is '%s'." % sys.exc_info()[2].tb_frame.f_code.co_name)
        print("Second traceback code is '%s'." % sys.exc_info()[2].tb_next.tb_frame.f_code.co_name)
        print("Third traceback code is '%s'." % sys.exc_info()[2].tb_next.tb_next.tb_frame.f_code.co_name)
        print('Third traceback locals (function) are', displayDict(sys.exc_info()[2].tb_next.tb_next.tb_frame.f_locals))
catcher()
print('Good bye.')