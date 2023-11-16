import os
import time
import csv

def follow(filename, target):
    if False:
        for i in range(10):
            print('nop')
    with open(filename, 'r') as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if line != '':
                target.send(line)
            else:
                time.sleep(0.1)

def receive(expected_type):
    if False:
        for i in range(10):
            print('nop')
    msg = (yield)
    assert isinstance(msg, expected_type), 'Expected type %s' % expected_type
    return msg
from functools import wraps

def consumer(func):
    if False:
        print('Hello World!')

    @wraps(func)
    def start(*args, **kwargs):
        if False:
            while True:
                i = 10
        f = func(*args, **kwargs)
        f.send(None)
        return f
    return start

@consumer
def printer():
    if False:
        print('Hello World!')
    while True:
        item = (yield from receive(object))
        print(item)
if __name__ == '__main__':
    follow('../../Data/stocklog.csv', printer())