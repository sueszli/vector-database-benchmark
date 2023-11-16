"""
Topic: sample
Desc : 
"""
from collections import Iterable

def flatten(items, ignore_types=(str, bytes)):
    if False:
        i = 10
        return i + 15
    for x in items:
        if isinstance(x, Iterable) and (not isinstance(x, ignore_types)):
            yield from flatten(x)
        else:
            yield x

def flatten_seq():
    if False:
        print('Hello World!')
    items = [1, 2, [3, 4, [5, 6], 7], 8]
    for x in flatten(items):
        print(x)
    items = ['Dave', 'Paula', ['Thomas', 'Lewis']]
    for x in flatten(items):
        print(x)
if __name__ == '__main__':
    flatten_seq()