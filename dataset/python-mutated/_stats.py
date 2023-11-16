import collections
import functools
from typing import OrderedDict
simple_call_counter: OrderedDict[str, int] = collections.OrderedDict()

def count_label(label):
    if False:
        for i in range(10):
            print('nop')
    prev = simple_call_counter.setdefault(label, 0)
    simple_call_counter[label] = prev + 1

def count(fn):
    if False:
        print('Hello World!')

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        if fn.__qualname__ not in simple_call_counter:
            simple_call_counter[fn.__qualname__] = 0
        simple_call_counter[fn.__qualname__] = simple_call_counter[fn.__qualname__] + 1
        return fn(*args, **kwargs)
    return wrapper