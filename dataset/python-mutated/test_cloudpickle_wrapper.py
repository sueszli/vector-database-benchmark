"""
Test that our implementation of wrap_non_picklable_objects mimics
properly the loky implementation.
"""
from .._cloudpickle_wrapper import wrap_non_picklable_objects
from .._cloudpickle_wrapper import _my_wrap_non_picklable_objects

def a_function(x):
    if False:
        print('Hello World!')
    return x

class AClass(object):

    def __call__(self, x):
        if False:
            while True:
                i = 10
        return x

def test_wrap_non_picklable_objects():
    if False:
        i = 10
        return i + 15
    for obj in (a_function, AClass()):
        wrapped_obj = wrap_non_picklable_objects(obj)
        my_wrapped_obj = _my_wrap_non_picklable_objects(obj)
        assert wrapped_obj(1) == my_wrapped_obj(1)