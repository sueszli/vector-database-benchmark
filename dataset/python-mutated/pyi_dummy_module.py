"""
pyi_dummy_module

This module exists only so it can be imported and its __file__ inspected in the test `test_compiled_filenames`.
"""

def dummy():
    if False:
        print('Hello World!')
    pass

class DummyClass(object):

    def dummyMethod(self):
        if False:
            i = 10
            return i + 15
        pass