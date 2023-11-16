"""
categories: Types,float
description: uPy allows implicit conversion of objects in maths operations while CPython does not.
cause: Unknown
workaround: Objects should be wrapped in ``float(obj)`` for compatibility with CPython.
"""

class Test:

    def __float__(self):
        if False:
            print('Hello World!')
        return 0.5
print(2.0 * Test())