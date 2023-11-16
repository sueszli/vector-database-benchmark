from __future__ import print_function
import cython

@cython.locals(d=dict)
def test_dict(d):
    if False:
        return 10
    '\n    >>> test_dict({})\n    dict\n    {}\n    '
    print(d.__class__.__name__)
    print(d.__class__())

@cython.locals(i=int)
def test_int(i):
    if False:
        while True:
            i = 10
    '\n    >>> test_int(0)\n    int\n    0\n    '
    print(i.__class__.__name__)
    print(i.__class__())

@cython.cclass
class C:

    def __str__(self):
        if False:
            while True:
                i = 10
        return "I'm a C object"

@cython.locals(c=C)
def test_cdef_class(c):
    if False:
        while True:
            i = 10
    "\n    # This wasn't actually broken but is worth testing anyway\n    >>> test_cdef_class(C())\n    C\n    I'm a C object\n    "
    print(c.__class__.__name__)
    print(c.__class__())

@cython.locals(d=object)
def test_object(o):
    if False:
        for i in range(10):
            print('nop')
    "\n    >>> test_object({})\n    dict\n    {}\n    >>> test_object(1)\n    int\n    0\n    >>> test_object(C())\n    C\n    I'm a C object\n    "
    print(o.__class__.__name__)
    print(o.__class__())