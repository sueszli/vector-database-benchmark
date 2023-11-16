from __future__ import print_function
import cython

@cython.cfunc
@cython.returns(cython.int)
def zero():
    if False:
        print('Hello World!')
    print('In zero')
    return 0

@cython.cfunc
@cython.returns(cython.int)
def five():
    if False:
        while True:
            i = 10
    print('In five')
    return 5

@cython.cfunc
@cython.returns(cython.int)
def one():
    if False:
        print('Hello World!')
    print('In one')
    return 1

@cython.test_assert_path_exists('//ForFromStatNode')
def genexp_array_slice_order():
    if False:
        return 10
    '\n    >>> list(genexp_array_slice_order())\n    In zero\n    In five\n    [0, 1, 2, 3, 4]\n    '
    x = cython.declare(cython.int[20])
    x = list(range(20))
    return (a for a in x[zero():five()])

@cython.test_assert_path_exists('//ForFromStatNode')
@cython.test_assert_path_exists('//InlinedGeneratorExpressionNode', '//ComprehensionAppendNode')
def list_array_slice_order():
    if False:
        while True:
            i = 10
    '\n    >>> list(list_array_slice_order())\n    In zero\n    In five\n    [0, 1, 2, 3, 4]\n    '
    x = cython.declare(cython.int[20])
    x = list(range(20))
    return list((a for a in x[zero():five()]))

class IndexableClass:

    def __getitem__(self, idx):
        if False:
            return 10
        print('In indexer')
        return [idx.start, idx.stop, idx.step]

class NoisyAttributeLookup:

    @property
    def indexer(self):
        if False:
            for i in range(10):
                print('nop')
        print('Getting indexer')
        return IndexableClass()

    @property
    def function(self):
        if False:
            while True:
                i = 10
        print('Getting function')

        def func(a, b, c):
            if False:
                i = 10
                return i + 15
            print('In func')
            return [a, b, c]
        return func

def genexp_index_order():
    if False:
        while True:
            i = 10
    '\n    >>> list(genexp_index_order())\n    Getting indexer\n    In zero\n    In five\n    In one\n    In indexer\n    Made generator expression\n    [0, 5, 1]\n    '
    obj = NoisyAttributeLookup()
    ret = (a for a in obj.indexer[zero():five():one()])
    print('Made generator expression')
    return ret

@cython.test_assert_path_exists('//InlinedGeneratorExpressionNode')
def list_index_order():
    if False:
        i = 10
        return i + 15
    '\n    >>> list_index_order()\n    Getting indexer\n    In zero\n    In five\n    In one\n    In indexer\n    [0, 5, 1]\n    '
    obj = NoisyAttributeLookup()
    return list((a for a in obj.indexer[zero():five():one()]))

def genexpr_fcall_order():
    if False:
        i = 10
        return i + 15
    '\n    >>> list(genexpr_fcall_order())\n    Getting function\n    In zero\n    In five\n    In one\n    In func\n    Made generator expression\n    [0, 5, 1]\n    '
    obj = NoisyAttributeLookup()
    ret = (a for a in obj.function(zero(), five(), one()))
    print('Made generator expression')
    return ret

@cython.test_assert_path_exists('//InlinedGeneratorExpressionNode')
def list_fcall_order():
    if False:
        while True:
            i = 10
    '\n    >>> list_fcall_order()\n    Getting function\n    In zero\n    In five\n    In one\n    In func\n    [0, 5, 1]\n    '
    obj = NoisyAttributeLookup()
    return list((a for a in obj.function(zero(), five(), one())))

def call1():
    if False:
        i = 10
        return i + 15
    print('In call1')
    return ['a']

def call2():
    if False:
        i = 10
        return i + 15
    print('In call2')
    return ['b']

def multiple_genexps_to_call_order():
    if False:
        while True:
            i = 10
    '\n    >>> multiple_genexps_to_call_order()\n    In call1\n    In call2\n    '

    def takes_two_genexps(a, b):
        if False:
            i = 10
            return i + 15
        pass
    return takes_two_genexps((x for x in call1()), (x for x in call2()))