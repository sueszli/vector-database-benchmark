from hypothesis.internal.reflection import source_exec_as_module

def test_can_eval_as_source():
    if False:
        for i in range(10):
            print('nop')
    assert source_exec_as_module('foo=1').foo == 1

def test_caches():
    if False:
        print('Hello World!')
    x = source_exec_as_module('foo=2')
    y = source_exec_as_module('foo=2')
    assert x is y
RECURSIVE = '\nfrom hypothesis.internal.reflection import source_exec_as_module\n\ndef test_recurse():\n    assert not (\n        source_exec_as_module("too_much_recursion = False").too_much_recursion)\n'

def test_can_call_self_recursively():
    if False:
        print('Hello World!')
    source_exec_as_module(RECURSIVE).test_recurse()