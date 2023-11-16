"""Some simple tests for the plugin while running scripts.
"""
import inspect

def test_trivial():
    if False:
        while True:
            i = 10
    'A trivial passing test.'
    pass

def doctest_run():
    if False:
        return 10
    'Test running a trivial script.\n\n    In [13]: run simplevars.py\n    x is: 1\n    '

def doctest_runvars():
    if False:
        for i in range(10):
            print('nop')
    'Test that variables defined in scripts get loaded correctly via %run.\n\n    In [13]: run simplevars.py\n    x is: 1\n\n    In [14]: x\n    Out[14]: 1\n    '

def doctest_ivars():
    if False:
        for i in range(10):
            print('nop')
    'Test that variables defined interactively are picked up.\n    In [5]: zz=1\n\n    In [6]: zz\n    Out[6]: 1\n    '