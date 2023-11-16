from __future__ import print_function
try:
    from __builtin__ import len as _len
except ImportError:
    from builtins import len as _len

def len(x):
    if False:
        for i in range(10):
            print('nop')
    print('Private built-in called with argument', repr(x))
    return _len(x)
print('Calling built-in len', len(range(9)))