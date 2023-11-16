def kwfunc(a, *, k):
    if False:
        for i in range(10):
            print('nop')
    pass
print('Call function with mixed arguments with too wrong keyword argument.')
try:
    kwfunc(k=3, b=5)
except TypeError as e:
    print(repr(e))
print('Call function with mixed arguments with too little positional arguments.')
try:
    kwfunc(k=3)
except TypeError as e:
    print(repr(e))
print('Call function with mixed arguments with too little positional arguments.')
try:
    kwfunc(3)
except TypeError as e:
    print(repr(e))
print('Call function with mixed arguments with too many positional arguments.')
try:
    kwfunc(1, 2, k=3)
except TypeError as e:
    print(repr(e))

def kwfuncdefaulted(a, b=None, *, c=None):
    if False:
        for i in range(10):
            print('nop')
    pass
print('Call function with mixed arguments and defaults but too many positional arguments.')
try:
    kwfuncdefaulted(1, 2, 3)
except TypeError as e:
    print(repr(e))

def kwfunc2(a, *, k, l, m):
    if False:
        i = 10
        return i + 15
    pass
print('Call function with mixed arguments with too little positional and keyword-only arguments.')
try:
    kwfunc2(1, l=2)
except TypeError as e:
    print(repr(e))
try:
    kwfunc2(1)
except TypeError as e:
    print(repr(e))