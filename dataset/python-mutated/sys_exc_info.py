import sys
try:
    sys.exc_info
except:
    print('SKIP')
    raise SystemExit

def f():
    if False:
        return 10
    print(sys.exc_info()[0:2])
try:
    raise ValueError('value', 123)
except:
    print(sys.exc_info()[0:2])
    f()
f()