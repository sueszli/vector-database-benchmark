try:
    import io
    import sys
except ImportError:
    print('SKIP')
    raise SystemExit
if hasattr(sys, 'print_exception'):
    print_exception = sys.print_exception
else:
    import traceback
    print_exception = lambda e, f: traceback.print_exception(None, e, sys.exc_info()[2], file=f)

def print_exc(e):
    if False:
        print('Hello World!')
    buf = io.StringIO()
    print_exception(e, buf)
    s = buf.getvalue()
    for l in s.split('\n'):
        if l.startswith('  File '):
            l = l.split('"')
            print(l[0], l[2])
        elif not l.startswith('    '):
            print(l)
try:
    raise Exception('msg')
except Exception as e:
    print('caught')
    print_exc(e)

def f():
    if False:
        while True:
            i = 10
    g()

def g():
    if False:
        return 10
    raise Exception('fail')
try:
    f()
except Exception as e:
    print('caught')
    print_exc(e)
try:
    try:
        f()
    finally:
        print('finally')
except Exception as e:
    print('caught')
    print_exc(e)
try:
    try:
        f()
    except Exception as e:
        print('reraise')
        print_exc(e)
        raise
except Exception as e:
    print('caught')
    print_exc(e)

def f():
    if False:
        i = 10
        return i + 15
    f([1, 2], [1, 2], [1, 2], {1: 1, 1: 1, 1: 1, 1: 1, 1: 1, 1: 1, 1: f.X})
    return 1
try:
    f()
except Exception as e:
    print_exc(e)
if hasattr(sys, 'print_exception'):
    try:
        sys.print_exception(Exception, 1)
        had_exception = False
    except OSError:
        had_exception = True
    assert had_exception