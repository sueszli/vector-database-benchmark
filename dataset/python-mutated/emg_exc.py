import micropython
import sys
try:
    import io
except ImportError:
    print('SKIP')
    raise SystemExit
try:
    micropython.alloc_emergency_exception_buf(256)
except AttributeError:
    pass

def f():
    if False:
        i = 10
        return i + 15
    micropython.heap_lock()
    try:
        raise ValueError(1)
    except ValueError as er:
        exc = er
    micropython.heap_unlock()
    buf = io.StringIO()
    sys.print_exception(exc, buf)
    for l in buf.getvalue().split('\n'):
        if l.startswith('  File '):
            print(l.split('"')[2])
        else:
            print(l)
f()