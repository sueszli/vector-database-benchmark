import micropython
import sys
try:
    import io
except ImportError:
    print('SKIP')
    raise SystemExit
global_exc = StopIteration()
try:
    raise global_exc
except:
    pass

def test():
    if False:
        for i in range(10):
            print('nop')
    micropython.heap_lock()
    global global_exc
    global_exc.__traceback__ = None
    try:
        raise global_exc
    except StopIteration:
        print('StopIteration')
    micropython.heap_unlock()
test()
buf = io.StringIO()
sys.print_exception(global_exc, buf)
for l in buf.getvalue().split('\n'):
    if l.startswith('  File '):
        l = l.split('"')
        print(l[0], l[2])
    else:
        print(l)