import micropython
try:

    def stackless():
        if False:
            return 10
        pass
    micropython.heap_lock()
    stackless()
    micropython.heap_unlock()
except RuntimeError:
    print('SKIP')
    raise SystemExit
try:
    micropython.alloc_emergency_exception_buf(256)
except AttributeError:
    pass

def main():
    if False:
        print('Hello World!')
    micropython.heap_lock()
    e = Exception(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    micropython.heap_unlock()
    print(repr(e))

    def f():
        if False:
            return 10
        pass
    micropython.heap_lock()
    try:
        f(abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz=1)
    except Exception as er:
        e = er
    micropython.heap_unlock()
    print(repr(e)[:10])
    lst = []
    while 1:
        try:
            lst = [lst]
        except MemoryError:
            break
    try:
        f(abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz=1)
    except Exception as er:
        e = er
    while lst:
        (lst[0], lst) = (None, lst[0])
    print(repr(e)[:10])

    def g():
        if False:
            i = 10
            return i + 15
        g()
    micropython.heap_lock()
    try:
        g()
    except Exception as er:
        e = er
    micropython.heap_unlock()
    print(repr(e)[:13])
    exc = Exception('my exception')
    try:
        raise exc
    except:
        pass

    def h(e):
        if False:
            print('Hello World!')
        raise e
    micropython.heap_lock()
    try:
        h(exc)
    except Exception as er:
        e = er
    micropython.heap_unlock()
    print(repr(e))
main()