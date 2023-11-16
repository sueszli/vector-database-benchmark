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

class A:

    def foo(self):
        if False:
            for i in range(10):
                print('nop')
        print('A foo')
        return 42

class B(A):

    def foo(self):
        if False:
            i = 10
            return i + 15
        print('B foo')
        print(super().foo())
b = B()
micropython.heap_lock()
b.foo()
micropython.heap_unlock()