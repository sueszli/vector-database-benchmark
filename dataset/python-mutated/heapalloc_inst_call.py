import micropython

class Foo0:

    def __call__(self):
        if False:
            while True:
                i = 10
        print('__call__')

class Foo1:

    def __call__(self, a):
        if False:
            i = 10
            return i + 15
        print('__call__', a)

class Foo2:

    def __call__(self, a, b):
        if False:
            return 10
        print('__call__', a, b)

class Foo3:

    def __call__(self, a, b, c):
        if False:
            print('Hello World!')
        print('__call__', a, b, c)
f0 = Foo0()
f1 = Foo1()
f2 = Foo2()
f3 = Foo3()
micropython.heap_lock()
f0()
f1(1)
f2(1, 2)
f3(1, 2, 3)
micropython.heap_unlock()