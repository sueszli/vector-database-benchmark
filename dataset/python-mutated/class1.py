def go():
    if False:
        for i in range(10):
            print('nop')

    class C:

        def f():
            if False:
                for i in range(10):
                    print('nop')
            print(1)

        def g(self):
            if False:
                while True:
                    i = 10
            print(2)

        def set(self, value):
            if False:
                print('Hello World!')
            self.value = value

        def print(self):
            if False:
                print('Hello World!')
            print(self.value)
    C.f()
    C()
    C().g()
    o = C()
    o.set(3)
    o.print()
    C.set(o, 4)
    C.print(o)
go()