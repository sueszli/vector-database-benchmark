class A:

    class A1:

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.a1 = True

        def foo(self):
            if False:
                print('Hello World!')
            self.b = True

    def __init__(self):
        if False:
            return 10
        self.a = True

    def foo(self):
        if False:
            print('Hello World!')
        self.fooed = True

class B:

    def __init__(self):
        if False:
            print('Hello World!')
        self.bed = True

    def bar(self):
        if False:
            print('Hello World!')
        self.barred = True

class C(A, B):

    def foobar(self):
        if False:
            return 10
        self.foobared = True
c = C()
c.foo()
c.bar()
c.foobar()