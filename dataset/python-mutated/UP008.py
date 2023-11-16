class Parent:

    def method(self):
        if False:
            while True:
                i = 10
        pass

    def wrong(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class Child(Parent):

    def method(self):
        if False:
            while True:
                i = 10
        parent = super()
        super().method()
        Parent.method(self)
        Parent.super(1, 2)

    def wrong(self):
        if False:
            while True:
                i = 10
        parent = super(Child, self)
        super(Child, self).method
        super(Child, self).method()

class BaseClass:

    def f(self):
        if False:
            for i in range(10):
                print('nop')
        print('f')

def defined_outside(self):
    if False:
        while True:
            i = 10
    super(MyClass, self).f()

class MyClass(BaseClass):

    def normal(self):
        if False:
            return 10
        super(MyClass, self).f()
        super().f()

    def different_argument(self, other):
        if False:
            i = 10
            return i + 15
        super(MyClass, other).f()

    def comprehension_scope(self):
        if False:
            for i in range(10):
                print('nop')
        [super(MyClass, self).f() for x in [1]]

    def inner_functions(self):
        if False:
            i = 10
            return i + 15

        def outer_argument():
            if False:
                while True:
                    i = 10
            super(MyClass, self).f()

        def inner_argument(self):
            if False:
                while True:
                    i = 10
            super(MyClass, self).f()
            super().f()
        outer_argument()
        inner_argument(self)

    def inner_class(self):
        if False:
            print('Hello World!')

        class InnerClass:
            super(MyClass, self).f()

            def method(inner_self):
                if False:
                    for i in range(10):
                        print('nop')
                super(MyClass, self).f()
        InnerClass().method()
    defined_outside = defined_outside