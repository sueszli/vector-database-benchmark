def f():
    if False:
        print('Hello World!')

    class A:
        pass

def g():
    if False:
        i = 10
        return i + 15

    class B:
        pass
    return B

def h(base):
    if False:
        print('Hello World!')

    class C(base):
        pass

    class D(C):
        pass
    return D()

class Outer:

    def method(self):
        if False:
            while True:
                i = 10

        class NestedClass:

            def fn():
                if False:
                    while True:
                        i = 10
                pass

        def nested_fn():
            if False:
                while True:
                    i = 10
            pass