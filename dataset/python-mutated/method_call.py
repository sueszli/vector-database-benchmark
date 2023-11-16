class A:

    def foo(self, x):
        if False:
            while True:
                i = 10
        pass

def bar():
    if False:
        while True:
            i = 10
    A().foo(10)