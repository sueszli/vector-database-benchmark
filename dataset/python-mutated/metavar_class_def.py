class Foo:

    def foo():
        if False:
            i = 10
            return i + 15
        foo(3)

class Anything(Foo):

    def foos():
        if False:
            print('Hello World!')
        foo(4)

def foo(var):
    if False:
        while True:
            i = 10
    foo(3)