class A:

    def foo(self):
        if False:
            i = 10
            return i + 15
        return 1
a = A()
print(a.foo())
a.foo = lambda : 2
print(a.foo())