class A:
    var = 132

    def __init__(self):
        if False:
            return 10
        self.var2 = 34

    def meth(self, i):
        if False:
            return 10
        return 42 + i
a = A()
print(getattr(a, 'var'))
print(getattr(a, 'var2'))
print(getattr(a, 'meth')(5))
print(getattr(a, '_none_such', 123))
print(getattr(list, 'foo', 456))
print(getattr(a, 'va' + 'r2'))

class B:

    def __getattr__(self, attr):
        if False:
            return 10
        if attr == 'a':
            return attr
        else:
            raise AttributeError
b = B()
print(getattr(b, 'a'))
print(getattr(b, 'a', 'default'))
print(getattr(b, 'b', 'default'))