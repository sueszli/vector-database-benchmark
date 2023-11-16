class A:

    def __init__(self, d):
        if False:
            i = 10
            return i + 15
        self.d = d

    def __getattr__(self, attr):
        if False:
            print('Hello World!')
        return self.d[attr]
a = A({'a': 1, 'b': 2})
print(a.a, a.b)

class A:

    def __getattr__(self, attr):
        if False:
            print('Hello World!')
        if attr == 'value':
            raise ValueError(123)
        else:
            raise AttributeError(456)
a = A()
try:
    a.value
except ValueError as er:
    print(er)
try:
    a.attr
except AttributeError as er:
    print(er)