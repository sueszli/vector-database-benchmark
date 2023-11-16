class Foo:

    def __radd__(self, other):
        if False:
            while True:
                i = 10
        pass
try:
    5 + Foo()
except TypeError:
    print('TypeError')