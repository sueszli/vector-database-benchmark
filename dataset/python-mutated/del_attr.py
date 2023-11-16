class C:

    def f():
        if False:
            i = 10
            return i + 15
        pass
del C.f
try:
    print(C.x)
except AttributeError:
    print('AttributeError')
try:
    del C.f
except AttributeError:
    print('AttributeError')
c = C()
c.x = 1
print(c.x)
del c.x
try:
    print(c.x)
except AttributeError:
    print('AttributeError')
try:
    del c.x
except AttributeError:
    print('AttributeError')
try:
    del int.to_bytes
except (AttributeError, TypeError):
    print('AttributeError/TypeError')