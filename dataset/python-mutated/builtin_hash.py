print(hash(False))
print(hash(True))
print({(): 1})
print({(1,): 1})
print(hash in {hash: 1})
print(type(hash(list.pop)))
print(type(hash([].pop)))
print(type(hash(object())))
print(type(hash(super(object, object))))
print(type(hash(classmethod(hash))))
print(type(hash(staticmethod(hash))))
print(type(hash(iter(''))))
print(type(hash(iter(b''))))
print(type(hash(iter(range(0)))))
print(type(hash(map(None, []))))
print(type(hash(zip([]))))

def f(x):
    if False:
        i = 10
        return i + 15

    def g():
        if False:
            for i in range(10):
                print('nop')
        return x
    return g
print(type(hash(f(1))))
try:
    hash([])
except TypeError:
    print('TypeError')

class A:

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return 123

    def __repr__(self):
        if False:
            return 10
        return 'a instance'
print(hash(A()))
print({A(): 1})

class B:
    pass
hash(B())

class C:

    def __eq__(self, another):
        if False:
            i = 10
            return i + 15
        return True
try:
    hash(C())
except TypeError:
    print('TypeError')

class D:

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return None
try:
    hash(D())
except TypeError:
    print('TypeError')

class E:

    def __hash__(self):
        if False:
            while True:
                i = 10
        return True
print(hash(E()))