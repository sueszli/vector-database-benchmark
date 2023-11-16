try:
    property
except:
    print('SKIP')
    raise SystemExit

class A:

    @property
    def x(self):
        if False:
            for i in range(10):
                print('nop')
        print('A x')
        return 123

class B(A):
    pass

class C(B):
    pass

class D:
    pass

class E(C, D):
    pass
print(A().x)
print(B().x)
print(C().x)
print(E().x)

class F:
    pass
F.foo = property(lambda self: print('foo get'))

class G(F):
    pass
F().foo
G().foo
F.bar = property(lambda self: print('bar get'))
F().bar
G().bar

class H:
    pass

class I(H):
    pass
H.val = 2
print(I().val)
I.baz = property(lambda self: print('baz get'))
I().baz