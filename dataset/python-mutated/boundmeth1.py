print(type(repr([].append)))

class A:

    def f(self):
        if False:
            return 10
        return 0

    def g(self, a):
        if False:
            print('Hello World!')
        return a

    def h(self, a, b, c, d, e, f):
        if False:
            i = 10
            return i + 15
        return a + b + c + d + e + f
m = A().f
print(m())
m = A().g
print(m(1))
m = A().h
print(m(1, 2, 3, 4, 5, 6))
try:
    A().f.x = 1
except AttributeError:
    print('AttributeError')
print('--------')
a = A()
m1 = a.f
m2 = a.f
print(m1 == a.f)
print(m2 == a.f)
print(m1 == m2)
print(m1 != a.f)
a1 = A()
a2 = A()
m1 = a1.f
m2 = a2.f
print(m1 == a2.f)
print(m2 == a1.f)
print(m1 != a2.f)
print(A().f == None)
print(A().f != None)
print(None == A().f)
print(None != A().f)
print('--------')
a = A()
m1 = a.f
m2 = a.f
print(hash(m1) == hash(a.f))
print(hash(m2) == hash(a.f))
print(hash(m1) == hash(m2))
print(hash(m1) != hash(a.g))
a2 = A()
m2 = a2.f
print(hash(m1) == hash(a2.f))