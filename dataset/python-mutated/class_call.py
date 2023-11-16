class C1:

    def __call__(self, val):
        if False:
            return 10
        print('call', val)
        return 'item'

class C2:

    def __getattr__(self, k):
        if False:
            i = 10
            return i + 15
        pass
c1 = C1()
print(c1(1))
c2 = C2()
try:
    print(c2(1))
except TypeError:
    print('TypeError')