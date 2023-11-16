class C:

    def __getitem__(self, item):
        if False:
            for i in range(10):
                print('nop')
        print('get', item)
        return 'item'

    def __setitem__(self, item, value):
        if False:
            for i in range(10):
                print('nop')
        print('set', item, value)

    def __delitem__(self, item):
        if False:
            print('Hello World!')
        print('del', item)
c = C()
print(c[1])
c[1] = 2
del c[3]

class A:
    pass
a = A()
try:
    a[1]
except TypeError:
    print('TypeError')