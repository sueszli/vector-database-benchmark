class Descriptor:

    def __get__(self, obj, cls):
        if False:
            i = 10
            return i + 15
        print('get')
        print(type(obj) is Main)
        print(cls is Main)
        return 'result'

    def __set__(self, obj, val):
        if False:
            i = 10
            return i + 15
        print('set')
        print(type(obj) is Main)
        print(val)

    def __delete__(self, obj):
        if False:
            i = 10
            return i + 15
        print('delete')
        print(type(obj) is Main)

class Main:
    Forward = Descriptor()
m = Main()
try:
    m.__class__
except AttributeError:
    print('SKIP')
    raise SystemExit
r = m.Forward
if 'Descriptor' in repr(r.__class__):
    print('SKIP')
    raise SystemExit
print(r)
m.Forward = 'a'
del m.Forward