try:
    import array
    import struct
except ImportError:
    print('SKIP')
    raise SystemExit
try:
    exec('def f(): super()')
except SyntaxError:
    print('SyntaxError')
try:
    ValueError().x = 0
except AttributeError:
    print('AttributeError')
try:
    a = array.array('b', (1, 2, 3))
    del a[1]
except TypeError:
    print('TypeError')
try:
    a = array.array('b', (1, 2, 3))
    print(a[3:2:2])
except NotImplementedError:
    print('NotImplementedError')
try:
    print(1 in array.array('B', b'12'))
except NotImplementedError:
    print('NotImplementedError')
try:
    '%c' % b'\x01\x02'
except (TypeError, ValueError):
    print('TypeError, ValueError')
try:
    print('{a[0]}'.format(a=[1, 2]))
except NotImplementedError:
    print('NotImplementedError')
try:
    str(b'abc', encoding='utf8')
except NotImplementedError:
    print('NotImplementedError')
try:
    'a a a'.rsplit(None, 1)
except NotImplementedError:
    print('NotImplementedError')
try:
    'abc'.endswith('c', 1)
except NotImplementedError:
    print('NotImplementedError')
try:
    print('abc'[1:2:3])
except NotImplementedError:
    print('NotImplementedError')
try:
    bytes('abc', encoding='utf8')
except NotImplementedError:
    print('NotImplementedError')
try:
    b'123'[0:3:2]
except NotImplementedError:
    print('NotImplementedError')
try:
    ()[2:3:4]
except NotImplementedError:
    print('NotImplementedError')
try:
    [][2:3:4] = []
except NotImplementedError:
    print('NotImplementedError')
try:
    del [][2:3:4]
except NotImplementedError:
    print('NotImplementedError')
print(struct.pack('bb', 1, 2, 3))
print(struct.pack('bb', 1))
try:
    bytearray(4)[0:1] = [1, 2]
except NotImplementedError:
    print('NotImplementedError')

def f():
    if False:
        while True:
            i = 10
    pass
try:
    f.x = 1
except AttributeError:
    print('AttributeError')
try:
    type(f)()
except TypeError:
    print('TypeError')

class A:

    def foo(self):
        if False:
            for i in range(10):
                print('nop')
        print('A.foo')

class B(object, A):
    pass
B().foo()

class A:
    pass

class B(A):
    pass
try:
    A.bar = property()
except AttributeError:
    print('AttributeError')