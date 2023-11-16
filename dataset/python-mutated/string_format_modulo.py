try:
    '' % ()
except TypeError:
    print('SKIP')
    raise SystemExit
print('%%' % ())
print('=%s=' % 1)
print('=%s=%s=' % (1, 2))
print('=%s=' % (1,))
print('=%s=' % [1, 2])
print('=%s=' % 'str')
print('=%r=' % 'str')

class A:

    def __int__(self):
        if False:
            i = 10
            return i + 15
        return 123
print('%d' % A())
try:
    print('=%s=%s=' % 1)
except TypeError:
    print('TypeError')
try:
    print('=%s=%s=%s=' % (1, 2))
except TypeError:
    print('TypeError')
try:
    print('=%s=' % (1, 2))
except TypeError:
    print('TypeError')
print('%s' % True)
print('%s' % 1)
print('%.1s' % 'ab')
print('%r' % True)
print('%r' % 1)
print('%c' % 48)
print('%c' % 'a')
print('%10s' % 'abc')
print('%-10s' % 'abc')
print('%c' % False)
print('%c' % True)
print('%s' % {})
print('%s' % ({},))
print('foo' % {})
try:
    print('%*s' % 5)
except TypeError:
    print('TypeError')
try:
    print('%*.*s' % (1, 15))
except TypeError:
    print('TypeError')
print('%(foo)s' % {'foo': 'bar', 'baz': False})
print('%s %(foo)s %(foo)s' % {'foo': 1})
try:
    print('%(foo)s %s %(foo)s' % {'foo': 1})
except TypeError:
    print('TypeError')
try:
    print('%(foo)s' % {})
except KeyError:
    print('KeyError')
try:
    print('%(foo)*s' % {'foo': 'bar'})
except TypeError:
    print('TypeError')
try:
    '%(foo)s' % 1
except TypeError:
    print('TypeError')
try:
    '%(foo)s' % ({},)
except TypeError:
    print('TypeError')
try:
    '%(a' % {'a': 1}
except ValueError:
    print('ValueError')
try:
    '%.*d %.*d' % (20, 5)
except TypeError:
    print('TypeError')
try:
    a = '%*' % 1
except ValueError:
    print('ValueError')
try:
    '%c' % 'aa'
except TypeError:
    print('TypeError')
try:
    '%l' % 1
except ValueError:
    print('ValueError')
try:
    'a%' % 1
except ValueError:
    print('ValueError')