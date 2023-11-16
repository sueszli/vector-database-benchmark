from builtins import bytes, str, super, range, zip, round, int, pow, object
b = bytes(b'ABCD')
assert list(b) == [65, 66, 67, 68]
assert repr(b) == "b'ABCD'"
try:
    b + 'EFGH'
except TypeError:
    pass
else:
    assert False, '`bytes + str` did not raise TypeError'
try:
    bytes(b',').join(['Fred', 'Bill'])
except TypeError:
    pass
else:
    assert False, '`bytes.join([str, str])` did not raise TypeError'
s = str('ABCD')
assert s != bytes(b'ABCD')
assert isinstance(s.encode('utf-8'), bytes)
assert isinstance(b.decode('utf-8'), str)
assert repr(s) == "'ABCD'"
try:
    bytes(b'B') in s
except TypeError:
    pass
else:
    assert False, '`bytes in str` did not raise TypeError'
try:
    s.find(bytes(b'A'))
except TypeError:
    pass
else:
    assert False, '`str.find(bytes)` did not raise TypeError'

class VerboseList(list):

    def append(self, item):
        if False:
            return 10
        print('Adding an item')
        super().append(item)
for i in range(2 ** 30)[:10]:
    pass
my_iter = zip(range(3), ['a', 'b', 'c'])
assert my_iter != list(my_iter)
assert round(0.125, 2) == 0.12
z = pow(-1, 0.5)
assert isinstance(2 ** 64, int)
assert isinstance('blah', str)
assert isinstance('blah', str)

class Upper(object):

    def __init__(self, iterable):
        if False:
            return 10
        self._iter = iter(iterable)

    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        return next(self._iter).upper()

    def __iter__(self):
        if False:
            print('Hello World!')
        return self
assert list(Upper('hello')) == list('HELLO')