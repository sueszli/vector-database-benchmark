import cython

@cython.test_assert_path_exists('//ForFromStatNode')
@cython.test_fail_if_path_exists('//ForInStatNode')
@cython.locals(x=bytearray)
def basic_bytearray_iter(x):
    if False:
        print('Hello World!')
    '\n    >>> basic_bytearray_iter(bytearray(b"hello"))\n    h\n    e\n    l\n    l\n    o\n    '
    for a in x:
        print(chr(a))

@cython.test_assert_path_exists('//ForFromStatNode')
@cython.test_fail_if_path_exists('//ForInStatNode')
@cython.locals(x=bytearray)
def reversed_bytearray_iter(x):
    if False:
        i = 10
        return i + 15
    '\n    >>> reversed_bytearray_iter(bytearray(b"hello"))\n    o\n    l\n    l\n    e\n    h\n    '
    for a in reversed(x):
        print(chr(a))

@cython.test_assert_path_exists('//ForFromStatNode')
@cython.test_fail_if_path_exists('//ForInStatNode')
@cython.locals(x=bytearray)
def modifying_bytearray_iter1(x):
    if False:
        print('Hello World!')
    '\n    >>> modifying_bytearray_iter1(bytearray(b"abcdef"))\n    a\n    b\n    c\n    3\n    '
    count = 0
    for a in x:
        print(chr(a))
        del x[-1]
        count += 1
    print(count)

@cython.test_assert_path_exists('//ForFromStatNode')
@cython.test_fail_if_path_exists('//ForInStatNode')
@cython.locals(x=bytearray)
def modifying_bytearray_iter2(x):
    if False:
        return 10
    '\n    >>> modifying_bytearray_iter2(bytearray(b"abcdef"))\n    a\n    c\n    e\n    3\n    '
    count = 0
    for a in x:
        print(chr(a))
        del x[0]
        count += 1
    print(count)

@cython.test_assert_path_exists('//ForFromStatNode')
@cython.test_fail_if_path_exists('//ForInStatNode')
@cython.locals(x=bytearray)
def modifying_reversed_bytearray_iter(x):
    if False:
        for i in range(10):
            print('nop')
    '\n    NOTE - I\'m not 100% sure how well-defined this behaviour is in Python.\n    However, for the moment Python and Cython seem to do the same thing.\n    Testing that it doesn\'t crash is probably more important than the exact output!\n    >>> modifying_reversed_bytearray_iter(bytearray(b"abcdef"))\n    f\n    f\n    f\n    f\n    f\n    f\n    '
    for a in reversed(x):
        print(chr(a))
        del x[0]

def test_bytearray_iteration(src):
    if False:
        print('Hello World!')
    "\n    >>> src = b'123'\n    >>> test_bytearray_iteration(src)\n    49\n    50\n    51\n    "
    data = bytearray(src)
    for elem in data:
        print(elem)