import cython

class Unhashable(object):

    def __hash__(self):
        if False:
            print('Hello World!')
        raise TypeError('I am not hashable')

class Hashable(object):

    def __hash__(self):
        if False:
            print('Hello World!')
        return 1

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(other, Hashable)

class CountedHashable(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.hash_count = 0
        self.eq_count = 0

    def __hash__(self):
        if False:
            print('Hello World!')
        self.hash_count += 1
        return 42

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        self.eq_count += 1
        return id(self) == id(other)

@cython.test_fail_if_path_exists('//AttributeNode')
@cython.test_assert_path_exists('//PythonCapiCallNode')
@cython.locals(d=dict)
def setdefault1(d, key):
    if False:
        print('Hello World!')
    "\n    >>> d = {}\n    >>> setdefault1(d, 1)\n    >>> len(d)\n    1\n    >>> setdefault1(d, 1)\n    >>> len(d)\n    1\n    >>> d[1]\n    >>> setdefault1(d, Unhashable())\n    Traceback (most recent call last):\n    TypeError: I am not hashable\n    >>> len(d)\n    1\n    >>> h1 = setdefault1(d, Hashable())\n    >>> len(d)\n    2\n    >>> h2 = setdefault1(d, Hashable())\n    >>> len(d)\n    2\n    >>> d[Hashable()]\n\n    # CPython's behaviour depends on version and py_debug setting, so just compare to it\n    >>> py_hashed1 = CountedHashable()\n    >>> y = {py_hashed1: 5}\n    >>> py_hashed2 = CountedHashable()\n    >>> y.setdefault(py_hashed2)\n\n    >>> cy_hashed1 = CountedHashable()\n    >>> y = {cy_hashed1: 5}\n    >>> cy_hashed2 = CountedHashable()\n    >>> setdefault1(y, cy_hashed2)\n    >>> py_hashed1.hash_count - cy_hashed1.hash_count\n    0\n    >>> py_hashed2.hash_count - cy_hashed2.hash_count\n    0\n    >>> (py_hashed1.eq_count + py_hashed2.eq_count) - (cy_hashed1.eq_count + cy_hashed2.eq_count)\n    0\n    "
    return d.setdefault(key)

@cython.test_fail_if_path_exists('//AttributeNode')
@cython.test_assert_path_exists('//PythonCapiCallNode')
@cython.locals(d=dict)
def setdefault2(d, key, value):
    if False:
        i = 10
        return i + 15
    "\n    >>> d = {}\n    >>> setdefault2(d, 1, 2)\n    2\n    >>> len(d)\n    1\n    >>> setdefault2(d, 1, 2)\n    2\n    >>> len(d)\n    1\n    >>> l = setdefault2(d, 2, [])\n    >>> len(d)\n    2\n    >>> l.append(1)\n    >>> setdefault2(d, 2, [])\n    [1]\n    >>> len(d)\n    2\n    >>> setdefault2(d, Unhashable(), 1)\n    Traceback (most recent call last):\n    TypeError: I am not hashable\n    >>> h1 = setdefault2(d, Hashable(), 55)\n    >>> len(d)\n    3\n    >>> h2 = setdefault2(d, Hashable(), 66)\n    >>> len(d)\n    3\n    >>> d[Hashable()]\n    55\n\n    # CPython's behaviour depends on version and py_debug setting, so just compare to it\n    >>> py_hashed1 = CountedHashable()\n    >>> y = {py_hashed1: 5}\n    >>> py_hashed2 = CountedHashable()\n    >>> y.setdefault(py_hashed2, [])\n    []\n\n    >>> cy_hashed1 = CountedHashable()\n    >>> y = {cy_hashed1: 5}\n    >>> cy_hashed2 = CountedHashable()\n    >>> setdefault2(y, cy_hashed2, [])\n    []\n    >>> py_hashed1.hash_count - cy_hashed1.hash_count\n    0\n    >>> py_hashed2.hash_count - cy_hashed2.hash_count\n    0\n    >>> (py_hashed1.eq_count + py_hashed2.eq_count) - (cy_hashed1.eq_count + cy_hashed2.eq_count)\n    0\n    "
    return d.setdefault(key, value)