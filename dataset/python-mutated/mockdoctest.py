class Counter:
    """a simple counter object for testing trial's doctest support

    >>> c = Counter()
    >>> c.value()
    0
    >>> c += 3
    >>> c.value()
    3
    >>> c.incr()
    >>> c.value() == 4
    True
    >>> c == 4
    True
    >>> c != 9
    True

    """
    _count = 0

    def __init__(self, initialValue=0, maxval=None):
        if False:
            print('Hello World!')
        self._count = initialValue
        self.maxval = maxval

    def __iadd__(self, other):
        if False:
            i = 10
            return i + 15
        'add other to my value and return self\n\n        >>> c = Counter(100)\n        >>> c += 333\n        >>> c == 433\n        True\n        '
        if self.maxval is not None and self._count + other > self.maxval:
            raise ValueError('sorry, counter got too big')
        else:
            self._count += other
        return self

    def __eq__(self, other: object) -> bool:
        if False:
            i = 10
            return i + 15
        'equality operator, compare other to my value()\n\n        >>> c = Counter()\n        >>> c == 0\n        True\n        >>> c += 10\n        >>> c.incr()\n        >>> c == 10   # fail this test on purpose\n        True\n\n        '
        return self._count == other

    def __ne__(self, other: object) -> bool:
        if False:
            return 10
        'inequality operator\n\n        >>> c = Counter()\n        >>> c != 10\n        True\n        '
        return not self.__eq__(other)

    def incr(self):
        if False:
            return 10
        'increment my value by 1\n\n        >>> from twisted.trial.test.mockdoctest import Counter\n        >>> c = Counter(10, 11)\n        >>> c.incr()\n        >>> c.value() == 11\n        True\n        >>> c.incr()\n        Traceback (most recent call last):\n          File "<stdin>", line 1, in ?\n          File "twisted/trial/test/mockdoctest.py", line 51, in incr\n            self.__iadd__(1)\n          File "twisted/trial/test/mockdoctest.py", line 39, in __iadd__\n            raise ValueError, "sorry, counter got too big"\n        ValueError: sorry, counter got too big\n        '
        self.__iadd__(1)

    def value(self):
        if False:
            while True:
                i = 10
        "return this counter's value\n\n        >>> c = Counter(555)\n        >>> c.value() == 555\n        True\n        "
        return self._count

    def unexpectedException(self):
        if False:
            return 10
        "i will raise an unexpected exception...\n        ... *CAUSE THAT'S THE KINDA GUY I AM*\n\n              >>> 1/0\n        "