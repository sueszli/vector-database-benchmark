class LenLibrary:
    """Library with default zero __len__.

    Example:

    >>> l = LenLibrary()
    >>> assert not l
    >>> l.set_length(1)
    >>> assert l
    """

    def __init__(self):
        if False:
            return 10
        self._length = 0

    def __len__(self):
        if False:
            while True:
                i = 10
        return self._length

    def set_length(self, length):
        if False:
            return 10
        self._length = int(length)