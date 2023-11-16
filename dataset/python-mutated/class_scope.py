class MethodRedef(object):
    """
    >>> MethodRedef().a(5)
    7
    """

    def a(self, i):
        if False:
            return 10
        return i + 1

    def a(self, i):
        if False:
            i = 10
            return i + 15
        return i + 2