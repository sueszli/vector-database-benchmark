class TwoNames:
    """f() and g() are two names for the same method"""

    def f(self):
        if False:
            i = 10
            return i + 15
        '\n        >>> print(TwoNames().f())\n        f\n        '
        return 'f'
    g = f