from sympy.core.logic import fuzzy_and, fuzzy_or, fuzzy_not, fuzzy_xor

class intervalMembership:
    """Represents a boolean expression returned by the comparison of
    the interval object.

    Parameters
    ==========

    (a, b) : (bool, bool)
        The first value determines the comparison as follows:
        - True: If the comparison is True throughout the intervals.
        - False: If the comparison is False throughout the intervals.
        - None: If the comparison is True for some part of the intervals.

        The second value is determined as follows:
        - True: If both the intervals in comparison are valid.
        - False: If at least one of the intervals is False, else
        - None
    """

    def __init__(self, a, b):
        if False:
            return 10
        self._wrapped = (a, b)

    def __getitem__(self, i):
        if False:
            i = 10
            return i + 15
        try:
            return self._wrapped[i]
        except IndexError:
            raise IndexError('{} must be a valid indexing for the 2-tuple.'.format(i))

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return 2

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter(self._wrapped)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return 'intervalMembership({}, {})'.format(*self)
    __repr__ = __str__

    def __and__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, intervalMembership):
            raise ValueError('The comparison is not supported for {}.'.format(other))
        (a1, b1) = self
        (a2, b2) = other
        return intervalMembership(fuzzy_and([a1, a2]), fuzzy_and([b1, b2]))

    def __or__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, intervalMembership):
            raise ValueError('The comparison is not supported for {}.'.format(other))
        (a1, b1) = self
        (a2, b2) = other
        return intervalMembership(fuzzy_or([a1, a2]), fuzzy_and([b1, b2]))

    def __invert__(self):
        if False:
            for i in range(10):
                print('nop')
        (a, b) = self
        return intervalMembership(fuzzy_not(a), b)

    def __xor__(self, other):
        if False:
            print('Hello World!')
        if not isinstance(other, intervalMembership):
            raise ValueError('The comparison is not supported for {}.'.format(other))
        (a1, b1) = self
        (a2, b2) = other
        return intervalMembership(fuzzy_xor([a1, a2]), fuzzy_and([b1, b2]))

    def __eq__(self, other):
        if False:
            return 10
        return self._wrapped == other

    def __ne__(self, other):
        if False:
            return 10
        return self._wrapped != other