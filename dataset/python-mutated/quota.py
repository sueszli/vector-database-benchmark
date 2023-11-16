from __future__ import absolute_import

class Quota(object):
    """An upper or lower bound for metrics"""

    def __init__(self, bound, is_upper):
        if False:
            print('Hello World!')
        self._bound = bound
        self._upper = is_upper

    @staticmethod
    def upper_bound(upper_bound):
        if False:
            i = 10
            return i + 15
        return Quota(upper_bound, True)

    @staticmethod
    def lower_bound(lower_bound):
        if False:
            print('Hello World!')
        return Quota(lower_bound, False)

    def is_upper_bound(self):
        if False:
            print('Hello World!')
        return self._upper

    @property
    def bound(self):
        if False:
            i = 10
            return i + 15
        return self._bound

    def is_acceptable(self, value):
        if False:
            for i in range(10):
                print('nop')
        return self.is_upper_bound() and value <= self.bound or (not self.is_upper_bound() and value >= self.bound)

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        prime = 31
        result = prime + self.bound
        return prime * result + self.is_upper_bound()

    def __eq__(self, other):
        if False:
            return 10
        if self is other:
            return True
        return type(self) == type(other) and self.bound == other.bound and (self.is_upper_bound() == other.is_upper_bound())

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        return not self.__eq__(other)