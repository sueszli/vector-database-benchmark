"""
@author: Gordeev Andrey <gordeev.and.and@gmail.com>

*TL;DR
Provides recombination business logic by chaining together using boolean logic.
"""
from abc import abstractmethod

class Specification:

    def and_specification(self, candidate):
        if False:
            return 10
        raise NotImplementedError()

    def or_specification(self, candidate):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def not_specification(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @abstractmethod
    def is_satisfied_by(self, candidate):
        if False:
            i = 10
            return i + 15
        pass

class CompositeSpecification(Specification):

    @abstractmethod
    def is_satisfied_by(self, candidate):
        if False:
            i = 10
            return i + 15
        pass

    def and_specification(self, candidate):
        if False:
            return 10
        return AndSpecification(self, candidate)

    def or_specification(self, candidate):
        if False:
            return 10
        return OrSpecification(self, candidate)

    def not_specification(self):
        if False:
            for i in range(10):
                print('nop')
        return NotSpecification(self)

class AndSpecification(CompositeSpecification):

    def __init__(self, one, other):
        if False:
            return 10
        self._one: Specification = one
        self._other: Specification = other

    def is_satisfied_by(self, candidate):
        if False:
            while True:
                i = 10
        return bool(self._one.is_satisfied_by(candidate) and self._other.is_satisfied_by(candidate))

class OrSpecification(CompositeSpecification):

    def __init__(self, one, other):
        if False:
            i = 10
            return i + 15
        self._one: Specification = one
        self._other: Specification = other

    def is_satisfied_by(self, candidate):
        if False:
            return 10
        return bool(self._one.is_satisfied_by(candidate) or self._other.is_satisfied_by(candidate))

class NotSpecification(CompositeSpecification):

    def __init__(self, wrapped):
        if False:
            i = 10
            return i + 15
        self._wrapped: Specification = wrapped

    def is_satisfied_by(self, candidate):
        if False:
            for i in range(10):
                print('nop')
        return bool(not self._wrapped.is_satisfied_by(candidate))

class User:

    def __init__(self, super_user=False):
        if False:
            while True:
                i = 10
        self.super_user = super_user

class UserSpecification(CompositeSpecification):

    def is_satisfied_by(self, candidate):
        if False:
            while True:
                i = 10
        return isinstance(candidate, User)

class SuperUserSpecification(CompositeSpecification):

    def is_satisfied_by(self, candidate):
        if False:
            while True:
                i = 10
        return getattr(candidate, 'super_user', False)

def main():
    if False:
        return 10
    "\n    >>> andrey = User()\n    >>> ivan = User(super_user=True)\n    >>> vasiliy = 'not User instance'\n\n    >>> root_specification = UserSpecification().and_specification(SuperUserSpecification())\n\n    # Is specification satisfied by <name>\n    >>> root_specification.is_satisfied_by(andrey), 'andrey'\n    (False, 'andrey')\n    >>> root_specification.is_satisfied_by(ivan), 'ivan'\n    (True, 'ivan')\n    >>> root_specification.is_satisfied_by(vasiliy), 'vasiliy'\n    (False, 'vasiliy')\n    "
if __name__ == '__main__':
    import doctest
    doctest.testmod()