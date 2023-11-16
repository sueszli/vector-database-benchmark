from abc import ABC, abstractmethod
from typing import Sequence, Tuple

class Transform(ABC):
    """Rewrite apply method in subclass."""

    def apply_batch(self, inputs: Sequence[Tuple]):
        if False:
            i = 10
            return i + 15
        return tuple((self.apply(input) for input in inputs))

    @abstractmethod
    def apply(self, input: Tuple):
        if False:
            for i in range(10):
                print('nop')
        pass

    def __repr__(self):
        if False:
            while True:
                i = 10
        return self.__class__.__name__

class PseudoTransform(Transform):

    def apply(self, input: Tuple):
        if False:
            for i in range(10):
                print('nop')
        return input