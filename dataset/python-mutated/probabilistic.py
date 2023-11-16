from nltk.internals import raise_unorderable_types
from nltk.probability import ProbabilisticMixIn
from nltk.tree.immutable import ImmutableProbabilisticTree
from nltk.tree.tree import Tree

class ProbabilisticTree(Tree, ProbabilisticMixIn):

    def __init__(self, node, children=None, **prob_kwargs):
        if False:
            for i in range(10):
                print('nop')
        Tree.__init__(self, node, children)
        ProbabilisticMixIn.__init__(self, **prob_kwargs)

    def _frozen_class(self):
        if False:
            return 10
        return ImmutableProbabilisticTree

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'{Tree.__repr__(self)} (p={self.prob()!r})'

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'{self.pformat(margin=60)} (p={self.prob():.6g})'

    def copy(self, deep=False):
        if False:
            while True:
                i = 10
        if not deep:
            return type(self)(self._label, self, prob=self.prob())
        else:
            return type(self).convert(self)

    @classmethod
    def convert(cls, val):
        if False:
            i = 10
            return i + 15
        if isinstance(val, Tree):
            children = [cls.convert(child) for child in val]
            if isinstance(val, ProbabilisticMixIn):
                return cls(val._label, children, prob=val.prob())
            else:
                return cls(val._label, children, prob=1.0)
        else:
            return val

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self.__class__ is other.__class__ and (self._label, list(self), self.prob()) == (other._label, list(other), other.prob())

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, Tree):
            raise_unorderable_types('<', self, other)
        if self.__class__ is other.__class__:
            return (self._label, list(self), self.prob()) < (other._label, list(other), other.prob())
        else:
            return self.__class__.__name__ < other.__class__.__name__
__all__ = ['ProbabilisticTree']