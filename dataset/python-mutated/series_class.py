"""
Contains the base class for series
Made using sequences in mind
"""
from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.cache import cacheit

class SeriesBase(Expr):
    """Base Class for series"""

    @property
    def interval(self):
        if False:
            return 10
        'The interval on which the series is defined'
        raise NotImplementedError('(%s).interval' % self)

    @property
    def start(self):
        if False:
            return 10
        'The starting point of the series. This point is included'
        raise NotImplementedError('(%s).start' % self)

    @property
    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        'The ending point of the series. This point is included'
        raise NotImplementedError('(%s).stop' % self)

    @property
    def length(self):
        if False:
            for i in range(10):
                print('nop')
        'Length of the series expansion'
        raise NotImplementedError('(%s).length' % self)

    @property
    def variables(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a tuple of variables that are bounded'
        return ()

    @property
    def free_symbols(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This method returns the symbols in the object, excluding those\n        that take on a specific value (i.e. the dummy symbols).\n        '
        return {j for i in self.args for j in i.free_symbols}.difference(self.variables)

    @cacheit
    def term(self, pt):
        if False:
            print('Hello World!')
        'Term at point pt of a series'
        if pt < self.start or pt > self.stop:
            raise IndexError('Index %s out of bounds %s' % (pt, self.interval))
        return self._eval_term(pt)

    def _eval_term(self, pt):
        if False:
            return 10
        raise NotImplementedError("The _eval_term method should be added to%s to return series term so it is availablewhen 'term' calls it." % self.func)

    def _ith_point(self, i):
        if False:
            print('Hello World!')
        "\n        Returns the i'th point of a series\n        If start point is negative infinity, point is returned from the end.\n        Assumes the first point to be indexed zero.\n\n        Examples\n        ========\n\n        TODO\n        "
        if self.start is S.NegativeInfinity:
            initial = self.stop
            step = -1
        else:
            initial = self.start
            step = 1
        return initial + i * step

    def __iter__(self):
        if False:
            while True:
                i = 10
        i = 0
        while i < self.length:
            pt = self._ith_point(i)
            yield self.term(pt)
            i += 1

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(index, int):
            index = self._ith_point(index)
            return self.term(index)
        elif isinstance(index, slice):
            (start, stop) = (index.start, index.stop)
            if start is None:
                start = 0
            if stop is None:
                stop = self.length
            return [self.term(self._ith_point(i)) for i in range(start, stop, index.step or 1)]