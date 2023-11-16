import copy
import math
from typing import Dict, Iterable, Optional
try:
    from numpy import maximum, minimum, sqrt
except ImportError:
    maximum = max
    minimum = min
    sqrt = math.sqrt

class StatCounter:

    def __init__(self, values: Optional[Iterable[float]]=None):
        if False:
            while True:
                i = 10
        if values is None:
            values = list()
        self.n = 0
        self.mu = 0.0
        self.m2 = 0.0
        self.maxValue = float('-inf')
        self.minValue = float('inf')
        for v in values:
            self.merge(v)

    def merge(self, value: float) -> 'StatCounter':
        if False:
            i = 10
            return i + 15
        delta = value - self.mu
        self.n += 1
        self.mu += delta / self.n
        self.m2 += delta * (value - self.mu)
        self.maxValue = maximum(self.maxValue, value)
        self.minValue = minimum(self.minValue, value)
        return self

    def mergeStats(self, other: 'StatCounter') -> 'StatCounter':
        if False:
            i = 10
            return i + 15
        if not isinstance(other, StatCounter):
            raise TypeError('Can only merge StatCounter but got %s' % type(other))
        if other is self:
            self.mergeStats(other.copy())
        elif self.n == 0:
            self.mu = other.mu
            self.m2 = other.m2
            self.n = other.n
            self.maxValue = other.maxValue
            self.minValue = other.minValue
        elif other.n != 0:
            delta = other.mu - self.mu
            if other.n * 10 < self.n:
                self.mu = self.mu + delta * other.n / (self.n + other.n)
            elif self.n * 10 < other.n:
                self.mu = other.mu - delta * self.n / (self.n + other.n)
            else:
                self.mu = (self.mu * self.n + other.mu * other.n) / (self.n + other.n)
            self.maxValue = maximum(self.maxValue, other.maxValue)
            self.minValue = minimum(self.minValue, other.minValue)
            self.m2 += other.m2 + delta * delta * self.n * other.n / (self.n + other.n)
            self.n += other.n
        return self

    def copy(self) -> 'StatCounter':
        if False:
            while True:
                i = 10
        return copy.deepcopy(self)

    def count(self) -> int:
        if False:
            while True:
                i = 10
        return int(self.n)

    def mean(self) -> float:
        if False:
            return 10
        return self.mu

    def sum(self) -> float:
        if False:
            while True:
                i = 10
        return self.n * self.mu

    def min(self) -> float:
        if False:
            i = 10
            return i + 15
        return self.minValue

    def max(self) -> float:
        if False:
            while True:
                i = 10
        return self.maxValue

    def variance(self) -> float:
        if False:
            print('Hello World!')
        if self.n == 0:
            return float('nan')
        else:
            return self.m2 / self.n

    def sampleVariance(self) -> float:
        if False:
            print('Hello World!')
        if self.n <= 1:
            return float('nan')
        else:
            return self.m2 / (self.n - 1)

    def stdev(self) -> float:
        if False:
            return 10
        return sqrt(self.variance())

    def sampleStdev(self) -> float:
        if False:
            return 10
        return sqrt(self.sampleVariance())

    def asDict(self, sample: bool=False) -> Dict[str, float]:
        if False:
            i = 10
            return i + 15
        "Returns the :class:`StatCounter` members as a ``dict``.\n\n        Examples\n        --------\n        >>> sc.parallelize([1., 2., 3., 4.]).stats().asDict()\n        {'count': 4L,\n         'max': 4.0,\n         'mean': 2.5,\n         'min': 1.0,\n         'stdev': 1.2909944487358056,\n         'sum': 10.0,\n         'variance': 1.6666666666666667}\n        "
        return {'count': self.count(), 'mean': self.mean(), 'sum': self.sum(), 'min': self.min(), 'max': self.max(), 'stdev': self.stdev() if sample else self.sampleStdev(), 'variance': self.variance() if sample else self.sampleVariance()}

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return '(count: %s, mean: %s, stdev: %s, max: %s, min: %s)' % (self.count(), self.mean(), self.stdev(), self.max(), self.min())