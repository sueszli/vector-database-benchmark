import math

class RunningStats:
    """Incrementally compute statistics"""

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.clear()

    def __add__(self: 'RunningStats', other: 'RunningStats') -> 'RunningStats':
        if False:
            while True:
                i = 10
        s = RunningStats()
        if other._n > 0:
            s._m1 = (self._m1 * self._n + other._m1 * other._n) / (self._n + other._n)
            s._n = self._n + other._n
            s._peak = max(self._peak, other._peak)
        else:
            s = self
        return s

    def clear(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Reset for new samples'
        self._n = 0
        self._m1 = self._m2 = self._m3 = self._m4 = 0.0
        self._peak = 0.0

    def push(self, x: float) -> None:
        if False:
            while True:
                i = 10
        'Add a sample'
        if x > self._peak:
            self._peak = x
        n1 = self._n
        self._n += 1
        delta = x - self._m1
        delta_n = delta / self._n
        delta_n2 = delta_n * delta_n
        term1 = delta * delta_n * n1
        self._m1 += delta_n
        self._m4 += term1 * delta_n2 * (self._n * self._n - 3 * self._n + 3) + 6 * delta_n2 * self._m2 - 4 * delta_n * self._m3
        self._m3 += term1 * delta_n * (self._n - 2) - 3 * delta_n * self._m2
        self._m2 += term1

    def peak(self) -> float:
        if False:
            while True:
                i = 10
        'The maximum sample seen.'
        return self._peak

    def size(self) -> int:
        if False:
            return 10
        'The number of samples'
        return self._n

    def mean(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        'Arithmetic mean, a.k.a. average'
        return self._m1

    def var(self) -> float:
        if False:
            print('Hello World!')
        'Variance'
        return self._m2 / (self._n - 1.0)

    def std(self) -> float:
        if False:
            while True:
                i = 10
        'Standard deviation'
        return math.sqrt(self.var())

    def sem(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        'Standard error of the mean'
        return self.std() / math.sqrt(self._n)