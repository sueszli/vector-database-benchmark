"""A library of basic cythonized CombineFn subclasses.

For internal use only; no backwards-compatibility guarantees.
"""
import operator
from apache_beam.transforms import core
try:
    from apache_beam.transforms.cy_dataflow_distribution_counter import DataflowDistributionCounter
except ImportError:
    from apache_beam.transforms.py_dataflow_distribution_counter import DataflowDistributionCounter

class AccumulatorCombineFn(core.CombineFn):

    def create_accumulator(self):
        if False:
            while True:
                i = 10
        return self._accumulator_type()

    @staticmethod
    def add_input(accumulator, element):
        if False:
            for i in range(10):
                print('nop')
        accumulator.add_input(element)
        return accumulator

    def merge_accumulators(self, accumulators):
        if False:
            while True:
                i = 10
        accumulator = self._accumulator_type()
        accumulator.merge(accumulators)
        return accumulator

    @staticmethod
    def extract_output(accumulator):
        if False:
            for i in range(10):
                print('nop')
        return accumulator.extract_output()

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return isinstance(other, AccumulatorCombineFn) and self._accumulator_type is other._accumulator_type

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash(self._accumulator_type)
_63 = 63
globals()['INT64_MAX'] = 2 ** _63 - 1
globals()['INT64_MIN'] = -2 ** _63

class CountAccumulator(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.value = 0

    def add_input(self, unused_element):
        if False:
            while True:
                i = 10
        self.value += 1

    def add_input_n(self, unused_element, n):
        if False:
            while True:
                i = 10
        self.value += n

    def merge(self, accumulators):
        if False:
            for i in range(10):
                print('nop')
        for accumulator in accumulators:
            self.value += accumulator.value

    def extract_output(self):
        if False:
            for i in range(10):
                print('nop')
        return self.value

class SumInt64Accumulator(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.value = 0

    def add_input(self, element):
        if False:
            for i in range(10):
                print('nop')
        global INT64_MAX, INT64_MIN
        element = int(element)
        if not INT64_MIN <= element <= INT64_MAX:
            raise OverflowError(element)
        self.value += element

    def add_input_n(self, element, n):
        if False:
            return 10
        global INT64_MAX, INT64_MIN
        element = int(element)
        if not INT64_MIN <= element <= INT64_MAX:
            raise OverflowError(element)
        self.value += element * n

    def merge(self, accumulators):
        if False:
            for i in range(10):
                print('nop')
        for accumulator in accumulators:
            self.value += accumulator.value

    def extract_output(self):
        if False:
            i = 10
            return i + 15
        if not INT64_MIN <= self.value <= INT64_MAX:
            self.value %= 2 ** 64
            if self.value >= INT64_MAX:
                self.value -= 2 ** 64
        return self.value

class MinInt64Accumulator(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.value = INT64_MAX

    def add_input(self, element):
        if False:
            for i in range(10):
                print('nop')
        element = int(element)
        if not INT64_MIN <= element <= INT64_MAX:
            raise OverflowError(element)
        if element < self.value:
            self.value = element

    def add_input_n(self, element, unused_n):
        if False:
            return 10
        self.add_input(element)

    def merge(self, accumulators):
        if False:
            return 10
        for accumulator in accumulators:
            if accumulator.value < self.value:
                self.value = accumulator.value

    def extract_output(self):
        if False:
            for i in range(10):
                print('nop')
        return self.value

class MaxInt64Accumulator(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.value = INT64_MIN

    def add_input(self, element):
        if False:
            i = 10
            return i + 15
        element = int(element)
        if not INT64_MIN <= element <= INT64_MAX:
            raise OverflowError(element)
        if element > self.value:
            self.value = element

    def add_input_n(self, element, unused_n):
        if False:
            i = 10
            return i + 15
        self.add_input(element)

    def merge(self, accumulators):
        if False:
            for i in range(10):
                print('nop')
        for accumulator in accumulators:
            if accumulator.value > self.value:
                self.value = accumulator.value

    def extract_output(self):
        if False:
            return 10
        return self.value

class MeanInt64Accumulator(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.sum = 0
        self.count = 0

    def add_input(self, element):
        if False:
            return 10
        element = int(element)
        if not INT64_MIN <= element <= INT64_MAX:
            raise OverflowError(element)
        self.sum += element
        self.count += 1

    def add_input_n(self, element, n):
        if False:
            print('Hello World!')
        element = int(element)
        if not INT64_MIN <= element <= INT64_MAX:
            raise OverflowError(element)
        self.sum += element * n
        self.count += n

    def merge(self, accumulators):
        if False:
            for i in range(10):
                print('nop')
        for accumulator in accumulators:
            self.sum += accumulator.sum
            self.count += accumulator.count

    def extract_output(self):
        if False:
            i = 10
            return i + 15
        if not INT64_MIN <= self.sum <= INT64_MAX:
            self.sum %= 2 ** 64
            if self.sum >= INT64_MAX:
                self.sum -= 2 ** 64
        return self.sum // self.count if self.count else _NAN

class DistributionInt64Accumulator(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.sum = 0
        self.count = 0
        self.min = INT64_MAX
        self.max = INT64_MIN

    def add_input(self, element):
        if False:
            while True:
                i = 10
        element = int(element)
        if not INT64_MIN <= element <= INT64_MAX:
            raise OverflowError(element)
        self.sum += element
        self.count += 1
        self.min = min(self.min, element)
        self.max = max(self.max, element)

    def add_input_n(self, element, n):
        if False:
            for i in range(10):
                print('nop')
        element = int(element)
        if not INT64_MIN <= element <= INT64_MAX:
            raise OverflowError(element)
        self.sum += element * n
        self.count += n
        self.min = min(self.min, element)
        self.max = max(self.max, element)

    def merge(self, accumulators):
        if False:
            while True:
                i = 10
        for accumulator in accumulators:
            self.sum += accumulator.sum
            self.count += accumulator.count
            self.min = min(self.min, accumulator.min)
            self.max = max(self.max, accumulator.max)

    def extract_output(self):
        if False:
            i = 10
            return i + 15
        if not INT64_MIN <= self.sum <= INT64_MAX:
            self.sum %= 2 ** 64
            if self.sum >= INT64_MAX:
                self.sum -= 2 ** 64
        mean = self.sum // self.count if self.count else _NAN
        return (mean, self.sum, self.count, self.min, self.max)

class CountCombineFn(AccumulatorCombineFn):
    _accumulator_type = CountAccumulator

class SumInt64Fn(AccumulatorCombineFn):
    _accumulator_type = SumInt64Accumulator

class MinInt64Fn(AccumulatorCombineFn):
    _accumulator_type = MinInt64Accumulator

class MaxInt64Fn(AccumulatorCombineFn):
    _accumulator_type = MaxInt64Accumulator

class MeanInt64Fn(AccumulatorCombineFn):
    _accumulator_type = MeanInt64Accumulator

class DistributionInt64Fn(AccumulatorCombineFn):
    _accumulator_type = DistributionInt64Accumulator
_POS_INF = float('inf')
_NEG_INF = float('-inf')
_NAN = float('nan')

class SumDoubleAccumulator(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.value = 0

    def add_input(self, element):
        if False:
            while True:
                i = 10
        element = float(element)
        self.value += element

    def merge(self, accumulators):
        if False:
            while True:
                i = 10
        for accumulator in accumulators:
            self.value += accumulator.value

    def extract_output(self):
        if False:
            for i in range(10):
                print('nop')
        return self.value

class MinDoubleAccumulator(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.value = _POS_INF

    def add_input(self, element):
        if False:
            print('Hello World!')
        element = float(element)
        if element < self.value:
            self.value = element

    def merge(self, accumulators):
        if False:
            while True:
                i = 10
        for accumulator in accumulators:
            if accumulator.value < self.value:
                self.value = accumulator.value

    def extract_output(self):
        if False:
            i = 10
            return i + 15
        return self.value

class MaxDoubleAccumulator(object):

    def __init__(self):
        if False:
            return 10
        self.value = _NEG_INF

    def add_input(self, element):
        if False:
            print('Hello World!')
        element = float(element)
        if element > self.value:
            self.value = element

    def merge(self, accumulators):
        if False:
            i = 10
            return i + 15
        for accumulator in accumulators:
            if accumulator.value > self.value:
                self.value = accumulator.value

    def extract_output(self):
        if False:
            i = 10
            return i + 15
        return self.value

class MeanDoubleAccumulator(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.sum = 0
        self.count = 0

    def add_input(self, element):
        if False:
            for i in range(10):
                print('nop')
        element = float(element)
        self.sum += element
        self.count += 1

    def merge(self, accumulators):
        if False:
            i = 10
            return i + 15
        for accumulator in accumulators:
            self.sum += accumulator.sum
            self.count += accumulator.count

    def extract_output(self):
        if False:
            i = 10
            return i + 15
        return self.sum // self.count if self.count else _NAN

class SumFloatFn(AccumulatorCombineFn):
    _accumulator_type = SumDoubleAccumulator

class MinFloatFn(AccumulatorCombineFn):
    _accumulator_type = MinDoubleAccumulator

class MaxFloatFn(AccumulatorCombineFn):
    _accumulator_type = MaxDoubleAccumulator

class MeanFloatFn(AccumulatorCombineFn):
    _accumulator_type = MeanDoubleAccumulator

class AllAccumulator(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.value = True

    def add_input(self, element):
        if False:
            for i in range(10):
                print('nop')
        self.value &= not not element

    def merge(self, accumulators):
        if False:
            for i in range(10):
                print('nop')
        for accumulator in accumulators:
            self.value &= accumulator.value

    def extract_output(self):
        if False:
            while True:
                i = 10
        return self.value

class AnyAccumulator(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.value = False

    def add_input(self, element):
        if False:
            return 10
        self.value |= not not element

    def merge(self, accumulators):
        if False:
            i = 10
            return i + 15
        for accumulator in accumulators:
            self.value |= accumulator.value

    def extract_output(self):
        if False:
            for i in range(10):
                print('nop')
        return self.value

class AnyCombineFn(AccumulatorCombineFn):
    _accumulator_type = AnyAccumulator

class AllCombineFn(AccumulatorCombineFn):
    _accumulator_type = AllAccumulator

class DataflowDistributionCounterFn(AccumulatorCombineFn):
    """A subclass of cy_combiners.AccumulatorCombineFn.

  Make DataflowDistributionCounter able to report to Dataflow service via
  CounterFactory.

  When cythonized DataflowDistributinoCounter available, make
  CounterFn combine with cythonized module, otherwise, combine with python
  version.
  """
    _accumulator_type = DataflowDistributionCounter

class ComparableValue(object):
    """A way to allow comparing elements in a rich fashion."""
    __slots__ = ('value', '_less_than_fn', '_comparable_value', 'requires_hydration')

    def __init__(self, value, less_than_fn, key_fn, _requires_hydration=False):
        if False:
            for i in range(10):
                print('nop')
        self.value = value
        self.hydrate(less_than_fn, key_fn)
        self.requires_hydration = _requires_hydration

    def hydrate(self, less_than_fn, key_fn):
        if False:
            return 10
        self._less_than_fn = less_than_fn if less_than_fn else operator.lt
        self._comparable_value = key_fn(self.value) if key_fn else self.value
        self.requires_hydration = False

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        assert not self.requires_hydration
        assert self._less_than_fn is other._less_than_fn
        return self._less_than_fn(self._comparable_value, other._comparable_value)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'ComparableValue[%s]' % str(self.value)

    def __reduce__(self):
        if False:
            print('Hello World!')
        return (ComparableValue, (self.value, None, None, True))