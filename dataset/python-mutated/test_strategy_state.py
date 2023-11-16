import hashlib
import math
from random import Random
from hypothesis import Verbosity, assume, settings
from hypothesis.database import ExampleDatabase
from hypothesis.internal.compat import PYPY
from hypothesis.internal.floats import float_to_int, int_to_float, is_negative
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule
from hypothesis.strategies import binary, booleans, complex_numbers, data, decimals, floats, fractions, integers, just, lists, none, sampled_from, text, tuples
AVERAGE_LIST_LENGTH = 2

def clamp(lower, value, upper):
    if False:
        for i in range(10):
            print('nop')
    "Given a value and optional lower/upper bounds, 'clamp' the value so that\n    it satisfies lower <= value <= upper."
    if lower is not None and upper is not None and (lower > upper):
        raise ValueError(f'Cannot clamp with lower > upper: {lower!r} > {upper!r}')
    if lower is not None:
        value = max(lower, value)
    if upper is not None:
        value = min(value, upper)
    return value

class HypothesisSpec(RuleBasedStateMachine):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.database = None
    strategies = Bundle('strategy')
    strategy_tuples = Bundle('tuples')
    objects = Bundle('objects')
    basic_data = Bundle('basic')
    varied_floats = Bundle('varied_floats')

    def teardown(self):
        if False:
            print('Hello World!')
        self.clear_database()

    @rule()
    def clear_database(self):
        if False:
            i = 10
            return i + 15
        if self.database is not None:
            self.database = None

    @rule()
    def set_database(self):
        if False:
            for i in range(10):
                print('nop')
        self.teardown()
        self.database = ExampleDatabase()

    @rule(target=strategies, spec=sampled_from((integers(), booleans(), floats(), complex_numbers(), fractions(), decimals(), text(), binary(), none(), tuples())))
    def strategy(self, spec):
        if False:
            print('Hello World!')
        return spec

    @rule(target=strategies, values=lists(integers() | text(), min_size=1))
    def sampled_from_strategy(self, values):
        if False:
            i = 10
            return i + 15
        return sampled_from(values)

    @rule(target=strategies, spec=strategy_tuples)
    def strategy_for_tupes(self, spec):
        if False:
            i = 10
            return i + 15
        return tuples(*spec)

    @rule(target=strategies, source=strategies, level=integers(1, 10), mixer=text())
    def filtered_strategy(self, source, level, mixer):
        if False:
            for i in range(10):
                print('nop')

        def is_good(x):
            if False:
                while True:
                    i = 10
            seed = hashlib.sha384((mixer + repr(x)).encode()).digest()
            return bool(Random(seed).randint(0, level))
        return source.filter(is_good)

    @rule(target=strategies, elements=strategies)
    def list_strategy(self, elements):
        if False:
            i = 10
            return i + 15
        return lists(elements)

    @rule(target=strategies, left=strategies, right=strategies)
    def or_strategy(self, left, right):
        if False:
            for i in range(10):
                print('nop')
        return left | right

    @rule(target=varied_floats, source=floats())
    def float(self, source):
        if False:
            while True:
                i = 10
        return source

    @rule(target=varied_floats, source=varied_floats, offset=integers(-100, 100))
    def adjust_float(self, source, offset):
        if False:
            while True:
                i = 10
        return int_to_float(clamp(0, float_to_int(source) + offset, 2 ** 64 - 1))

    @rule(target=strategies, left=varied_floats, right=varied_floats)
    def float_range(self, left, right):
        if False:
            i = 10
            return i + 15
        assume(math.isfinite(left) and math.isfinite(right))
        (left, right) = sorted((left, right))
        assert left <= right
        assume(left or right or (not (is_negative(right) and (not is_negative(left)))))
        return floats(left, right)

    @rule(target=strategies, source=strategies, result1=strategies, result2=strategies, mixer=text(), p=floats(0, 1))
    def flatmapped_strategy(self, source, result1, result2, mixer, p):
        if False:
            for i in range(10):
                print('nop')
        assume(result1 is not result2)

        def do_map(value):
            if False:
                while True:
                    i = 10
            rep = repr(value)
            random = Random(hashlib.sha384((mixer + rep).encode()).digest())
            if random.random() <= p:
                return result1
            else:
                return result2
        return source.flatmap(do_map)

    @rule(target=strategies, value=objects)
    def just_strategy(self, value):
        if False:
            while True:
                i = 10
        return just(value)

    @rule(target=strategy_tuples, source=strategies)
    def single_tuple(self, source):
        if False:
            i = 10
            return i + 15
        return (source,)

    @rule(target=strategy_tuples, left=strategy_tuples, right=strategy_tuples)
    def cat_tuples(self, left, right):
        if False:
            while True:
                i = 10
        return left + right

    @rule(target=objects, strat=strategies, data=data())
    def get_example(self, strat, data):
        if False:
            i = 10
            return i + 15
        data.draw(strat)

    @rule(target=strategies, left=integers(), right=integers())
    def integer_range(self, left, right):
        if False:
            for i in range(10):
                print('nop')
        (left, right) = sorted((left, right))
        return integers(left, right)

    @rule(strat=strategies)
    def repr_is_good(self, strat):
        if False:
            i = 10
            return i + 15
        assert ' at 0x' not in repr(strat)
MAIN = __name__ == '__main__'
TestHypothesis = HypothesisSpec.TestCase
TestHypothesis.settings = settings(TestHypothesis.settings, stateful_step_count=10 if PYPY else 50, verbosity=max(TestHypothesis.settings.verbosity, Verbosity.verbose), max_examples=10000 if MAIN else 200)
if MAIN:
    TestHypothesis().runTest()