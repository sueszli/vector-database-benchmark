from collections import OrderedDict
import pytest
from hypothesis import given, strategies as st
from hypothesis.control import reject
from hypothesis.errors import HypothesisDeprecationWarning, InvalidArgument

def foo(x):
    if False:
        return 10
    pass

def bar(x):
    if False:
        while True:
            i = 10
    pass

def baz(x):
    if False:
        while True:
            i = 10
    pass
fns = [foo, bar, baz]

def builds_ignoring_invalid(target, *args, **kwargs):
    if False:
        i = 10
        return i + 15

    def splat(value):
        if False:
            print('Hello World!')
        try:
            result = target(*value[0], **value[1])
            result.validate()
            return result
        except (HypothesisDeprecationWarning, InvalidArgument):
            reject()
    return st.tuples(st.tuples(*args), st.fixed_dictionaries(kwargs)).map(splat)
size_strategies = {'min_size': st.integers(min_value=0, max_value=100), 'max_size': st.integers(min_value=0, max_value=100) | st.none()}
values = st.integers() | st.text()
Strategies = st.recursive(st.one_of(st.sampled_from([st.none(), st.booleans(), st.randoms(use_true_random=True), st.complex_numbers(), st.randoms(use_true_random=True), st.fractions(), st.decimals()]), st.builds(st.just, values), st.builds(st.sampled_from, st.lists(values, min_size=1)), builds_ignoring_invalid(st.floats, st.floats(), st.floats())), lambda x: st.one_of(builds_ignoring_invalid(st.lists, x, **size_strategies), builds_ignoring_invalid(st.sets, x, **size_strategies), builds_ignoring_invalid(lambda v: st.tuples(*v), st.lists(x)), builds_ignoring_invalid(lambda v: st.one_of(*v), st.lists(x, min_size=1)), builds_ignoring_invalid(st.dictionaries, x, x, dict_class=st.sampled_from([dict, OrderedDict]), **size_strategies), st.builds(lambda s, f: s.map(f), x, st.sampled_from(fns))))
strategy_globals = {k: getattr(st, k) for k in dir(st)}
strategy_globals['OrderedDict'] = OrderedDict
strategy_globals['inf'] = float('inf')
strategy_globals['nan'] = float('nan')
strategy_globals['foo'] = foo
strategy_globals['bar'] = bar
strategy_globals['baz'] = baz

@given(Strategies)
def test_repr_evals_to_thing_with_same_repr(strategy):
    if False:
        for i in range(10):
            print('nop')
    r = repr(strategy)
    via_eval = eval(r, strategy_globals)
    r2 = repr(via_eval)
    assert r == r2

@pytest.mark.parametrize('r', ['none().filter(foo).map(bar)', 'just(1).filter(foo).map(bar)', 'sampled_from([1, 2, 3]).filter(foo).map(bar)'])
def test_sampled_transform_reprs(r):
    if False:
        return 10
    assert repr(eval(r, strategy_globals)) == r