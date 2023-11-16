import pytest
from hypothesis import example, given, strategies as st
from hypothesis.errors import InvalidArgument
base_reusable_strategies = (st.text(), st.binary(), st.dates(), st.times(), st.timedeltas(), st.booleans(), st.complex_numbers(), st.floats(), st.floats(-1.0, 1.0), st.integers(), st.integers(1, 10), st.integers(1), st.just([]), st.sampled_from([[]]), st.tuples(st.integers()))

@st.deferred
def reusable():
    if False:
        for i in range(10):
            print('nop')
    'Meta-strategy that produces strategies that should have\n    ``.has_reusable_values == True``.'
    return st.one_of(st.sampled_from(base_reusable_strategies), st.builds(st.floats, min_value=st.none() | st.floats(allow_nan=False), max_value=st.none() | st.floats(allow_nan=False), allow_infinity=st.booleans(), allow_nan=st.booleans()), st.builds(st.just, st.builds(list)), st.builds(st.sampled_from, st.lists(st.builds(list), min_size=1)), st.lists(reusable).map(st.one_of), st.lists(reusable).map(lambda ls: st.tuples(*ls)))

def is_valid(s):
    if False:
        i = 10
        return i + 15
    try:
        s.validate()
        return True
    except InvalidArgument:
        return False
reusable = reusable.filter(is_valid)
assert not reusable.is_empty

def many_examples(examples):
    if False:
        while True:
            i = 10
    'Helper decorator to apply the ``@example`` decorator multiple times,\n    once for each given example.'

    def accept(f):
        if False:
            i = 10
            return i + 15
        for e in examples:
            f = example(e)(f)
        return f
    return accept

@many_examples(base_reusable_strategies)
@many_examples((st.tuples(s) for s in base_reusable_strategies))
@given(reusable)
def test_reusable_strategies_are_all_reusable(s):
    if False:
        while True:
            i = 10
    assert s.has_reusable_values

@many_examples(base_reusable_strategies)
@given(reusable)
def test_filter_breaks_reusability(s):
    if False:
        print('Hello World!')
    cond = True

    def nontrivial_filter(x):
        if False:
            print('Hello World!')
        'Non-trivial filtering function, intended to remain opaque even if\n        some strategies introspect their filters.'
        return cond
    assert s.has_reusable_values
    assert not s.filter(nontrivial_filter).has_reusable_values

@many_examples(base_reusable_strategies)
@given(reusable)
def test_map_breaks_reusability(s):
    if False:
        while True:
            i = 10
    cond = True

    def nontrivial_map(x):
        if False:
            while True:
                i = 10
        'Non-trivial mapping function, intended to remain opaque even if\n        some strategies introspect their mappings.'
        if cond:
            return x
        else:
            return None
    assert s.has_reusable_values
    assert not s.map(nontrivial_map).has_reusable_values

@many_examples(base_reusable_strategies)
@given(reusable)
def test_flatmap_breaks_reusability(s):
    if False:
        i = 10
        return i + 15
    cond = True

    def nontrivial_flatmap(x):
        if False:
            for i in range(10):
                print('nop')
        'Non-trivial flat-mapping function, intended to remain opaque even\n        if some strategies introspect their flat-mappings.'
        if cond:
            return st.just(x)
        else:
            return st.none()
    assert s.has_reusable_values
    assert not s.flatmap(nontrivial_flatmap).has_reusable_values

@pytest.mark.parametrize('strat', [st.lists(st.booleans()), st.sets(st.booleans()), st.dictionaries(st.booleans(), st.booleans())])
def test_mutable_collections_do_not_have_reusable_values(strat):
    if False:
        for i in range(10):
            print('nop')
    assert not strat.has_reusable_values

def test_recursion_does_not_break_reusability():
    if False:
        i = 10
        return i + 15
    x = st.deferred(lambda : st.none() | st.tuples(x))
    assert x.has_reusable_values