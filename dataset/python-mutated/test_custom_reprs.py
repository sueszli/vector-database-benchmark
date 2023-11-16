import re
import pytest
from hypothesis import given, settings, strategies as st

def test_includes_non_default_args_in_repr():
    if False:
        print('Hello World!')
    assert repr(st.integers()) == 'integers()'
    assert repr(st.integers(min_value=1)) == 'integers(min_value=1)'

def test_sampled_repr_leaves_range_as_range():
    if False:
        while True:
            i = 10
    huge = 10 ** 100
    assert repr(st.sampled_from(range(huge))) == f'sampled_from(range(0, {huge}))'

def hi(there, stuff):
    if False:
        i = 10
        return i + 15
    return there

def test_supports_positional_and_keyword_args_in_builds():
    if False:
        while True:
            i = 10
    assert repr(st.builds(hi, st.integers(), there=st.booleans())) == 'builds(hi, integers(), there=booleans())'

def test_preserves_sequence_type_of_argument():
    if False:
        return 10
    assert repr(st.sampled_from([0, 1])) == 'sampled_from([0, 1])'
    assert repr(st.sampled_from((0, 1))) == 'sampled_from((0, 1))'

class IHaveABadRepr:

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        raise ValueError('Oh no!')

def test_errors_are_deferred_until_repr_is_calculated():
    if False:
        while True:
            i = 10
    s = st.builds(lambda x, y: 1, st.just(IHaveABadRepr()), y=st.one_of(st.sampled_from((IHaveABadRepr(),)), st.just(IHaveABadRepr()))).map(lambda t: t).filter(lambda t: True).flatmap(lambda t: st.just(IHaveABadRepr()))
    with pytest.raises(ValueError):
        repr(s)

@given(st.iterables(st.integers()))
def test_iterables_repr_is_useful(it):
    if False:
        while True:
            i = 10
    assert repr(it) == f'iter({it._values!r})'

class Foo:

    def __init__(self, x: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.x = x

class Bar(Foo):
    pass

def test_reprs_as_created():
    if False:
        while True:
            i = 10

    @given(foo=st.builds(Foo), bar=st.from_type(Bar), baz=st.none().map(Foo))
    @settings(print_blob=False, max_examples=10000)
    def inner(foo, bar, baz):
        if False:
            i = 10
            return i + 15
        assert baz.x is None
        assert foo.x <= 0 or bar.x >= 0
    with pytest.raises(AssertionError) as err:
        inner()
    expected = '\nFalsifying example: inner(\n    foo=Foo(x=1),\n    bar=Bar(x=-1),\n    baz=Foo(None),\n)\n'
    assert '\n'.join(err.value.__notes__).strip() == expected.strip()

def test_reprs_as_created_interactive():
    if False:
        for i in range(10):
            print('nop')

    @given(st.data())
    @settings(print_blob=False, max_examples=10000)
    def inner(data):
        if False:
            while True:
                i = 10
        bar = data.draw(st.builds(Bar, st.just(10)))
        assert not bar.x
    with pytest.raises(AssertionError) as err:
        inner()
    expected = '\nFalsifying example: inner(\n    data=data(...),\n)\nDraw 1: Bar(10)\n'
    assert '\n'.join(err.value.__notes__).strip() == expected.strip()
CONSTANT_FOO = Foo(None)

def some_foo(*_):
    if False:
        i = 10
        return i + 15
    return CONSTANT_FOO

def test_as_created_reprs_fallback_for_distinct_calls_same_obj():
    if False:
        while True:
            i = 10

    @given(st.builds(some_foo), st.builds(some_foo, st.none()))
    @settings(print_blob=False, max_examples=10000)
    def inner(a, b):
        if False:
            i = 10
            return i + 15
        assert a is not b
    with pytest.raises(AssertionError) as err:
        inner()
    expected_re = '\nFalsifying example: inner\\(\n    a=<.*Foo object at 0x[0-9A-Fa-f]+>,\n    b=<.*Foo object at 0x[0-9A-Fa-f]+>,\n\\)\n'.strip()
    got = '\n'.join(err.value.__notes__).strip()
    assert re.fullmatch(expected_re, got), got

def test_reprs_as_created_consistent_calls_despite_indentation():
    if False:
        print('Hello World!')
    aas = 'a' * 60
    strat = st.builds(some_foo, st.just(aas))

    @given(strat, st.builds(Bar, strat))
    @settings(print_blob=False, max_examples=10000)
    def inner(a, b):
        if False:
            print('Hello World!')
        assert a == b
    with pytest.raises(AssertionError) as err:
        inner()
    expected = f'\nFalsifying example: inner(\n    a=some_foo({aas!r}),\n    b=Bar(\n        some_foo(\n            {aas!r},\n        ),\n    ),\n)\n'
    assert '\n'.join(err.value.__notes__).strip() == expected.strip()