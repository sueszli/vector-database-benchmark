import functools
from collections import namedtuple
import pytest
from hypothesis.errors import InvalidArgument
from hypothesis.internal.conjecture.data import ConjectureData
from hypothesis.strategies import booleans, integers, just, none, tuples
from tests.common.debug import assert_no_examples

def test_or_errors_when_given_non_strategy():
    if False:
        i = 10
        return i + 15
    bools = tuples(booleans())
    with pytest.raises(ValueError):
        bools | 'foo'
SomeNamedTuple = namedtuple('SomeNamedTuple', ('a', 'b'))

def last(xs):
    if False:
        print('Hello World!')
    t = None
    for x in xs:
        t = x
    return t

def test_just_strategy_uses_repr():
    if False:
        print('Hello World!')

    class WeirdRepr:

        def __repr__(self):
            if False:
                for i in range(10):
                    print('nop')
            return 'ABCDEFG'
    assert repr(just(WeirdRepr())) == f'just({WeirdRepr()!r})'

def test_just_strategy_does_not_draw():
    if False:
        for i in range(10):
            print('nop')
    data = ConjectureData.for_buffer(b'')
    s = just('hello')
    assert s.do_draw(data) == 'hello'

def test_none_strategy_does_not_draw():
    if False:
        i = 10
        return i + 15
    data = ConjectureData.for_buffer(b'')
    s = none()
    assert s.do_draw(data) is None

def test_can_map():
    if False:
        i = 10
        return i + 15
    s = integers().map(pack=lambda t: 'foo')
    assert s.example() == 'foo'

def test_example_raises_unsatisfiable_when_too_filtered():
    if False:
        print('Hello World!')
    assert_no_examples(integers().filter(lambda x: False))

def nameless_const(x):
    if False:
        print('Hello World!')

    def f(u, v):
        if False:
            return 10
        return u
    return functools.partial(f, x)

def test_can_map_nameless():
    if False:
        return 10
    f = nameless_const(2)
    assert repr(f) in repr(integers().map(f))

def test_can_flatmap_nameless():
    if False:
        while True:
            i = 10
    f = nameless_const(just(3))
    assert repr(f) in repr(integers().flatmap(f))

def test_flatmap_with_invalid_expand():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(InvalidArgument):
        just(100).flatmap(lambda n: 'a').example()