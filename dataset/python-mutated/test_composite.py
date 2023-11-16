import sys
import pytest
from hypothesis import assume, given, strategies as st
from hypothesis.errors import HypothesisDeprecationWarning, HypothesisWarning, InvalidArgument
from tests.common.debug import minimal
from tests.common.utils import flaky

@st.composite
def badly_draw_lists(draw, m=0):
    if False:
        for i in range(10):
            print('nop')
    length = draw(st.integers(m, m + 10))
    return [draw(st.integers()) for _ in range(length)]

def test_simplify_draws():
    if False:
        return 10
    assert minimal(badly_draw_lists(), lambda x: len(x) >= 3) == [0] * 3

def test_can_pass_through_arguments():
    if False:
        while True:
            i = 10
    assert minimal(badly_draw_lists(5), lambda x: True) == [0] * 5
    assert minimal(badly_draw_lists(m=6), lambda x: True) == [0] * 6

@st.composite
def draw_ordered_with_assume(draw):
    if False:
        i = 10
        return i + 15
    x = draw(st.floats())
    y = draw(st.floats())
    assume(x < y)
    return (x, y)

@given(draw_ordered_with_assume())
def test_can_assume_in_draw(xy):
    if False:
        while True:
            i = 10
    assert xy[0] < xy[1]

def test_uses_definitions_for_reprs():
    if False:
        return 10
    assert repr(badly_draw_lists()) == 'badly_draw_lists()'
    assert repr(badly_draw_lists(1)) == 'badly_draw_lists(m=1)'
    assert repr(badly_draw_lists(m=1)) == 'badly_draw_lists(m=1)'

def test_errors_given_default_for_draw():
    if False:
        print('Hello World!')
    with pytest.raises(InvalidArgument):

        @st.composite
        def foo(x=None):
            if False:
                while True:
                    i = 10
            pass

def test_errors_given_function_of_no_arguments():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(InvalidArgument):

        @st.composite
        def foo():
            if False:
                while True:
                    i = 10
            pass

def test_errors_given_kwargs_only():
    if False:
        i = 10
        return i + 15
    with pytest.raises(InvalidArgument):

        @st.composite
        def foo(**kwargs):
            if False:
                for i in range(10):
                    print('nop')
            pass

def test_warning_given_no_drawfn_call():
    if False:
        i = 10
        return i + 15
    with pytest.warns(HypothesisDeprecationWarning):

        @st.composite
        def foo(_):
            if False:
                return 10
            return 'bar'

def test_can_use_pure_args():
    if False:
        while True:
            i = 10

    @st.composite
    def stuff(*args):
        if False:
            print('Hello World!')
        return args[0](st.sampled_from(args[1:]))
    assert minimal(stuff(1, 2, 3, 4, 5), lambda x: True) == 1

def test_composite_of_lists():
    if False:
        i = 10
        return i + 15

    @st.composite
    def f(draw):
        if False:
            print('Hello World!')
        return draw(st.integers()) + draw(st.integers())
    assert minimal(st.lists(f()), lambda x: len(x) >= 10) == [0] * 10

@flaky(min_passes=2, max_runs=5)
def test_can_shrink_matrices_with_length_param():
    if False:
        i = 10
        return i + 15

    @st.composite
    def matrix(draw):
        if False:
            while True:
                i = 10
        rows = draw(st.integers(1, 10))
        columns = draw(st.integers(1, 10))
        return [[draw(st.integers(0, 10000)) for _ in range(columns)] for _ in range(rows)]

    def transpose(m):
        if False:
            for i in range(10):
                print('nop')
        return [[row[i] for row in m] for i in range(len(m[0]))]

    def is_square(m):
        if False:
            return 10
        return len(m) == len(m[0])
    value = minimal(matrix(), lambda m: is_square(m) and transpose(m) != m)
    assert len(value) == 2
    assert len(value[0]) == 2
    assert sorted(value[0] + value[1]) == [0, 0, 0, 1]

class MyList(list):
    pass

@given(st.data(), st.lists(st.integers()).map(MyList))
def test_does_not_change_arguments(data, ls):
    if False:
        for i in range(10):
            print('nop')

    @st.composite
    def strat(draw, arg):
        if False:
            i = 10
            return i + 15
        draw(st.none())
        return arg
    ex = data.draw(strat(ls))
    assert ex is ls

class ClsWithStrategyMethods:

    @classmethod
    @st.composite
    def st_classmethod_then_composite(draw, cls):
        if False:
            for i in range(10):
                print('nop')
        return draw(st.integers(0, 10))

    @st.composite
    @classmethod
    def st_composite_then_classmethod(draw, cls):
        if False:
            i = 10
            return i + 15
        return draw(st.integers(0, 10))

    @staticmethod
    @st.composite
    def st_staticmethod_then_composite(draw):
        if False:
            while True:
                i = 10
        return draw(st.integers(0, 10))

    @st.composite
    @staticmethod
    def st_composite_then_staticmethod(draw):
        if False:
            print('Hello World!')
        return draw(st.integers(0, 10))

    @st.composite
    def st_composite_method(draw, self):
        if False:
            i = 10
            return i + 15
        return draw(st.integers(0, 10))

@given(st.data())
def test_applying_composite_decorator_to_methods(data):
    if False:
        while True:
            i = 10
    instance = ClsWithStrategyMethods()
    for strategy in [ClsWithStrategyMethods.st_classmethod_then_composite(), ClsWithStrategyMethods.st_composite_then_classmethod(), ClsWithStrategyMethods.st_staticmethod_then_composite(), ClsWithStrategyMethods.st_composite_then_staticmethod(), instance.st_classmethod_then_composite(), instance.st_composite_then_classmethod(), instance.st_staticmethod_then_composite(), instance.st_composite_then_staticmethod(), instance.st_composite_method()]:
        x = data.draw(strategy)
        assert isinstance(x, int)
        assert 0 <= x <= 10

def test_drawfn_cannot_be_instantiated():
    if False:
        print('Hello World!')
    with pytest.raises(TypeError):
        st.DrawFn()

@pytest.mark.skipif(sys.version_info[:2] == (3, 9), reason='stack depth varies???')
def test_warns_on_strategy_annotation():
    if False:
        print('Hello World!')
    with pytest.warns(HypothesisWarning, match='Return-type annotation') as w:

        @st.composite
        def my_integers(draw: st.DrawFn) -> st.SearchStrategy[int]:
            if False:
                print('Hello World!')
            return draw(st.integers())
    assert len(w.list) == 1
    assert w.list[0].filename == __file__