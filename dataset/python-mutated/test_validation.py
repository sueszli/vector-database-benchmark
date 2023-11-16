import functools
import pytest
from hypothesis import find, given
from hypothesis.errors import InvalidArgument
from hypothesis.internal.validation import check_type
from hypothesis.strategies import SearchStrategy as ActualSearchStrategy, binary, booleans, data, dictionaries, floats, frozensets, integers, lists, nothing, recursive, sets, text
from hypothesis.strategies._internal.strategies import check_strategy
from tests.common.utils import fails_with

def test_errors_when_given_varargs():
    if False:
        print('Hello World!')

    @given(integers())
    def has_varargs(*args):
        if False:
            for i in range(10):
                print('nop')
        pass
    with pytest.raises(InvalidArgument) as e:
        has_varargs()
    assert 'varargs' in e.value.args[0]

def test_varargs_without_positional_arguments_allowed():
    if False:
        while True:
            i = 10

    @given(somearg=integers())
    def has_varargs(somearg, *args):
        if False:
            print('Hello World!')
        pass

def test_errors_when_given_varargs_and_kwargs_with_positional_arguments():
    if False:
        for i in range(10):
            print('nop')

    @given(integers())
    def has_varargs(*args, **kw):
        if False:
            return 10
        pass
    with pytest.raises(InvalidArgument) as e:
        has_varargs()
    assert 'varargs' in e.value.args[0]

def test_varargs_and_kwargs_without_positional_arguments_allowed():
    if False:
        while True:
            i = 10

    @given(somearg=integers())
    def has_varargs(*args, **kw):
        if False:
            while True:
                i = 10
        pass

def test_bare_given_errors():
    if False:
        for i in range(10):
            print('nop')

    @given()
    def test():
        if False:
            return 10
        pass
    with pytest.raises(InvalidArgument):
        test()

def test_errors_on_unwanted_kwargs():
    if False:
        while True:
            i = 10

    @given(hello=int, world=int)
    def greet(world):
        if False:
            for i in range(10):
                print('nop')
        pass
    with pytest.raises(InvalidArgument):
        greet()

def test_errors_on_too_many_positional_args():
    if False:
        i = 10
        return i + 15

    @given(integers(), int, int)
    def foo(x, y):
        if False:
            i = 10
            return i + 15
        pass
    with pytest.raises(InvalidArgument):
        foo()

def test_errors_on_any_varargs():
    if False:
        i = 10
        return i + 15

    @given(integers())
    def oops(*args):
        if False:
            print('Hello World!')
        pass
    with pytest.raises(InvalidArgument):
        oops()

def test_can_put_arguments_in_the_middle():
    if False:
        while True:
            i = 10

    @given(y=integers())
    def foo(x, y, z):
        if False:
            return 10
        pass
    foo(1, 2)

def test_float_ranges():
    if False:
        i = 10
        return i + 15
    with pytest.raises(InvalidArgument):
        floats(float('nan'), 0).example()
    with pytest.raises(InvalidArgument):
        floats(1, -1).example()

def test_float_range_and_allow_nan_cannot_both_be_enabled():
    if False:
        return 10
    with pytest.raises(InvalidArgument):
        floats(min_value=1, allow_nan=True).example()
    with pytest.raises(InvalidArgument):
        floats(max_value=1, allow_nan=True).example()

def test_float_finite_range_and_allow_infinity_cannot_both_be_enabled():
    if False:
        return 10
    with pytest.raises(InvalidArgument):
        floats(0, 1, allow_infinity=True).example()

def test_does_not_error_if_min_size_is_bigger_than_default_size():
    if False:
        print('Hello World!')
    lists(integers(), min_size=50).example()
    sets(integers(), min_size=50).example()
    frozensets(integers(), min_size=50).example()
    lists(integers(), min_size=50, unique=True).example()

def test_list_unique_and_unique_by_cannot_both_be_enabled():
    if False:
        i = 10
        return i + 15

    @given(lists(integers(), unique=True, unique_by=lambda x: x))
    def boom(t):
        if False:
            return 10
        pass
    with pytest.raises(InvalidArgument) as e:
        boom()
    assert 'unique ' in e.value.args[0]
    assert 'unique_by' in e.value.args[0]

def test_min_before_max():
    if False:
        return 10
    with pytest.raises(InvalidArgument):
        integers(min_value=1, max_value=0).validate()

def test_filter_validates():
    if False:
        i = 10
        return i + 15
    with pytest.raises(InvalidArgument):
        integers(min_value=1, max_value=0).filter(bool).validate()

def test_recursion_validates_base_case():
    if False:
        return 10
    with pytest.raises(InvalidArgument):
        recursive(integers(min_value=1, max_value=0), lists).validate()

def test_recursion_validates_recursive_step():
    if False:
        print('Hello World!')
    with pytest.raises(InvalidArgument):
        recursive(integers(), lambda x: lists(x, min_size=3, max_size=1)).validate()

@fails_with(InvalidArgument)
@given(x=integers())
def test_stuff_keyword(x=1):
    if False:
        i = 10
        return i + 15
    pass

@fails_with(InvalidArgument)
@given(integers())
def test_stuff_positional(x=1):
    if False:
        print('Hello World!')
    pass

@fails_with(InvalidArgument)
@given(integers(), integers())
def test_too_many_positional(x):
    if False:
        i = 10
        return i + 15
    pass

def test_given_warns_on_use_of_non_strategies():
    if False:
        return 10

    @given(bool)
    def test(x):
        if False:
            i = 10
            return i + 15
        pass
    with pytest.raises(InvalidArgument):
        test()

def test_given_warns_when_mixing_positional_with_keyword():
    if False:
        return 10

    @given(booleans(), y=booleans())
    def test(x, y):
        if False:
            i = 10
            return i + 15
        pass
    with pytest.raises(InvalidArgument):
        test()

def test_cannot_find_non_strategies():
    if False:
        print('Hello World!')
    with pytest.raises(InvalidArgument):
        find(bool, bool)

@pytest.mark.parametrize('strategy', [functools.partial(lists, elements=integers()), functools.partial(dictionaries, keys=integers(), values=integers()), text, binary])
@pytest.mark.parametrize('min_size,max_size', [(0, '10'), ('0', 10)])
def test_valid_sizes(strategy, min_size, max_size):
    if False:
        return 10

    @given(strategy(min_size=min_size, max_size=max_size))
    def test(x):
        if False:
            for i in range(10):
                print('nop')
        pass
    with pytest.raises(InvalidArgument):
        test()

def test_check_type_with_tuple_of_length_two():
    if False:
        i = 10
        return i + 15

    def type_checker(x):
        if False:
            return 10
        check_type((int, str), x, 'x')
    type_checker(1)
    type_checker('1')
    with pytest.raises(InvalidArgument, match='Expected one of int, str but got '):
        type_checker(1.0)

def test_validation_happens_on_draw():
    if False:
        i = 10
        return i + 15

    @given(data())
    def test(data):
        if False:
            while True:
                i = 10
        data.draw(integers().flatmap(lambda _: lists(nothing(), min_size=1)))
    with pytest.raises(InvalidArgument, match='has no values'):
        test()

class SearchStrategy:
    """Not the SearchStrategy type you were looking for."""

def check_type_(*args):
    if False:
        for i in range(10):
            print('nop')
    return check_type(*args)

def test_check_type_suggests_check_strategy():
    if False:
        i = 10
        return i + 15
    check_type_(SearchStrategy, SearchStrategy(), 'this is OK')
    with pytest.raises(AssertionError, match='use check_strategy instead'):
        check_type_(ActualSearchStrategy, None, 'SearchStrategy assertion')

def check_strategy_(*args):
    if False:
        return 10
    return check_strategy(*args)

def test_check_strategy_might_suggest_sampled_from():
    if False:
        return 10
    with pytest.raises(InvalidArgument) as excinfo:
        check_strategy_('not a strategy')
    assert 'sampled_from' not in str(excinfo.value)
    with pytest.raises(InvalidArgument, match='such as st.sampled_from'):
        check_strategy_([1, 2, 3])
    with pytest.raises(InvalidArgument, match='such as st.sampled_from'):
        check_strategy_((1, 2, 3))
    check_strategy_(integers(), 'passes for our custom coverage check')