import collections
import decimal
import enum
import fractions
import math
from datetime import date, datetime, time, timedelta
from ipaddress import IPv4Network, IPv6Network
import pytest
from hypothesis import given, settings, strategies as ds
from hypothesis.errors import InvalidArgument
from hypothesis.vendor.pretty import pretty
from tests.common.debug import minimal

def fn_test(*fnkwargs):
    if False:
        return 10
    fnkwargs = list(fnkwargs)
    return pytest.mark.parametrize(('fn', 'args'), fnkwargs, ids=['{}({})'.format(fn.__name__, ', '.join(map(pretty, args))) for (fn, args) in fnkwargs])

def fn_ktest(*fnkwargs):
    if False:
        for i in range(10):
            print('nop')
    fnkwargs = list(fnkwargs)
    return pytest.mark.parametrize(('fn', 'kwargs'), fnkwargs, ids=[f'{fn.__name__}(**{pretty(kwargs)})' for (fn, kwargs) in fnkwargs])

@fn_ktest((ds.integers, {'min_value': math.nan}), (ds.integers, {'min_value': 2, 'max_value': 1}), (ds.integers, {'min_value': math.nan}), (ds.integers, {'max_value': math.nan}), (ds.integers, {'min_value': decimal.Decimal('1.5')}), (ds.integers, {'max_value': decimal.Decimal('1.5')}), (ds.integers, {'min_value': -1.5, 'max_value': -0.5}), (ds.integers, {'min_value': 0.1, 'max_value': 0.2}), (ds.dates, {'min_value': 'fish'}), (ds.dates, {'max_value': 'fish'}), (ds.dates, {'min_value': date(2017, 8, 22), 'max_value': date(2017, 8, 21)}), (ds.datetimes, {'min_value': 'fish'}), (ds.datetimes, {'max_value': 'fish'}), (ds.datetimes, {'allow_imaginary': 0}), (ds.datetimes, {'min_value': datetime(2017, 8, 22), 'max_value': datetime(2017, 8, 21)}), (ds.decimals, {'min_value': math.nan}), (ds.decimals, {'max_value': math.nan}), (ds.decimals, {'min_value': 2, 'max_value': 1}), (ds.decimals, {'max_value': '-snan'}), (ds.decimals, {'max_value': complex(1, 2)}), (ds.decimals, {'places': -1}), (ds.decimals, {'places': 0.5}), (ds.decimals, {'max_value': 0.0, 'min_value': 1.0}), (ds.decimals, {'min_value': 1.0, 'max_value': 0.0}), (ds.decimals, {'min_value': 0.0, 'max_value': 1.0, 'allow_infinity': True}), (ds.decimals, {'min_value': 'inf'}), (ds.decimals, {'max_value': '-inf'}), (ds.decimals, {'min_value': '-inf', 'allow_infinity': False}), (ds.decimals, {'max_value': 'inf', 'allow_infinity': False}), (ds.decimals, {'min_value': complex(1, 2)}), (ds.decimals, {'min_value': '0.1', 'max_value': '0.9', 'places': 0}), (ds.dictionaries, {'keys': ds.booleans(), 'values': ds.booleans(), 'min_size': 10, 'max_size': 1}), (ds.floats, {'min_value': math.nan}), (ds.floats, {'max_value': math.nan}), (ds.floats, {'min_value': complex(1, 2)}), (ds.floats, {'max_value': complex(1, 2)}), (ds.floats, {'exclude_min': None}), (ds.floats, {'exclude_max': None}), (ds.floats, {'exclude_min': True}), (ds.floats, {'exclude_max': True}), (ds.floats, {'min_value': 1.8, 'width': 32}), (ds.floats, {'max_value': 1.8, 'width': 32}), (ds.fractions, {'min_value': 2, 'max_value': 1}), (ds.fractions, {'min_value': math.nan}), (ds.fractions, {'max_value': math.nan}), (ds.fractions, {'max_denominator': 0}), (ds.fractions, {'max_denominator': 1.5}), (ds.fractions, {'min_value': complex(1, 2)}), (ds.fractions, {'min_value': '1/3', 'max_value': '1/2', 'max_denominator': 2}), (ds.fractions, {'min_value': '0', 'max_value': '1/3', 'max_denominator': 2}), (ds.fractions, {'min_value': '1/3', 'max_value': '1/3', 'max_denominator': 2}), (ds.lists, {'elements': ds.integers(), 'min_size': 10, 'max_size': 9}), (ds.lists, {'elements': ds.integers(), 'min_size': -10, 'max_size': -9}), (ds.lists, {'elements': ds.integers(), 'max_size': -9}), (ds.lists, {'elements': ds.integers(), 'min_size': -10}), (ds.lists, {'elements': ds.integers(), 'min_size': math.nan}), (ds.lists, {'elements': ds.nothing(), 'max_size': 1}), (ds.lists, {'elements': 'hi'}), (ds.lists, {'elements': ds.integers(), 'unique_by': 1}), (ds.lists, {'elements': ds.integers(), 'unique_by': ()}), (ds.lists, {'elements': ds.integers(), 'unique_by': (1,)}), (ds.lists, {'elements': ds.sampled_from([0, 1]), 'min_size': 3, 'unique': True}), (ds.text, {'min_size': 10, 'max_size': 9}), (ds.text, {'alphabet': [1]}), (ds.text, {'alphabet': ['abc']}), (ds.text, {'alphabet': ds.just('abc')}), (ds.text, {'alphabet': ds.sampled_from(['abc', 'def'])}), (ds.text, {'alphabet': ds.just(123)}), (ds.text, {'alphabet': ds.sampled_from([123, 456])}), (ds.text, {'alphabet': ds.builds(lambda : 'abc')}), (ds.text, {'alphabet': ds.builds(lambda : 123)}), (ds.from_regex, {'regex': 123}), (ds.from_regex, {'regex': b'abc', 'alphabet': 'abc'}), (ds.from_regex, {'regex': b'abc', 'alphabet': b'def'}), (ds.from_regex, {'regex': 'abc', 'alphabet': 'def'}), (ds.from_regex, {'regex': '[abc]', 'alphabet': 'def'}), (ds.from_regex, {'regex': '[a-d]', 'alphabet': 'def'}), (ds.from_regex, {'regex': '[f-z]', 'alphabet': 'def'}), (ds.from_regex, {'regex': '[ab]x[de]', 'alphabet': 'abcdef'}), (ds.from_regex, {'regex': '...', 'alphabet': ds.builds(lambda : 'a')}), (ds.from_regex, {'regex': 'abc', 'alphabet': ds.sampled_from('def')}), (ds.from_regex, {'regex': 'abc', 'alphabet': ds.characters(min_codepoint=128)}), (ds.from_regex, {'regex': 'abc', 'alphabet': 123}), (ds.binary, {'min_size': 10, 'max_size': 9}), (ds.floats, {'min_value': math.nan}), (ds.floats, {'min_value': '0'}), (ds.floats, {'max_value': '0'}), (ds.floats, {'min_value': 0.0, 'max_value': -0.0}), (ds.floats, {'min_value': 0.0, 'max_value': 1.0, 'allow_infinity': True}), (ds.floats, {'max_value': 0.0, 'min_value': 1.0}), (ds.floats, {'min_value': 0.0, 'allow_nan': True}), (ds.floats, {'max_value': 0.0, 'allow_nan': True}), (ds.floats, {'min_value': 0.0, 'max_value': 1.0, 'allow_infinity': True}), (ds.floats, {'min_value': math.inf, 'allow_infinity': False}), (ds.floats, {'max_value': -math.inf, 'allow_infinity': False}), (ds.complex_numbers, {'min_magnitude': None}), (ds.complex_numbers, {'min_magnitude': math.nan}), (ds.complex_numbers, {'max_magnitude': math.nan}), (ds.complex_numbers, {'max_magnitude': complex(1, 2)}), (ds.complex_numbers, {'min_magnitude': -1}), (ds.complex_numbers, {'max_magnitude': -1}), (ds.complex_numbers, {'min_magnitude': 3, 'max_magnitude': 2}), (ds.complex_numbers, {'max_magnitude': 2, 'allow_infinity': True}), (ds.complex_numbers, {'max_magnitude': 2, 'allow_nan': True}), (ds.complex_numbers, {'width': None}), (ds.complex_numbers, {'width': 16}), (ds.complex_numbers, {'width': 196}), (ds.complex_numbers, {'width': 256}), (ds.fixed_dictionaries, {'mapping': 'fish'}), (ds.fixed_dictionaries, {'mapping': {1: 'fish'}}), (ds.fixed_dictionaries, {'mapping': {}, 'optional': 'fish'}), (ds.fixed_dictionaries, {'mapping': {}, 'optional': {1: 'fish'}}), (ds.fixed_dictionaries, {'mapping': {}, 'optional': collections.OrderedDict()}), (ds.fixed_dictionaries, {'mapping': {1: ds.none()}, 'optional': {1: ds.none()}}), (ds.dictionaries, {'keys': ds.integers(), 'values': 1}), (ds.dictionaries, {'keys': 1, 'values': ds.integers()}), (ds.text, {'alphabet': '', 'min_size': 1}), (ds.timedeltas, {'min_value': 'fish'}), (ds.timedeltas, {'max_value': 'fish'}), (ds.timedeltas, {'min_value': timedelta(hours=1), 'max_value': timedelta(minutes=1)}), (ds.times, {'min_value': 'fish'}), (ds.times, {'max_value': 'fish'}), (ds.times, {'min_value': time(2, 0), 'max_value': time(1, 0)}), (ds.uuids, {'version': 6}), (ds.characters, {'min_codepoint': -1}), (ds.characters, {'min_codepoint': '1'}), (ds.characters, {'max_codepoint': -1}), (ds.characters, {'max_codepoint': '1'}), (ds.characters, {'categories': []}), (ds.characters, {'categories': ['Nd'], 'exclude_categories': ['Nd']}), (ds.characters, {'whitelist_categories': ['Nd'], 'blacklist_categories': ['Nd']}), (ds.characters, {'include_characters': 'a', 'blacklist_characters': 'b'}), (ds.characters, {'codec': 100}), (ds.characters, {'codec': 'this is not a valid codec name'}), (ds.characters, {'codec': 'ascii', 'include_characters': 'é'}), (ds.characters, {'codec': 'utf-8', 'categories': 'Cs'}), (ds.slices, {'size': None}), (ds.slices, {'size': 'chips'}), (ds.slices, {'size': -1}), (ds.slices, {'size': 2.3}), (ds.sampled_from, {'elements': ()}), (ds.ip_addresses, {'v': '4'}), (ds.ip_addresses, {'v': 4.0}), (ds.ip_addresses, {'v': 5}), (ds.ip_addresses, {'v': 4, 'network': '::/64'}), (ds.ip_addresses, {'v': 6, 'network': '127.0.0.0/8'}), (ds.ip_addresses, {'network': b'127.0.0.0/8'}), (ds.ip_addresses, {'network': b'::/64'}), (ds.randoms, {'use_true_random': 'False'}), (ds.randoms, {'note_method_calls': 'True'}))
def test_validates_keyword_arguments(fn, kwargs):
    if False:
        return 10
    with pytest.raises(InvalidArgument):
        fn(**kwargs).example()

@fn_ktest((ds.integers, {'min_value': 0}), (ds.integers, {'min_value': 11}), (ds.integers, {'min_value': 11, 'max_value': 100}), (ds.integers, {'max_value': 0}), (ds.integers, {'min_value': -2, 'max_value': -1}), (ds.decimals, {'min_value': 1.0, 'max_value': 1.5}), (ds.decimals, {'min_value': '1.0', 'max_value': '1.5'}), (ds.decimals, {'min_value': decimal.Decimal('1.5')}), (ds.decimals, {'max_value': 1.0, 'min_value': -1.0, 'allow_infinity': False}), (ds.decimals, {'min_value': 1.0, 'allow_nan': False}), (ds.decimals, {'max_value': 1.0, 'allow_nan': False}), (ds.decimals, {'max_value': 1.0, 'min_value': -1.0, 'allow_nan': False}), (ds.decimals, {'min_value': '-inf'}), (ds.decimals, {'max_value': 'inf'}), (ds.fractions, {'min_value': -1, 'max_value': 1, 'max_denominator': 1000}), (ds.fractions, {'min_value': 1, 'max_value': 1}), (ds.fractions, {'min_value': 1, 'max_value': 1, 'max_denominator': 2}), (ds.fractions, {'min_value': 1.0}), (ds.fractions, {'min_value': decimal.Decimal('1.0')}), (ds.fractions, {'min_value': fractions.Fraction(1, 2)}), (ds.fractions, {'min_value': '1/2', 'max_denominator': 2}), (ds.fractions, {'max_value': '1/2', 'max_denominator': 3}), (ds.lists, {'elements': ds.nothing(), 'max_size': 0}), (ds.lists, {'elements': ds.integers()}), (ds.lists, {'elements': ds.integers(), 'max_size': 5}), (ds.lists, {'elements': ds.booleans(), 'min_size': 5}), (ds.lists, {'elements': ds.booleans(), 'min_size': 5, 'max_size': 10}), (ds.sets, {'min_size': 10, 'max_size': 10, 'elements': ds.integers()}), (ds.booleans, {}), (ds.just, {'value': 'hi'}), (ds.integers, {'min_value': 12, 'max_value': 12}), (ds.floats, {}), (ds.floats, {'min_value': 1.0}), (ds.floats, {'max_value': 1.0}), (ds.floats, {'min_value': math.inf}), (ds.floats, {'max_value': -math.inf}), (ds.floats, {'max_value': 1.0, 'min_value': -1.0}), (ds.floats, {'max_value': 1.0, 'min_value': -1.0, 'allow_infinity': False}), (ds.floats, {'min_value': 1.0, 'allow_nan': False}), (ds.floats, {'max_value': 1.0, 'allow_nan': False}), (ds.floats, {'max_value': 1.0, 'min_value': -1.0, 'allow_nan': False}), (ds.complex_numbers, {}), (ds.complex_numbers, {'min_magnitude': 3, 'max_magnitude': 3}), (ds.complex_numbers, {'max_magnitude': 0}), (ds.complex_numbers, {'allow_nan': True}), (ds.complex_numbers, {'allow_nan': True, 'allow_infinity': True}), (ds.complex_numbers, {'allow_nan': True, 'allow_infinity': False}), (ds.complex_numbers, {'allow_nan': False}), (ds.complex_numbers, {'allow_nan': False, 'allow_infinity': True}), (ds.complex_numbers, {'allow_nan': False, 'allow_infinity': False}), (ds.complex_numbers, {'max_magnitude': math.inf, 'allow_infinity': True}), (ds.complex_numbers, {'width': 32}), (ds.complex_numbers, {'width': 64}), (ds.complex_numbers, {'width': 128}), (ds.sampled_from, {'elements': [1]}), (ds.sampled_from, {'elements': [1, 2, 3]}), (ds.fixed_dictionaries, {'mapping': {1: ds.integers()}}), (ds.dictionaries, {'keys': ds.booleans(), 'values': ds.integers()}), (ds.text, {'alphabet': 'abc'}), (ds.text, {'alphabet': set('abc')}), (ds.text, {'alphabet': ''}), (ds.text, {'alphabet': ds.just('a')}), (ds.text, {'alphabet': ds.sampled_from('abc')}), (ds.text, {'alphabet': ds.builds(lambda : 'a')}), (ds.characters, {'codec': 'ascii'}), (ds.characters, {'codec': 'latin1'}), (ds.characters, {'categories': ['N']}), (ds.characters, {'exclude_categories': []}), (ds.characters, {'whitelist_characters': 'a', 'codec': 'ascii'}), (ds.characters, {'blacklist_characters': 'a'}), (ds.characters, {'whitelist_categories': ['Nd']}), (ds.characters, {'blacklist_categories': ['Nd']}), (ds.from_regex, {'regex': 'abc', 'alphabet': 'abc'}), (ds.from_regex, {'regex': 'abc', 'alphabet': 'abcdef'}), (ds.from_regex, {'regex': '[abc]', 'alphabet': 'abcdef'}), (ds.from_regex, {'regex': '[a-f]', 'alphabet': 'abef'}), (ds.from_regex, {'regex': 'abc', 'alphabet': ds.sampled_from('abc')}), (ds.from_regex, {'regex': 'abc', 'alphabet': ds.characters(codec='ascii')}), (ds.ip_addresses, {}), (ds.ip_addresses, {'v': 4}), (ds.ip_addresses, {'v': 6}), (ds.ip_addresses, {'network': '127.0.0.0/8'}), (ds.ip_addresses, {'network': '::/64'}), (ds.ip_addresses, {'v': 4, 'network': '127.0.0.0/8'}), (ds.ip_addresses, {'v': 6, 'network': '::/64'}), (ds.ip_addresses, {'network': IPv4Network('127.0.0.0/8')}), (ds.ip_addresses, {'network': IPv6Network('::/64')}), (ds.ip_addresses, {'v': 4, 'network': IPv4Network('127.0.0.0/8')}), (ds.ip_addresses, {'v': 6, 'network': IPv6Network('::/64')}))
def test_produces_valid_examples_from_keyword(fn, kwargs):
    if False:
        return 10
    fn(**kwargs).example()

@fn_test((ds.one_of, (1,)), (ds.one_of, (1, ds.integers())), (ds.tuples, (1,)))
def test_validates_args(fn, args):
    if False:
        print('Hello World!')
    with pytest.raises(InvalidArgument):
        fn(*args).example()

@fn_test((ds.one_of, (ds.booleans(), ds.tuples(ds.booleans()))), (ds.one_of, (ds.booleans(),)), (ds.text, ()), (ds.binary, ()), (ds.builds, (lambda x, y: x + y, ds.integers(), ds.integers())))
def test_produces_valid_examples_from_args(fn, args):
    if False:
        while True:
            i = 10
    fn(*args).example()

def test_build_class_with_target_kwarg():
    if False:
        while True:
            i = 10
    NamedTupleWithTargetField = collections.namedtuple('Something', ['target'])
    ds.builds(NamedTupleWithTargetField, target=ds.integers()).example()

def test_builds_raises_with_no_target():
    if False:
        i = 10
        return i + 15
    with pytest.raises(TypeError):
        ds.builds().example()

@pytest.mark.parametrize('non_callable', [1, 'abc', ds.integers()])
def test_builds_raises_if_non_callable_as_target_kwarg(non_callable):
    if False:
        return 10
    with pytest.raises(TypeError):
        ds.builds(target=non_callable).example()

@pytest.mark.parametrize('non_callable', [1, 'abc', ds.integers()])
def test_builds_raises_if_non_callable_as_first_arg(non_callable):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(InvalidArgument):
        ds.builds(non_callable, target=lambda x: x).example()

def test_tuples_raise_error_on_bad_kwargs():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(TypeError):
        ds.tuples(stuff='things')

@given(ds.lists(ds.booleans(), min_size=10, max_size=10))
def test_has_specified_length(xs):
    if False:
        print('Hello World!')
    assert len(xs) == 10

@given(ds.integers(max_value=100))
@settings(max_examples=100)
def test_has_upper_bound(x):
    if False:
        return 10
    assert x <= 100

@given(ds.integers(min_value=100))
def test_has_lower_bound(x):
    if False:
        print('Hello World!')
    assert x >= 100

@given(ds.integers(min_value=1, max_value=2))
def test_is_in_bounds(x):
    if False:
        for i in range(10):
            print('nop')
    assert 1 <= x <= 2

@given(ds.fractions(min_value=-1, max_value=1, max_denominator=1000))
def test_fraction_is_in_bounds(x):
    if False:
        while True:
            i = 10
    assert -1 <= x <= 1
    assert abs(x.denominator) <= 1000

@given(ds.fractions(min_value=fractions.Fraction(1, 2)))
def test_fraction_gt_positive(x):
    if False:
        while True:
            i = 10
    assert fractions.Fraction(1, 2) <= x

@given(ds.fractions(max_value=fractions.Fraction(-1, 2)))
def test_fraction_lt_negative(x):
    if False:
        return 10
    assert x <= fractions.Fraction(-1, 2)

@given(ds.decimals(min_value=-1.5, max_value=1.5))
def test_decimal_is_in_bounds(x):
    if False:
        print('Hello World!')
    assert decimal.Decimal('-1.5') <= x <= decimal.Decimal('1.5')

def test_float_can_find_max_value_inf():
    if False:
        i = 10
        return i + 15
    assert minimal(ds.floats(max_value=math.inf), math.isinf) == float('inf')
    assert minimal(ds.floats(min_value=0.0), math.isinf) == math.inf

def test_float_can_find_min_value_inf():
    if False:
        i = 10
        return i + 15
    minimal(ds.floats(), lambda x: x < 0 and math.isinf(x))
    minimal(ds.floats(min_value=-math.inf, max_value=0.0), math.isinf)

def test_can_find_none_list():
    if False:
        print('Hello World!')
    assert minimal(ds.lists(ds.none()), lambda x: len(x) >= 3) == [None] * 3

def test_fractions():
    if False:
        return 10
    assert minimal(ds.fractions(), lambda f: f >= 1) == 1

def test_decimals():
    if False:
        return 10
    assert minimal(ds.decimals(), lambda f: f.is_finite() and f >= 1) == 1

def test_non_float_decimal():
    if False:
        for i in range(10):
            print('nop')
    minimal(ds.decimals(), lambda d: d.is_finite() and decimal.Decimal(float(d)) != d)

def test_produces_dictionaries_of_at_least_minimum_size():
    if False:
        i = 10
        return i + 15
    t = minimal(ds.dictionaries(ds.booleans(), ds.integers(), min_size=2), lambda x: True)
    assert t == {False: 0, True: 0}

@given(ds.dictionaries(ds.integers(), ds.integers(), max_size=5))
@settings(max_examples=50)
def test_dictionaries_respect_size(d):
    if False:
        while True:
            i = 10
    assert len(d) <= 5

@given(ds.dictionaries(ds.integers(), ds.integers(), max_size=0))
@settings(max_examples=50)
def test_dictionaries_respect_zero_size(d):
    if False:
        while True:
            i = 10
    assert len(d) <= 5

@given(ds.lists(ds.none(), max_size=5))
def test_none_lists_respect_max_size(ls):
    if False:
        i = 10
        return i + 15
    assert len(ls) <= 5

@given(ds.lists(ds.none(), max_size=5, min_size=1))
def test_none_lists_respect_max_and_min_size(ls):
    if False:
        i = 10
        return i + 15
    assert 1 <= len(ls) <= 5

@given(ds.iterables(ds.integers(), max_size=5, min_size=1))
def test_iterables_are_exhaustible(it):
    if False:
        for i in range(10):
            print('nop')
    for _ in it:
        pass
    with pytest.raises(StopIteration):
        next(it)

def test_minimal_iterable():
    if False:
        print('Hello World!')
    assert list(minimal(ds.iterables(ds.integers()), lambda x: True)) == []

@pytest.mark.parametrize('parameter_name', ['min_value', 'max_value'])
@pytest.mark.parametrize('value', [-1, 0, 1])
def test_no_infinity_for_min_max_values(value, parameter_name):
    if False:
        return 10
    kwargs = {'allow_infinity': False, parameter_name: value}

    @given(ds.floats(**kwargs))
    def test_not_infinite(xs):
        if False:
            while True:
                i = 10
        assert not math.isinf(xs)
    test_not_infinite()

@pytest.mark.parametrize('parameter_name', ['min_value', 'max_value'])
@pytest.mark.parametrize('value', [-1, 0, 1])
def test_no_nan_for_min_max_values(value, parameter_name):
    if False:
        for i in range(10):
            print('nop')
    kwargs = {'allow_nan': False, parameter_name: value}

    @given(ds.floats(**kwargs))
    def test_not_nan(xs):
        if False:
            while True:
                i = 10
        assert not math.isnan(xs)
    test_not_nan()

class Sneaky:
    """It's like a strategy, but it's not a strategy."""
    is_empty = False
    depth = 0
    label = 0

    def do_draw(self, data):
        if False:
            for i in range(10):
                print('nop')
        pass

    def validate(self):
        if False:
            print('Hello World!')
        pass

@pytest.mark.parametrize('value', [5, Sneaky()])
@pytest.mark.parametrize('label', [None, 'not a strategy'])
@given(data=ds.data())
def test_data_explicitly_rejects_non_strategies(data, value, label):
    if False:
        return 10
    with pytest.raises(InvalidArgument):
        data.draw(value, label=label)

@given(ds.integers().filter(bool).filter(lambda x: x % 3))
def test_chained_filter(x):
    if False:
        for i in range(10):
            print('nop')
    assert x
    assert x % 3

def test_chained_filter_tracks_all_conditions():
    if False:
        while True:
            i = 10
    s = ds.integers().filter(bool).filter(lambda x: x % 3)
    assert len(s.wrapped_strategy.flat_conditions) == 2

@pytest.mark.parametrize('version', [4, 6])
@given(data=ds.data())
def test_ipaddress_from_network_is_always_correct_version(data, version):
    if False:
        print('Hello World!')
    ip = data.draw(ds.ip_addresses(v=version), label='address')
    assert ip.version == version

@given(data=ds.data(), network=ds.from_type(IPv4Network) | ds.from_type(IPv6Network))
def test_ipaddress_from_network_is_always_in_network(data, network):
    if False:
        i = 10
        return i + 15
    ip = data.draw(ds.ip_addresses(network=network), label='address')
    assert ip in network
    assert ip.version == network.version

class AnEnum(enum.Enum):
    a = 1

def requires_arg(value):
    if False:
        print('Hello World!')
    'Similar to the enum.Enum.__call__ method.'

@given(ds.data())
def test_builds_error_messages(data):
    if False:
        while True:
            i = 10
    with pytest.raises(TypeError):
        requires_arg()
    with pytest.raises(TypeError):
        AnEnum()
    assert issubclass(InvalidArgument, TypeError)
    with pytest.raises(TypeError):
        data.draw(ds.builds(requires_arg))
    with pytest.raises(InvalidArgument, match='.* try using sampled_from\\(.+\\) instead of builds\\(.+\\)'):
        data.draw(ds.builds(AnEnum))
    data.draw(ds.sampled_from(AnEnum))