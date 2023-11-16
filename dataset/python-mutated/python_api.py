import math
import pprint
from collections.abc import Collection
from collections.abc import Sized
from decimal import Decimal
from numbers import Complex
from types import TracebackType
from typing import Any
from typing import Callable
from typing import cast
from typing import ContextManager
from typing import final
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Pattern
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import _pytest._code
from _pytest.compat import STRING_TYPES
from _pytest.outcomes import fail
if TYPE_CHECKING:
    from numpy import ndarray

def _non_numeric_type_error(value, at: Optional[str]) -> TypeError:
    if False:
        i = 10
        return i + 15
    at_str = f' at {at}' if at else ''
    return TypeError('cannot make approximate comparisons to non-numeric values: {!r} {}'.format(value, at_str))

def _compare_approx(full_object: object, message_data: Sequence[Tuple[str, str, str]], number_of_elements: int, different_ids: Sequence[object], max_abs_diff: float, max_rel_diff: float) -> List[str]:
    if False:
        return 10
    message_list = list(message_data)
    message_list.insert(0, ('Index', 'Obtained', 'Expected'))
    max_sizes = [0, 0, 0]
    for (index, obtained, expected) in message_list:
        max_sizes[0] = max(max_sizes[0], len(index))
        max_sizes[1] = max(max_sizes[1], len(obtained))
        max_sizes[2] = max(max_sizes[2], len(expected))
    explanation = [f'comparison failed. Mismatched elements: {len(different_ids)} / {number_of_elements}:', f'Max absolute difference: {max_abs_diff}', f'Max relative difference: {max_rel_diff}'] + [f'{indexes:<{max_sizes[0]}} | {obtained:<{max_sizes[1]}} | {expected:<{max_sizes[2]}}' for (indexes, obtained, expected) in message_list]
    return explanation

class ApproxBase:
    """Provide shared utilities for making approximate comparisons between
    numbers or sequences of numbers."""
    __array_ufunc__ = None
    __array_priority__ = 100

    def __init__(self, expected, rel=None, abs=None, nan_ok: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        __tracebackhide__ = True
        self.expected = expected
        self.abs = abs
        self.rel = rel
        self.nan_ok = nan_ok
        self._check_type()

    def __repr__(self) -> str:
        if False:
            return 10
        raise NotImplementedError

    def _repr_compare(self, other_side: Any) -> List[str]:
        if False:
            i = 10
            return i + 15
        return ['comparison failed', f'Obtained: {other_side}', f'Expected: {self}']

    def __eq__(self, actual) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return all((a == self._approx_scalar(x) for (a, x) in self._yield_comparisons(actual)))

    def __bool__(self):
        if False:
            for i in range(10):
                print('nop')
        __tracebackhide__ = True
        raise AssertionError('approx() is not supported in a boolean context.\nDid you mean: `assert a == approx(b)`?')
    __hash__ = None

    def __ne__(self, actual) -> bool:
        if False:
            print('Hello World!')
        return not actual == self

    def _approx_scalar(self, x) -> 'ApproxScalar':
        if False:
            i = 10
            return i + 15
        if isinstance(x, Decimal):
            return ApproxDecimal(x, rel=self.rel, abs=self.abs, nan_ok=self.nan_ok)
        return ApproxScalar(x, rel=self.rel, abs=self.abs, nan_ok=self.nan_ok)

    def _yield_comparisons(self, actual):
        if False:
            print('Hello World!')
        'Yield all the pairs of numbers to be compared.\n\n        This is used to implement the `__eq__` method.\n        '
        raise NotImplementedError

    def _check_type(self) -> None:
        if False:
            return 10
        'Raise a TypeError if the expected value is not a valid type.'

def _recursive_sequence_map(f, x):
    if False:
        return 10
    'Recursively map a function over a sequence of arbitrary depth'
    if isinstance(x, (list, tuple)):
        seq_type = type(x)
        return seq_type((_recursive_sequence_map(f, xi) for xi in x))
    else:
        return f(x)

class ApproxNumpy(ApproxBase):
    """Perform approximate comparisons where the expected value is numpy array."""

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        list_scalars = _recursive_sequence_map(self._approx_scalar, self.expected.tolist())
        return f'approx({list_scalars!r})'

    def _repr_compare(self, other_side: 'ndarray') -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        import itertools
        import math

        def get_value_from_nested_list(nested_list: List[Any], nd_index: Tuple[Any, ...]) -> Any:
            if False:
                print('Hello World!')
            "\n            Helper function to get the value out of a nested list, given an n-dimensional index.\n            This mimics numpy's indexing, but for raw nested python lists.\n            "
            value: Any = nested_list
            for i in nd_index:
                value = value[i]
            return value
        np_array_shape = self.expected.shape
        approx_side_as_seq = _recursive_sequence_map(self._approx_scalar, self.expected.tolist())
        if np_array_shape != other_side.shape:
            return ['Impossible to compare arrays with different shapes.', f'Shapes: {np_array_shape} and {other_side.shape}']
        number_of_elements = self.expected.size
        max_abs_diff = -math.inf
        max_rel_diff = -math.inf
        different_ids = []
        for index in itertools.product(*(range(i) for i in np_array_shape)):
            approx_value = get_value_from_nested_list(approx_side_as_seq, index)
            other_value = get_value_from_nested_list(other_side, index)
            if approx_value != other_value:
                abs_diff = abs(approx_value.expected - other_value)
                max_abs_diff = max(max_abs_diff, abs_diff)
                if other_value == 0.0:
                    max_rel_diff = math.inf
                else:
                    max_rel_diff = max(max_rel_diff, abs_diff / abs(other_value))
                different_ids.append(index)
        message_data = [(str(index), str(get_value_from_nested_list(other_side, index)), str(get_value_from_nested_list(approx_side_as_seq, index))) for index in different_ids]
        return _compare_approx(self.expected, message_data, number_of_elements, different_ids, max_abs_diff, max_rel_diff)

    def __eq__(self, actual) -> bool:
        if False:
            return 10
        import numpy as np
        if not np.isscalar(actual):
            try:
                actual = np.asarray(actual)
            except Exception as e:
                raise TypeError(f"cannot compare '{actual}' to numpy.ndarray") from e
        if not np.isscalar(actual) and actual.shape != self.expected.shape:
            return False
        return super().__eq__(actual)

    def _yield_comparisons(self, actual):
        if False:
            print('Hello World!')
        import numpy as np
        if np.isscalar(actual):
            for i in np.ndindex(self.expected.shape):
                yield (actual, self.expected[i].item())
        else:
            for i in np.ndindex(self.expected.shape):
                yield (actual[i].item(), self.expected[i].item())

class ApproxMapping(ApproxBase):
    """Perform approximate comparisons where the expected value is a mapping
    with numeric values (the keys can be anything)."""

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'approx({!r})'.format({k: self._approx_scalar(v) for (k, v) in self.expected.items()})

    def _repr_compare(self, other_side: Mapping[object, float]) -> List[str]:
        if False:
            while True:
                i = 10
        import math
        approx_side_as_map = {k: self._approx_scalar(v) for (k, v) in self.expected.items()}
        number_of_elements = len(approx_side_as_map)
        max_abs_diff = -math.inf
        max_rel_diff = -math.inf
        different_ids = []
        for ((approx_key, approx_value), other_value) in zip(approx_side_as_map.items(), other_side.values()):
            if approx_value != other_value:
                if approx_value.expected is not None and other_value is not None:
                    max_abs_diff = max(max_abs_diff, abs(approx_value.expected - other_value))
                    if approx_value.expected == 0.0:
                        max_rel_diff = math.inf
                    else:
                        max_rel_diff = max(max_rel_diff, abs((approx_value.expected - other_value) / approx_value.expected))
                different_ids.append(approx_key)
        message_data = [(str(key), str(other_side[key]), str(approx_side_as_map[key])) for key in different_ids]
        return _compare_approx(self.expected, message_data, number_of_elements, different_ids, max_abs_diff, max_rel_diff)

    def __eq__(self, actual) -> bool:
        if False:
            print('Hello World!')
        try:
            if set(actual.keys()) != set(self.expected.keys()):
                return False
        except AttributeError:
            return False
        return super().__eq__(actual)

    def _yield_comparisons(self, actual):
        if False:
            print('Hello World!')
        for k in self.expected.keys():
            yield (actual[k], self.expected[k])

    def _check_type(self) -> None:
        if False:
            i = 10
            return i + 15
        __tracebackhide__ = True
        for (key, value) in self.expected.items():
            if isinstance(value, type(self.expected)):
                msg = 'pytest.approx() does not support nested dictionaries: key={!r} value={!r}\n  full mapping={}'
                raise TypeError(msg.format(key, value, pprint.pformat(self.expected)))

class ApproxSequenceLike(ApproxBase):
    """Perform approximate comparisons where the expected value is a sequence of numbers."""

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        seq_type = type(self.expected)
        if seq_type not in (tuple, list):
            seq_type = list
        return 'approx({!r})'.format(seq_type((self._approx_scalar(x) for x in self.expected)))

    def _repr_compare(self, other_side: Sequence[float]) -> List[str]:
        if False:
            while True:
                i = 10
        import math
        if len(self.expected) != len(other_side):
            return ['Impossible to compare lists with different sizes.', f'Lengths: {len(self.expected)} and {len(other_side)}']
        approx_side_as_map = _recursive_sequence_map(self._approx_scalar, self.expected)
        number_of_elements = len(approx_side_as_map)
        max_abs_diff = -math.inf
        max_rel_diff = -math.inf
        different_ids = []
        for (i, (approx_value, other_value)) in enumerate(zip(approx_side_as_map, other_side)):
            if approx_value != other_value:
                abs_diff = abs(approx_value.expected - other_value)
                max_abs_diff = max(max_abs_diff, abs_diff)
                if other_value == 0.0:
                    max_rel_diff = math.inf
                else:
                    max_rel_diff = max(max_rel_diff, abs_diff / abs(other_value))
                different_ids.append(i)
        message_data = [(str(i), str(other_side[i]), str(approx_side_as_map[i])) for i in different_ids]
        return _compare_approx(self.expected, message_data, number_of_elements, different_ids, max_abs_diff, max_rel_diff)

    def __eq__(self, actual) -> bool:
        if False:
            print('Hello World!')
        try:
            if len(actual) != len(self.expected):
                return False
        except TypeError:
            return False
        return super().__eq__(actual)

    def _yield_comparisons(self, actual):
        if False:
            while True:
                i = 10
        return zip(actual, self.expected)

    def _check_type(self) -> None:
        if False:
            return 10
        __tracebackhide__ = True
        for (index, x) in enumerate(self.expected):
            if isinstance(x, type(self.expected)):
                msg = 'pytest.approx() does not support nested data structures: {!r} at index {}\n  full sequence: {}'
                raise TypeError(msg.format(x, index, pprint.pformat(self.expected)))

class ApproxScalar(ApproxBase):
    """Perform approximate comparisons where the expected value is a single number."""
    DEFAULT_ABSOLUTE_TOLERANCE: Union[float, Decimal] = 1e-12
    DEFAULT_RELATIVE_TOLERANCE: Union[float, Decimal] = 1e-06

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        'Return a string communicating both the expected value and the\n        tolerance for the comparison being made.\n\n        For example, ``1.0 ± 1e-6``, ``(3+4j) ± 5e-6 ∠ ±180°``.\n        '
        if not isinstance(self.expected, (Complex, Decimal)) or math.isinf(abs(self.expected)):
            return str(self.expected)
        try:
            vetted_tolerance = f'{self.tolerance:.1e}'
            if isinstance(self.expected, Complex) and self.expected.imag and (not math.isinf(self.tolerance)):
                vetted_tolerance += ' ∠ ±180°'
        except ValueError:
            vetted_tolerance = '???'
        return f'{self.expected} ± {vetted_tolerance}'

    def __eq__(self, actual) -> bool:
        if False:
            while True:
                i = 10
        'Return whether the given value is equal to the expected value\n        within the pre-specified tolerance.'
        asarray = _as_numpy_array(actual)
        if asarray is not None:
            return all((self.__eq__(a) for a in asarray.flat))
        if actual == self.expected:
            return True
        if not (isinstance(self.expected, (Complex, Decimal)) and isinstance(actual, (Complex, Decimal))):
            return False
        if math.isnan(abs(self.expected)):
            return self.nan_ok and math.isnan(abs(actual))
        if math.isinf(abs(self.expected)):
            return False
        result: bool = abs(self.expected - actual) <= self.tolerance
        return result
    __hash__ = None

    @property
    def tolerance(self):
        if False:
            print('Hello World!')
        'Return the tolerance for the comparison.\n\n        This could be either an absolute tolerance or a relative tolerance,\n        depending on what the user specified or which would be larger.\n        '

        def set_default(x, default):
            if False:
                for i in range(10):
                    print('nop')
            return x if x is not None else default
        absolute_tolerance = set_default(self.abs, self.DEFAULT_ABSOLUTE_TOLERANCE)
        if absolute_tolerance < 0:
            raise ValueError(f"absolute tolerance can't be negative: {absolute_tolerance}")
        if math.isnan(absolute_tolerance):
            raise ValueError("absolute tolerance can't be NaN.")
        if self.rel is None:
            if self.abs is not None:
                return absolute_tolerance
        relative_tolerance = set_default(self.rel, self.DEFAULT_RELATIVE_TOLERANCE) * abs(self.expected)
        if relative_tolerance < 0:
            raise ValueError(f"relative tolerance can't be negative: {relative_tolerance}")
        if math.isnan(relative_tolerance):
            raise ValueError("relative tolerance can't be NaN.")
        return max(relative_tolerance, absolute_tolerance)

class ApproxDecimal(ApproxScalar):
    """Perform approximate comparisons where the expected value is a Decimal."""
    DEFAULT_ABSOLUTE_TOLERANCE = Decimal('1e-12')
    DEFAULT_RELATIVE_TOLERANCE = Decimal('1e-6')

def approx(expected, rel=None, abs=None, nan_ok: bool=False) -> ApproxBase:
    if False:
        return 10
    'Assert that two numbers (or two ordered sequences of numbers) are equal to each other\n    within some tolerance.\n\n    Due to the :doc:`python:tutorial/floatingpoint`, numbers that we\n    would intuitively expect to be equal are not always so::\n\n        >>> 0.1 + 0.2 == 0.3\n        False\n\n    This problem is commonly encountered when writing tests, e.g. when making\n    sure that floating-point values are what you expect them to be.  One way to\n    deal with this problem is to assert that two floating-point numbers are\n    equal to within some appropriate tolerance::\n\n        >>> abs((0.1 + 0.2) - 0.3) < 1e-6\n        True\n\n    However, comparisons like this are tedious to write and difficult to\n    understand.  Furthermore, absolute comparisons like the one above are\n    usually discouraged because there\'s no tolerance that works well for all\n    situations.  ``1e-6`` is good for numbers around ``1``, but too small for\n    very big numbers and too big for very small ones.  It\'s better to express\n    the tolerance as a fraction of the expected value, but relative comparisons\n    like that are even more difficult to write correctly and concisely.\n\n    The ``approx`` class performs floating-point comparisons using a syntax\n    that\'s as intuitive as possible::\n\n        >>> from pytest import approx\n        >>> 0.1 + 0.2 == approx(0.3)\n        True\n\n    The same syntax also works for ordered sequences of numbers::\n\n        >>> (0.1 + 0.2, 0.2 + 0.4) == approx((0.3, 0.6))\n        True\n\n    ``numpy`` arrays::\n\n        >>> import numpy as np                                                          # doctest: +SKIP\n        >>> np.array([0.1, 0.2]) + np.array([0.2, 0.4]) == approx(np.array([0.3, 0.6])) # doctest: +SKIP\n        True\n\n    And for a ``numpy`` array against a scalar::\n\n        >>> import numpy as np                                         # doctest: +SKIP\n        >>> np.array([0.1, 0.2]) + np.array([0.2, 0.1]) == approx(0.3) # doctest: +SKIP\n        True\n\n    Only ordered sequences are supported, because ``approx`` needs\n    to infer the relative position of the sequences without ambiguity. This means\n    ``sets`` and other unordered sequences are not supported.\n\n    Finally, dictionary *values* can also be compared::\n\n        >>> {\'a\': 0.1 + 0.2, \'b\': 0.2 + 0.4} == approx({\'a\': 0.3, \'b\': 0.6})\n        True\n\n    The comparison will be true if both mappings have the same keys and their\n    respective values match the expected tolerances.\n\n    **Tolerances**\n\n    By default, ``approx`` considers numbers within a relative tolerance of\n    ``1e-6`` (i.e. one part in a million) of its expected value to be equal.\n    This treatment would lead to surprising results if the expected value was\n    ``0.0``, because nothing but ``0.0`` itself is relatively close to ``0.0``.\n    To handle this case less surprisingly, ``approx`` also considers numbers\n    within an absolute tolerance of ``1e-12`` of its expected value to be\n    equal.  Infinity and NaN are special cases.  Infinity is only considered\n    equal to itself, regardless of the relative tolerance.  NaN is not\n    considered equal to anything by default, but you can make it be equal to\n    itself by setting the ``nan_ok`` argument to True.  (This is meant to\n    facilitate comparing arrays that use NaN to mean "no data".)\n\n    Both the relative and absolute tolerances can be changed by passing\n    arguments to the ``approx`` constructor::\n\n        >>> 1.0001 == approx(1)\n        False\n        >>> 1.0001 == approx(1, rel=1e-3)\n        True\n        >>> 1.0001 == approx(1, abs=1e-3)\n        True\n\n    If you specify ``abs`` but not ``rel``, the comparison will not consider\n    the relative tolerance at all.  In other words, two numbers that are within\n    the default relative tolerance of ``1e-6`` will still be considered unequal\n    if they exceed the specified absolute tolerance.  If you specify both\n    ``abs`` and ``rel``, the numbers will be considered equal if either\n    tolerance is met::\n\n        >>> 1 + 1e-8 == approx(1)\n        True\n        >>> 1 + 1e-8 == approx(1, abs=1e-12)\n        False\n        >>> 1 + 1e-8 == approx(1, rel=1e-6, abs=1e-12)\n        True\n\n    You can also use ``approx`` to compare nonnumeric types, or dicts and\n    sequences containing nonnumeric types, in which case it falls back to\n    strict equality. This can be useful for comparing dicts and sequences that\n    can contain optional values::\n\n        >>> {"required": 1.0000005, "optional": None} == approx({"required": 1, "optional": None})\n        True\n        >>> [None, 1.0000005] == approx([None,1])\n        True\n        >>> ["foo", 1.0000005] == approx([None,1])\n        False\n\n    If you\'re thinking about using ``approx``, then you might want to know how\n    it compares to other good ways of comparing floating-point numbers.  All of\n    these algorithms are based on relative and absolute tolerances and should\n    agree for the most part, but they do have meaningful differences:\n\n    - ``math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)``:  True if the relative\n      tolerance is met w.r.t. either ``a`` or ``b`` or if the absolute\n      tolerance is met.  Because the relative tolerance is calculated w.r.t.\n      both ``a`` and ``b``, this test is symmetric (i.e.  neither ``a`` nor\n      ``b`` is a "reference value").  You have to specify an absolute tolerance\n      if you want to compare to ``0.0`` because there is no tolerance by\n      default.  More information: :py:func:`math.isclose`.\n\n    - ``numpy.isclose(a, b, rtol=1e-5, atol=1e-8)``: True if the difference\n      between ``a`` and ``b`` is less that the sum of the relative tolerance\n      w.r.t. ``b`` and the absolute tolerance.  Because the relative tolerance\n      is only calculated w.r.t. ``b``, this test is asymmetric and you can\n      think of ``b`` as the reference value.  Support for comparing sequences\n      is provided by :py:func:`numpy.allclose`.  More information:\n      :std:doc:`numpy:reference/generated/numpy.isclose`.\n\n    - ``unittest.TestCase.assertAlmostEqual(a, b)``: True if ``a`` and ``b``\n      are within an absolute tolerance of ``1e-7``.  No relative tolerance is\n      considered , so this function is not appropriate for very large or very\n      small numbers.  Also, it\'s only available in subclasses of ``unittest.TestCase``\n      and it\'s ugly because it doesn\'t follow PEP8.  More information:\n      :py:meth:`unittest.TestCase.assertAlmostEqual`.\n\n    - ``a == pytest.approx(b, rel=1e-6, abs=1e-12)``: True if the relative\n      tolerance is met w.r.t. ``b`` or if the absolute tolerance is met.\n      Because the relative tolerance is only calculated w.r.t. ``b``, this test\n      is asymmetric and you can think of ``b`` as the reference value.  In the\n      special case that you explicitly specify an absolute tolerance but not a\n      relative tolerance, only the absolute tolerance is considered.\n\n    .. note::\n\n        ``approx`` can handle numpy arrays, but we recommend the\n        specialised test helpers in :std:doc:`numpy:reference/routines.testing`\n        if you need support for comparisons, NaNs, or ULP-based tolerances.\n\n        To match strings using regex, you can use\n        `Matches <https://github.com/asottile/re-assert#re_assertmatchespattern-str-args-kwargs>`_\n        from the\n        `re_assert package <https://github.com/asottile/re-assert>`_.\n\n    .. warning::\n\n       .. versionchanged:: 3.2\n\n       In order to avoid inconsistent behavior, :py:exc:`TypeError` is\n       raised for ``>``, ``>=``, ``<`` and ``<=`` comparisons.\n       The example below illustrates the problem::\n\n           assert approx(0.1) > 0.1 + 1e-10  # calls approx(0.1).__gt__(0.1 + 1e-10)\n           assert 0.1 + 1e-10 > approx(0.1)  # calls approx(0.1).__lt__(0.1 + 1e-10)\n\n       In the second example one expects ``approx(0.1).__le__(0.1 + 1e-10)``\n       to be called. But instead, ``approx(0.1).__lt__(0.1 + 1e-10)`` is used to\n       comparison. This is because the call hierarchy of rich comparisons\n       follows a fixed behavior. More information: :py:meth:`object.__ge__`\n\n    .. versionchanged:: 3.7.1\n       ``approx`` raises ``TypeError`` when it encounters a dict value or\n       sequence element of nonnumeric type.\n\n    .. versionchanged:: 6.1.0\n       ``approx`` falls back to strict equality for nonnumeric types instead\n       of raising ``TypeError``.\n    '
    __tracebackhide__ = True
    if isinstance(expected, Decimal):
        cls: Type[ApproxBase] = ApproxDecimal
    elif isinstance(expected, Mapping):
        cls = ApproxMapping
    elif _is_numpy_array(expected):
        expected = _as_numpy_array(expected)
        cls = ApproxNumpy
    elif hasattr(expected, '__getitem__') and isinstance(expected, Sized) and (not isinstance(expected, STRING_TYPES)):
        cls = ApproxSequenceLike
    elif isinstance(expected, Collection) and (not isinstance(expected, STRING_TYPES)):
        msg = f'pytest.approx() only supports ordered sequences, but got: {repr(expected)}'
        raise TypeError(msg)
    else:
        cls = ApproxScalar
    return cls(expected, rel, abs, nan_ok)

def _is_numpy_array(obj: object) -> bool:
    if False:
        print('Hello World!')
    '\n    Return true if the given object is implicitly convertible to ndarray,\n    and numpy is already imported.\n    '
    return _as_numpy_array(obj) is not None

def _as_numpy_array(obj: object) -> Optional['ndarray']:
    if False:
        while True:
            i = 10
    '\n    Return an ndarray if the given object is implicitly convertible to ndarray,\n    and numpy is already imported, otherwise None.\n    '
    import sys
    np: Any = sys.modules.get('numpy')
    if np is not None:
        if np.isscalar(obj):
            return None
        elif isinstance(obj, np.ndarray):
            return obj
        elif hasattr(obj, '__array__') or hasattr('obj', '__array_interface__'):
            return np.asarray(obj)
    return None
E = TypeVar('E', bound=BaseException)

@overload
def raises(expected_exception: Union[Type[E], Tuple[Type[E], ...]], *, match: Optional[Union[str, Pattern[str]]]=...) -> 'RaisesContext[E]':
    if False:
        while True:
            i = 10
    ...

@overload
def raises(expected_exception: Union[Type[E], Tuple[Type[E], ...]], func: Callable[..., Any], *args: Any, **kwargs: Any) -> _pytest._code.ExceptionInfo[E]:
    if False:
        i = 10
        return i + 15
    ...

def raises(expected_exception: Union[Type[E], Tuple[Type[E], ...]], *args: Any, **kwargs: Any) -> Union['RaisesContext[E]', _pytest._code.ExceptionInfo[E]]:
    if False:
        i = 10
        return i + 15
    'Assert that a code block/function call raises an exception type, or one of its subclasses.\n\n    :param typing.Type[E] | typing.Tuple[typing.Type[E], ...] expected_exception:\n        The expected exception type, or a tuple if one of multiple possible\n        exception types are expected. Note that subclasses of the passed exceptions\n        will also match.\n\n    :kwparam str | typing.Pattern[str] | None match:\n        If specified, a string containing a regular expression,\n        or a regular expression object, that is tested against the string\n        representation of the exception and its `PEP-678 <https://peps.python.org/pep-0678/>` `__notes__`\n        using :func:`re.search`.\n\n        To match a literal string that may contain :ref:`special characters\n        <re-syntax>`, the pattern can first be escaped with :func:`re.escape`.\n\n        (This is only used when :py:func:`pytest.raises` is used as a context manager,\n        and passed through to the function otherwise.\n        When using :py:func:`pytest.raises` as a function, you can use:\n        ``pytest.raises(Exc, func, match="passed on").match("my pattern")``.)\n\n    .. currentmodule:: _pytest._code\n\n    Use ``pytest.raises`` as a context manager, which will capture the exception of the given\n    type, or any of its subclasses::\n\n        >>> import pytest\n        >>> with pytest.raises(ZeroDivisionError):\n        ...    1/0\n\n    If the code block does not raise the expected exception (:class:`ZeroDivisionError` in the example\n    above), or no exception at all, the check will fail instead.\n\n    You can also use the keyword argument ``match`` to assert that the\n    exception matches a text or regex::\n\n        >>> with pytest.raises(ValueError, match=\'must be 0 or None\'):\n        ...     raise ValueError("value must be 0 or None")\n\n        >>> with pytest.raises(ValueError, match=r\'must be \\d+$\'):\n        ...     raise ValueError("value must be 42")\n\n    The ``match`` argument searches the formatted exception string, which includes any\n    `PEP-678 <https://peps.python.org/pep-0678/>`__ ``__notes__``:\n\n        >>> with pytest.raises(ValueError, match=r\'had a note added\'):  # doctest: +SKIP\n        ...    e = ValueError("value must be 42")\n        ...    e.add_note("had a note added")\n        ...    raise e\n\n    The context manager produces an :class:`ExceptionInfo` object which can be used to inspect the\n    details of the captured exception::\n\n        >>> with pytest.raises(ValueError) as exc_info:\n        ...     raise ValueError("value must be 42")\n        >>> assert exc_info.type is ValueError\n        >>> assert exc_info.value.args[0] == "value must be 42"\n\n    .. warning::\n\n       Given that ``pytest.raises`` matches subclasses, be wary of using it to match :class:`Exception` like this::\n\n           with pytest.raises(Exception):  # Careful, this will catch ANY exception raised.\n                some_function()\n\n       Because :class:`Exception` is the base class of almost all exceptions, it is easy for this to hide\n       real bugs, where the user wrote this expecting a specific exception, but some other exception is being\n       raised due to a bug introduced during a refactoring.\n\n       Avoid using ``pytest.raises`` to catch :class:`Exception` unless certain that you really want to catch\n       **any** exception raised.\n\n    .. note::\n\n       When using ``pytest.raises`` as a context manager, it\'s worthwhile to\n       note that normal context manager rules apply and that the exception\n       raised *must* be the final line in the scope of the context manager.\n       Lines of code after that, within the scope of the context manager will\n       not be executed. For example::\n\n           >>> value = 15\n           >>> with pytest.raises(ValueError) as exc_info:\n           ...     if value > 10:\n           ...         raise ValueError("value must be <= 10")\n           ...     assert exc_info.type is ValueError  # This will not execute.\n\n       Instead, the following approach must be taken (note the difference in\n       scope)::\n\n           >>> with pytest.raises(ValueError) as exc_info:\n           ...     if value > 10:\n           ...         raise ValueError("value must be <= 10")\n           ...\n           >>> assert exc_info.type is ValueError\n\n    **Using with** ``pytest.mark.parametrize``\n\n    When using :ref:`pytest.mark.parametrize ref`\n    it is possible to parametrize tests such that\n    some runs raise an exception and others do not.\n\n    See :ref:`parametrizing_conditional_raising` for an example.\n\n    .. seealso::\n\n        :ref:`assertraises` for more examples and detailed discussion.\n\n    **Legacy form**\n\n    It is possible to specify a callable by passing a to-be-called lambda::\n\n        >>> raises(ZeroDivisionError, lambda: 1/0)\n        <ExceptionInfo ...>\n\n    or you can specify an arbitrary callable with arguments::\n\n        >>> def f(x): return 1/x\n        ...\n        >>> raises(ZeroDivisionError, f, 0)\n        <ExceptionInfo ...>\n        >>> raises(ZeroDivisionError, f, x=0)\n        <ExceptionInfo ...>\n\n    The form above is fully supported but discouraged for new code because the\n    context manager form is regarded as more readable and less error-prone.\n\n    .. note::\n        Similar to caught exception objects in Python, explicitly clearing\n        local references to returned ``ExceptionInfo`` objects can\n        help the Python interpreter speed up its garbage collection.\n\n        Clearing those references breaks a reference cycle\n        (``ExceptionInfo`` --> caught exception --> frame stack raising\n        the exception --> current frame stack --> local variables -->\n        ``ExceptionInfo``) which makes Python keep all objects referenced\n        from that cycle (including all local variables in the current\n        frame) alive until the next cyclic garbage collection run.\n        More detailed information can be found in the official Python\n        documentation for :ref:`the try statement <python:try>`.\n    '
    __tracebackhide__ = True
    if not expected_exception:
        raise ValueError(f"Expected an exception type or a tuple of exception types, but got `{expected_exception!r}`. Raising exceptions is already understood as failing the test, so you don't need any special code to say 'this should never raise an exception'.")
    if isinstance(expected_exception, type):
        expected_exceptions: Tuple[Type[E], ...] = (expected_exception,)
    else:
        expected_exceptions = expected_exception
    for exc in expected_exceptions:
        if not isinstance(exc, type) or not issubclass(exc, BaseException):
            msg = 'expected exception must be a BaseException type, not {}'
            not_a = exc.__name__ if isinstance(exc, type) else type(exc).__name__
            raise TypeError(msg.format(not_a))
    message = f'DID NOT RAISE {expected_exception}'
    if not args:
        match: Optional[Union[str, Pattern[str]]] = kwargs.pop('match', None)
        if kwargs:
            msg = 'Unexpected keyword arguments passed to pytest.raises: '
            msg += ', '.join(sorted(kwargs))
            msg += '\nUse context-manager form instead?'
            raise TypeError(msg)
        return RaisesContext(expected_exception, message, match)
    else:
        func = args[0]
        if not callable(func):
            raise TypeError(f'{func!r} object (type: {type(func)}) must be callable')
        try:
            func(*args[1:], **kwargs)
        except expected_exception as e:
            return _pytest._code.ExceptionInfo.from_exception(e)
    fail(message)
raises.Exception = fail.Exception

@final
class RaisesContext(ContextManager[_pytest._code.ExceptionInfo[E]]):

    def __init__(self, expected_exception: Union[Type[E], Tuple[Type[E], ...]], message: str, match_expr: Optional[Union[str, Pattern[str]]]=None) -> None:
        if False:
            return 10
        self.expected_exception = expected_exception
        self.message = message
        self.match_expr = match_expr
        self.excinfo: Optional[_pytest._code.ExceptionInfo[E]] = None

    def __enter__(self) -> _pytest._code.ExceptionInfo[E]:
        if False:
            return 10
        self.excinfo = _pytest._code.ExceptionInfo.for_later()
        return self.excinfo

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> bool:
        if False:
            i = 10
            return i + 15
        __tracebackhide__ = True
        if exc_type is None:
            fail(self.message)
        assert self.excinfo is not None
        if not issubclass(exc_type, self.expected_exception):
            return False
        exc_info = cast(Tuple[Type[E], E, TracebackType], (exc_type, exc_val, exc_tb))
        self.excinfo.fill_unfilled(exc_info)
        if self.match_expr is not None:
            self.excinfo.match(self.match_expr)
        return True