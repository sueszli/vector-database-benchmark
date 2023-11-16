from typing import Any, Dict, List, Mapping, MutableMapping, MutableSet, MutableSequence, Set, Sequence, Tuple, Union
try:
    from typing_extensions import TypedDict
except ImportError:
    from typing import TypedDict
from robot.api.deco import not_keyword
TypedDict = not_keyword(TypedDict)

class Point2D(TypedDict):
    x: int
    y: int

class Point(Point2D, total=False):
    z: int

class Stringified(TypedDict):
    a: 'int'
    b: 'int | float'

class BadIntMeta(type(int)):

    def __instancecheck__(self, instance):
        if False:
            i = 10
            return i + 15
        raise TypeError('Bang!')

class BadInt(int, metaclass=BadIntMeta):
    pass

def list_(argument: List, expected=None):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected)

def list_with_types(argument: List[int], expected=None, same=False):
    if False:
        i = 10
        return i + 15
    _validate_type(argument, expected, same)

def tuple_(argument: Tuple, expected=None):
    if False:
        while True:
            i = 10
    _validate_type(argument, expected)

def tuple_with_types(argument: Tuple[bool, int], expected=None, same=False):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected, same)

def homogenous_tuple(argument: Tuple[int, ...], expected=None, same=False):
    if False:
        print('Hello World!')
    _validate_type(argument, expected, same)

def sequence(argument: Sequence, expected=None):
    if False:
        return 10
    _validate_type(argument, expected)

def sequence_with_types(argument: Sequence[Union[int, float]], expected=None):
    if False:
        i = 10
        return i + 15
    _validate_type(argument, expected)

def mutable_sequence(argument: MutableSequence, expected=None):
    if False:
        i = 10
        return i + 15
    _validate_type(argument, expected)

def mutable_sequence_with_types(argument: MutableSequence[int], expected=None):
    if False:
        print('Hello World!')
    _validate_type(argument, expected)

def dict_(argument: Dict, expected=None):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected)

def dict_with_types(argument: Dict[int, float], expected=None, same=False):
    if False:
        i = 10
        return i + 15
    _validate_type(argument, expected, same)

def mapping(argument: Mapping, expected=None):
    if False:
        while True:
            i = 10
    _validate_type(argument, expected)

def mapping_with_types(argument: Mapping[int, float], expected=None):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected)

def mutable_mapping(argument: MutableMapping, expected=None):
    if False:
        return 10
    _validate_type(argument, expected)

def mutable_mapping_with_types(argument: MutableMapping[int, float], expected=None):
    if False:
        return 10
    _validate_type(argument, expected)

def typeddict(argument: Point2D, expected=None):
    if False:
        i = 10
        return i + 15
    _validate_type(argument, expected)

def typeddict_with_optional(argument: Point, expected=None):
    if False:
        i = 10
        return i + 15
    _validate_type(argument, expected)

def stringified_typeddict(argument: Stringified, expected=None):
    if False:
        i = 10
        return i + 15
    _validate_type(argument, expected)

def set_(argument: Set, expected=None):
    if False:
        while True:
            i = 10
    _validate_type(argument, expected)

def set_with_types(argument: Set[int], expected=None, same=False):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(argument, expected, same)

def mutable_set(argument: MutableSet, expected=None):
    if False:
        while True:
            i = 10
    _validate_type(argument, expected)

def mutable_set_with_types(argument: MutableSet[float], expected=None):
    if False:
        print('Hello World!')
    _validate_type(argument, expected)

def any_(argument: Any=1, expected=None):
    if False:
        while True:
            i = 10
    _validate_type(argument, expected)

def none_as_default(argument: List=None, expected=None):
    if False:
        while True:
            i = 10
    _validate_type(argument, expected)

def none_as_default_with_any(argument: Any=None, expected=None):
    if False:
        print('Hello World!')
    _validate_type(argument, expected)

def forward_reference(argument: 'List', expected=None):
    if False:
        i = 10
        return i + 15
    _validate_type(argument, expected)

def forward_ref_with_types(argument: 'List[int]', expected=None):
    if False:
        print('Hello World!')
    _validate_type(argument, expected)

def not_liking_isinstance(argument: BadInt, expected=None):
    if False:
        return 10
    _validate_type(argument, expected)

def _validate_type(argument, expected, same=False, evaluate=True):
    if False:
        return 10
    if isinstance(expected, str) and evaluate:
        expected = eval(expected)
    if argument != expected or type(argument) != type(expected):
        atype = type(argument).__name__
        etype = type(expected).__name__
        raise AssertionError(f'{argument!r} ({atype}) != {expected!r} ({etype})')
    if isinstance(argument, (list, tuple)):
        for (a, e) in zip(argument, expected):
            _validate_type(a, e, same, evaluate=False)
    if isinstance(argument, dict):
        for (a, e) in zip(argument, expected):
            _validate_type(a, e, same, evaluate=False)
            _validate_type(argument[a], expected[e], same, evaluate=False)
    if same and argument is not expected:
        raise AssertionError(f'{argument} (id: {id(argument)}) is not same as {expected} (id: {id(expected)})')