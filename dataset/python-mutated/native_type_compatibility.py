"""Module to convert Python's native typing types to Beam types."""
import collections
import logging
import sys
import types
import typing
from apache_beam.typehints import typehints
_LOGGER = logging.getLogger(__name__)
_TypeMapEntry = collections.namedtuple('_TypeMapEntry', ['match', 'arity', 'beam_type'])
_BUILTINS_TO_TYPING = {dict: typing.Dict, list: typing.List, tuple: typing.Tuple, set: typing.Set, frozenset: typing.FrozenSet}
_CONVERTED_COLLECTIONS = [collections.abc.Set, collections.abc.MutableSet, collections.abc.Collection]

def _get_args(typ):
    if False:
        print('Hello World!')
    'Returns a list of arguments to the given type.\n\n  Args:\n    typ: A typing module typing type.\n\n  Returns:\n    A tuple of args.\n  '
    try:
        if typ.__args__ is None:
            return ()
        return typ.__args__
    except AttributeError:
        if isinstance(typ, typing.TypeVar):
            return (typ.__name__,)
        return ()

def _safe_issubclass(derived, parent):
    if False:
        i = 10
        return i + 15
    "Like issubclass, but swallows TypeErrors.\n\n  This is useful for when either parameter might not actually be a class,\n  e.g. typing.Union isn't actually a class.\n\n  Args:\n    derived: As in issubclass.\n    parent: As in issubclass.\n\n  Returns:\n    issubclass(derived, parent), or False if a TypeError was raised.\n  "
    try:
        return issubclass(derived, parent)
    except (TypeError, AttributeError):
        if hasattr(derived, '__origin__'):
            try:
                return issubclass(derived.__origin__, parent)
            except TypeError:
                pass
        return False

def _match_issubclass(match_against):
    if False:
        while True:
            i = 10
    return lambda user_type: _safe_issubclass(user_type, match_against)

def _match_is_exactly_mapping(user_type):
    if False:
        i = 10
        return i + 15
    if sys.version_info < (3, 7):
        expected_origin = typing.Mapping
    else:
        expected_origin = collections.abc.Mapping
    return getattr(user_type, '__origin__', None) is expected_origin

def _match_is_exactly_iterable(user_type):
    if False:
        while True:
            i = 10
    if user_type is typing.Iterable:
        return True
    if sys.version_info < (3, 7):
        expected_origin = typing.Iterable
    else:
        expected_origin = collections.abc.Iterable
    return getattr(user_type, '__origin__', None) is expected_origin

def _match_is_exactly_collection(user_type):
    if False:
        return 10
    return getattr(user_type, '__origin__', None) is collections.abc.Collection

def match_is_named_tuple(user_type):
    if False:
        for i in range(10):
            print('nop')
    return _safe_issubclass(user_type, typing.Tuple) and hasattr(user_type, '__annotations__')

def _match_is_optional(user_type):
    if False:
        while True:
            i = 10
    return _match_is_union(user_type) and sum((tp is type(None) for tp in _get_args(user_type))) == 1

def extract_optional_type(user_type):
    if False:
        i = 10
        return i + 15
    'Extracts the non-None type from Optional type user_type.\n\n  If user_type is not Optional, returns None\n  '
    if not _match_is_optional(user_type):
        return None
    else:
        return next((tp for tp in _get_args(user_type) if tp is not type(None)))

def _match_is_union(user_type):
    if False:
        while True:
            i = 10
    if user_type is typing.Union:
        return True
    try:
        return user_type.__origin__ is typing.Union
    except AttributeError:
        pass
    return False

def match_is_set(user_type):
    if False:
        while True:
            i = 10
    if _safe_issubclass(user_type, typing.Set):
        return True
    elif getattr(user_type, '__origin__', None) is not None:
        return _safe_issubclass(user_type.__origin__, collections.abc.Set)
    else:
        return False

def is_any(typ):
    if False:
        i = 10
        return i + 15
    return typ is typing.Any

def is_new_type(typ):
    if False:
        i = 10
        return i + 15
    return hasattr(typ, '__supertype__')
_ForwardRef = typing.ForwardRef

def is_forward_ref(typ):
    if False:
        return 10
    return isinstance(typ, _ForwardRef)
_type_var_cache = {}

def convert_builtin_to_typing(typ):
    if False:
        return 10
    'Convert recursively a given builtin to a typing object.\n\n  Args:\n    typ (`builtins`): builtin object that exist in _BUILTINS_TO_TYPING.\n\n  Returns:\n    type: The given builtins converted to a type.\n\n  '
    if getattr(typ, '__origin__', None) in _BUILTINS_TO_TYPING:
        args = map(convert_builtin_to_typing, typ.__args__)
        typ = _BUILTINS_TO_TYPING[typ.__origin__].copy_with(tuple(args))
    return typ

def convert_collections_to_typing(typ):
    if False:
        for i in range(10):
            print('nop')
    'Converts a given collections.abc type to a typing object.\n\n  Args:\n    typ: an object inheriting from a collections.abc object\n\n  Returns:\n    type: The corresponding typing object.\n  '
    if hasattr(typ, '__iter__'):
        if hasattr(typ, '__next__'):
            typ = typing.Iterator[typ.__args__]
        elif hasattr(typ, 'send') and hasattr(typ, 'throw'):
            typ = typing.Generator[typ.__args__]
        elif _match_is_exactly_iterable(typ):
            typ = typing.Iterable[typ.__args__]
    return typ

def convert_to_beam_type(typ):
    if False:
        while True:
            i = 10
    'Convert a given typing type to a Beam type.\n\n  Args:\n    typ (`typing.Union[type, str]`): typing type or string literal representing\n      a type.\n\n  Returns:\n    type: The given type converted to a Beam type as far as we can do the\n    conversion.\n\n  Raises:\n    ValueError: The type was malformed.\n  '
    if (sys.version_info.major == 3 and sys.version_info.minor >= 10) and isinstance(typ, types.UnionType):
        typ = typing.Union[typ]
    if sys.version_info >= (3, 9) and isinstance(typ, types.GenericAlias):
        typ = convert_builtin_to_typing(typ)
    if sys.version_info >= (3, 9) and getattr(typ, '__module__', None) == 'collections.abc':
        typ = convert_collections_to_typing(typ)
    typ_module = getattr(typ, '__module__', None)
    if isinstance(typ, typing.TypeVar):
        if id(typ) not in _type_var_cache:
            new_type_variable = typehints.TypeVariable(typ.__name__)
            _type_var_cache[id(typ)] = new_type_variable
            _type_var_cache[id(new_type_variable)] = typ
        return _type_var_cache[id(typ)]
    elif isinstance(typ, str):
        _LOGGER.info('Converting string literal type hint to Any: "%s"', typ)
        return typehints.Any
    elif sys.version_info >= (3, 10) and isinstance(typ, typing.NewType):
        _LOGGER.info('Converting NewType type hint to Any: "%s"', typ)
        return typehints.Any
    elif typ_module != 'typing' and typ_module != 'collections.abc':
        return typ
    if typ_module == 'collections.abc' and typ.__origin__ not in _CONVERTED_COLLECTIONS:
        return typ
    type_map = [_TypeMapEntry(match=is_new_type, arity=0, beam_type=typehints.Any), _TypeMapEntry(match=is_forward_ref, arity=0, beam_type=typehints.Any), _TypeMapEntry(match=is_any, arity=0, beam_type=typehints.Any), _TypeMapEntry(match=_match_issubclass(typing.Dict), arity=2, beam_type=typehints.Dict), _TypeMapEntry(match=_match_is_exactly_iterable, arity=1, beam_type=typehints.Iterable), _TypeMapEntry(match=_match_issubclass(typing.List), arity=1, beam_type=typehints.List), _TypeMapEntry(match=_match_issubclass(typing.FrozenSet), arity=1, beam_type=typehints.FrozenSet), _TypeMapEntry(match=match_is_set, arity=1, beam_type=typehints.Set), _TypeMapEntry(match=match_is_named_tuple, arity=0, beam_type=typehints.Any), _TypeMapEntry(match=_match_issubclass(typing.Tuple), arity=-1, beam_type=typehints.Tuple), _TypeMapEntry(match=_match_is_union, arity=-1, beam_type=typehints.Union), _TypeMapEntry(match=_match_issubclass(typing.Generator), arity=3, beam_type=typehints.Generator), _TypeMapEntry(match=_match_issubclass(typing.Iterator), arity=1, beam_type=typehints.Iterator), _TypeMapEntry(match=_match_is_exactly_collection, arity=1, beam_type=typehints.Collection)]
    matched_entry = next((entry for entry in type_map if entry.match(typ)), None)
    if not matched_entry:
        _LOGGER.info('Using Any for unsupported type: %s', typ)
        return typehints.Any
    args = _get_args(typ)
    len_args = len(args)
    if len_args == 0 and len_args != matched_entry.arity:
        arity = matched_entry.arity
        if _match_issubclass(typing.Tuple)(typ):
            args = (typehints.TypeVariable('T'), Ellipsis)
        elif _match_is_union(typ):
            raise ValueError('Unsupported Union with no arguments.')
        elif _match_issubclass(typing.Generator)(typ):
            args = (typehints.TypeVariable('T_co'), type(None), type(None))
        elif _match_issubclass(typing.Dict)(typ):
            args = (typehints.TypeVariable('KT'), typehints.TypeVariable('VT'))
        elif _match_issubclass(typing.Iterator)(typ) or _match_is_exactly_iterable(typ):
            args = (typehints.TypeVariable('T_co'),)
        else:
            args = (typehints.TypeVariable('T'),) * arity
    elif matched_entry.arity == -1:
        arity = len_args
    else:
        arity = matched_entry.arity
        if len_args != arity:
            raise ValueError('expecting type %s to have arity %d, had arity %d instead' % (str(typ), arity, len_args))
    typs = convert_to_beam_types(args)
    if arity == 0:
        return matched_entry.beam_type
    elif arity == 1:
        return matched_entry.beam_type[typs[0]]
    else:
        return matched_entry.beam_type[tuple(typs)]

def convert_to_beam_types(args):
    if False:
        print('Hello World!')
    'Convert the given list or dictionary of args to Beam types.\n\n  Args:\n    args: Either an iterable of types, or a dictionary where the values are\n    types.\n\n  Returns:\n    If given an iterable, a list of converted types. If given a dictionary,\n    a dictionary with the same keys, and values which have been converted.\n  '
    if isinstance(args, dict):
        return {k: convert_to_beam_type(v) for (k, v) in args.items()}
    else:
        return [convert_to_beam_type(v) for v in args]

def convert_to_typing_type(typ):
    if False:
        i = 10
        return i + 15
    'Converts a given Beam type to a typing type.\n\n  This is the reverse of convert_to_beam_type.\n\n  Args:\n    typ: If a typehints.TypeConstraint, the type to convert. Otherwise, typ\n      will be unchanged.\n\n  Returns:\n    Converted version of typ, or unchanged.\n\n  Raises:\n    ValueError: The type was malformed or could not be converted.\n  '
    if isinstance(typ, typehints.TypeVariable):
        if id(typ) not in _type_var_cache:
            new_type_variable = typing.TypeVar(typ.name)
            _type_var_cache[id(typ)] = new_type_variable
            _type_var_cache[id(new_type_variable)] = typ
        return _type_var_cache[id(typ)]
    elif not getattr(typ, '__module__', None).endswith('typehints'):
        return typ
    if isinstance(typ, typehints.AnyTypeConstraint):
        return typing.Any
    if isinstance(typ, typehints.DictConstraint):
        return typing.Dict[convert_to_typing_type(typ.key_type), convert_to_typing_type(typ.value_type)]
    if isinstance(typ, typehints.ListConstraint):
        return typing.List[convert_to_typing_type(typ.inner_type)]
    if isinstance(typ, typehints.IterableTypeConstraint):
        return typing.Iterable[convert_to_typing_type(typ.inner_type)]
    if isinstance(typ, typehints.UnionConstraint):
        return typing.Union[tuple(convert_to_typing_types(typ.union_types))]
    if isinstance(typ, typehints.SetTypeConstraint):
        return typing.Set[convert_to_typing_type(typ.inner_type)]
    if isinstance(typ, typehints.FrozenSetTypeConstraint):
        return typing.FrozenSet[convert_to_typing_type(typ.inner_type)]
    if isinstance(typ, typehints.TupleConstraint):
        return typing.Tuple[tuple(convert_to_typing_types(typ.tuple_types))]
    if isinstance(typ, typehints.TupleSequenceConstraint):
        return typing.Tuple[convert_to_typing_type(typ.inner_type), ...]
    if isinstance(typ, typehints.IteratorTypeConstraint):
        return typing.Iterator[convert_to_typing_type(typ.yielded_type)]
    raise ValueError('Failed to convert Beam type: %s' % typ)

def convert_to_typing_types(args):
    if False:
        print('Hello World!')
    'Convert the given list or dictionary of args to typing types.\n\n  Args:\n    args: Either an iterable of types, or a dictionary where the values are\n    types.\n\n  Returns:\n    If given an iterable, a list of converted types. If given a dictionary,\n    a dictionary with the same keys, and values which have been converted.\n  '
    if isinstance(args, dict):
        return {k: convert_to_typing_type(v) for (k, v) in args.items()}
    else:
        return [convert_to_typing_type(v) for v in args]