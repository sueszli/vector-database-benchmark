"""Logic for interacting with type annotations, mostly extensions, shims and hacks to wrap python's typing module."""
from __future__ import annotations as _annotations
import dataclasses
import sys
import types
import typing
from collections.abc import Callable
from functools import partial
from types import GetSetDescriptorType
from typing import TYPE_CHECKING, Any, Final, ForwardRef
from typing_extensions import Annotated, Literal, TypeAliasType, TypeGuard, get_args, get_origin
if TYPE_CHECKING:
    from ._dataclasses import StandardDataclass
try:
    from typing import _TypingBase
except ImportError:
    from typing import _Final as _TypingBase
typing_base = _TypingBase
if sys.version_info < (3, 9):
    TypingGenericAlias = ()
else:
    from typing import GenericAlias as TypingGenericAlias
if sys.version_info < (3, 11):
    from typing_extensions import NotRequired, Required
else:
    from typing import NotRequired, Required
if sys.version_info < (3, 10):

    def origin_is_union(tp: type[Any] | None) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return tp is typing.Union
    WithArgsTypes = (TypingGenericAlias,)
else:

    def origin_is_union(tp: type[Any] | None) -> bool:
        if False:
            return 10
        return tp is typing.Union or tp is types.UnionType
    WithArgsTypes = (typing._GenericAlias, types.GenericAlias, types.UnionType)
if sys.version_info < (3, 10):
    NoneType = type(None)
    EllipsisType = type(Ellipsis)
else:
    from types import NoneType as NoneType
LITERAL_TYPES: set[Any] = {Literal}
if hasattr(typing, 'Literal'):
    LITERAL_TYPES.add(typing.Literal)
NONE_TYPES: tuple[Any, ...] = (None, NoneType, *(tp[None] for tp in LITERAL_TYPES))
TypeVarType = Any

def is_none_type(type_: Any) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return type_ in NONE_TYPES

def is_callable_type(type_: type[Any]) -> bool:
    if False:
        print('Hello World!')
    return type_ is Callable or get_origin(type_) is Callable

def is_literal_type(type_: type[Any]) -> bool:
    if False:
        print('Hello World!')
    return Literal is not None and get_origin(type_) in LITERAL_TYPES

def literal_values(type_: type[Any]) -> tuple[Any, ...]:
    if False:
        for i in range(10):
            print('nop')
    return get_args(type_)

def all_literal_values(type_: type[Any]) -> list[Any]:
    if False:
        print('Hello World!')
    'This method is used to retrieve all Literal values as\n    Literal can be used recursively (see https://www.python.org/dev/peps/pep-0586)\n    e.g. `Literal[Literal[Literal[1, 2, 3], "foo"], 5, None]`.\n    '
    if not is_literal_type(type_):
        return [type_]
    values = literal_values(type_)
    return list((x for value in values for x in all_literal_values(value)))

def is_annotated(ann_type: Any) -> bool:
    if False:
        i = 10
        return i + 15
    from ._utils import lenient_issubclass
    origin = get_origin(ann_type)
    return origin is not None and lenient_issubclass(origin, Annotated)

def is_namedtuple(type_: type[Any]) -> bool:
    if False:
        return 10
    'Check if a given class is a named tuple.\n    It can be either a `typing.NamedTuple` or `collections.namedtuple`.\n    '
    from ._utils import lenient_issubclass
    return lenient_issubclass(type_, tuple) and hasattr(type_, '_fields')
test_new_type = typing.NewType('test_new_type', str)

def is_new_type(type_: type[Any]) -> bool:
    if False:
        return 10
    "Check whether type_ was created using typing.NewType.\n\n    Can't use isinstance because it fails <3.10.\n    "
    return isinstance(type_, test_new_type.__class__) and hasattr(type_, '__supertype__')

def _check_classvar(v: type[Any] | None) -> bool:
    if False:
        while True:
            i = 10
    if v is None:
        return False
    return v.__class__ == typing.ClassVar.__class__ and getattr(v, '_name', None) == 'ClassVar'

def is_classvar(ann_type: type[Any]) -> bool:
    if False:
        print('Hello World!')
    if _check_classvar(ann_type) or _check_classvar(get_origin(ann_type)):
        return True
    if ann_type.__class__ == typing.ForwardRef and ann_type.__forward_arg__.startswith('ClassVar['):
        return True
    return False

def _check_finalvar(v: type[Any] | None) -> bool:
    if False:
        while True:
            i = 10
    'Check if a given type is a `typing.Final` type.'
    if v is None:
        return False
    return v.__class__ == Final.__class__ and (sys.version_info < (3, 8) or getattr(v, '_name', None) == 'Final')

def is_finalvar(ann_type: Any) -> bool:
    if False:
        i = 10
        return i + 15
    return _check_finalvar(ann_type) or _check_finalvar(get_origin(ann_type))

def parent_frame_namespace(*, parent_depth: int=2) -> dict[str, Any] | None:
    if False:
        for i in range(10):
            print('nop')
    "We allow use of items in parent namespace to get around the issue with `get_type_hints` only looking in the\n    global module namespace. See https://github.com/pydantic/pydantic/issues/2678#issuecomment-1008139014 -> Scope\n    and suggestion at the end of the next comment by @gvanrossum.\n\n    WARNING 1: it matters exactly where this is called. By default, this function will build a namespace from the\n    parent of where it is called.\n\n    WARNING 2: this only looks in the parent namespace, not other parents since (AFAIK) there's no way to collect a\n    dict of exactly what's in scope. Using `f_back` would work sometimes but would be very wrong and confusing in many\n    other cases. See https://discuss.python.org/t/is-there-a-way-to-access-parent-nested-namespaces/20659.\n    "
    frame = sys._getframe(parent_depth)
    if frame.f_back is None:
        return None
    else:
        return frame.f_locals

def add_module_globals(obj: Any, globalns: dict[str, Any] | None=None) -> dict[str, Any]:
    if False:
        while True:
            i = 10
    module_name = getattr(obj, '__module__', None)
    if module_name:
        try:
            module_globalns = sys.modules[module_name].__dict__
        except KeyError:
            pass
        else:
            if globalns:
                return {**module_globalns, **globalns}
            else:
                return module_globalns.copy()
    return globalns or {}

def get_cls_types_namespace(cls: type[Any], parent_namespace: dict[str, Any] | None=None) -> dict[str, Any]:
    if False:
        return 10
    ns = add_module_globals(cls, parent_namespace)
    ns[cls.__name__] = cls
    return ns

def get_cls_type_hints_lenient(obj: Any, globalns: dict[str, Any] | None=None) -> dict[str, Any]:
    if False:
        while True:
            i = 10
    'Collect annotations from a class, including those from parent classes.\n\n    Unlike `typing.get_type_hints`, this function will not error if a forward reference is not resolvable.\n    '
    hints = {}
    for base in reversed(obj.__mro__):
        ann = base.__dict__.get('__annotations__')
        localns = dict(vars(base))
        if ann is not None and ann is not GetSetDescriptorType:
            for (name, value) in ann.items():
                hints[name] = eval_type_lenient(value, globalns, localns)
    return hints

def eval_type_lenient(value: Any, globalns: dict[str, Any] | None, localns: dict[str, Any] | None) -> Any:
    if False:
        i = 10
        return i + 15
    "Behaves like typing._eval_type, except it won't raise an error if a forward reference can't be resolved."
    if value is None:
        value = NoneType
    elif isinstance(value, str):
        value = _make_forward_ref(value, is_argument=False, is_class=True)
    try:
        return typing._eval_type(value, globalns, localns)
    except NameError:
        return value

def get_function_type_hints(function: Callable[..., Any], *, include_keys: set[str] | None=None, types_namespace: dict[str, Any] | None=None) -> dict[str, Any]:
    if False:
        return 10
    "Like `typing.get_type_hints`, but doesn't convert `X` to `Optional[X]` if the default value is `None`, also\n    copes with `partial`.\n    "
    if isinstance(function, partial):
        annotations = function.func.__annotations__
    else:
        annotations = function.__annotations__
    globalns = add_module_globals(function)
    type_hints = {}
    for (name, value) in annotations.items():
        if include_keys is not None and name not in include_keys:
            continue
        if value is None:
            value = NoneType
        elif isinstance(value, str):
            value = _make_forward_ref(value)
        type_hints[name] = typing._eval_type(value, globalns, types_namespace)
    return type_hints
if sys.version_info < (3, 9, 8) or (3, 10) <= sys.version_info < (3, 10, 1):

    def _make_forward_ref(arg: Any, is_argument: bool=True, *, is_class: bool=False) -> typing.ForwardRef:
        if False:
            for i in range(10):
                print('nop')
        "Wrapper for ForwardRef that accounts for the `is_class` argument missing in older versions.\n        The `module` argument is omitted as it breaks <3.9.8, =3.10.0 and isn't used in the calls below.\n\n        See https://github.com/python/cpython/pull/28560 for some background.\n        The backport happened on 3.9.8, see:\n        https://github.com/pydantic/pydantic/discussions/6244#discussioncomment-6275458,\n        and on 3.10.1 for the 3.10 branch, see:\n        https://github.com/pydantic/pydantic/issues/6912\n\n        Implemented as EAFP with memory.\n        "
        return typing.ForwardRef(arg, is_argument)
else:
    _make_forward_ref = typing.ForwardRef
if sys.version_info >= (3, 10):
    get_type_hints = typing.get_type_hints
else:
    '\n    For older versions of python, we have a custom implementation of `get_type_hints` which is a close as possible to\n    the implementation in CPython 3.10.8.\n    '

    @typing.no_type_check
    def get_type_hints(obj: Any, globalns: dict[str, Any] | None=None, localns: dict[str, Any] | None=None, include_extras: bool=False) -> dict[str, Any]:
        if False:
            return 10
        "Taken verbatim from python 3.10.8 unchanged, except:\n        * type annotations of the function definition above.\n        * prefixing `typing.` where appropriate\n        * Use `_make_forward_ref` instead of `typing.ForwardRef` to handle the `is_class` argument.\n\n        https://github.com/python/cpython/blob/aaaf5174241496afca7ce4d4584570190ff972fe/Lib/typing.py#L1773-L1875\n\n        DO NOT CHANGE THIS METHOD UNLESS ABSOLUTELY NECESSARY.\n        ======================================================\n\n        Return type hints for an object.\n\n        This is often the same as obj.__annotations__, but it handles\n        forward references encoded as string literals, adds Optional[t] if a\n        default value equal to None is set and recursively replaces all\n        'Annotated[T, ...]' with 'T' (unless 'include_extras=True').\n\n        The argument may be a module, class, method, or function. The annotations\n        are returned as a dictionary. For classes, annotations include also\n        inherited members.\n\n        TypeError is raised if the argument is not of a type that can contain\n        annotations, and an empty dictionary is returned if no annotations are\n        present.\n\n        BEWARE -- the behavior of globalns and localns is counterintuitive\n        (unless you are familiar with how eval() and exec() work).  The\n        search order is locals first, then globals.\n\n        - If no dict arguments are passed, an attempt is made to use the\n          globals from obj (or the respective module's globals for classes),\n          and these are also used as the locals.  If the object does not appear\n          to have globals, an empty dictionary is used.  For classes, the search\n          order is globals first then locals.\n\n        - If one dict argument is passed, it is used for both globals and\n          locals.\n\n        - If two dict arguments are passed, they specify globals and\n          locals, respectively.\n        "
        if getattr(obj, '__no_type_check__', None):
            return {}
        if isinstance(obj, type):
            hints = {}
            for base in reversed(obj.__mro__):
                if globalns is None:
                    base_globals = getattr(sys.modules.get(base.__module__, None), '__dict__', {})
                else:
                    base_globals = globalns
                ann = base.__dict__.get('__annotations__', {})
                if isinstance(ann, types.GetSetDescriptorType):
                    ann = {}
                base_locals = dict(vars(base)) if localns is None else localns
                if localns is None and globalns is None:
                    (base_globals, base_locals) = (base_locals, base_globals)
                for (name, value) in ann.items():
                    if value is None:
                        value = type(None)
                    if isinstance(value, str):
                        value = _make_forward_ref(value, is_argument=False, is_class=True)
                    value = typing._eval_type(value, base_globals, base_locals)
                    hints[name] = value
            return hints if include_extras else {k: typing._strip_annotations(t) for (k, t) in hints.items()}
        if globalns is None:
            if isinstance(obj, types.ModuleType):
                globalns = obj.__dict__
            else:
                nsobj = obj
                while hasattr(nsobj, '__wrapped__'):
                    nsobj = nsobj.__wrapped__
                globalns = getattr(nsobj, '__globals__', {})
            if localns is None:
                localns = globalns
        elif localns is None:
            localns = globalns
        hints = getattr(obj, '__annotations__', None)
        if hints is None:
            if isinstance(obj, typing._allowed_types):
                return {}
            else:
                raise TypeError(f'{obj!r} is not a module, class, method, or function.')
        defaults = typing._get_defaults(obj)
        hints = dict(hints)
        for (name, value) in hints.items():
            if value is None:
                value = type(None)
            if isinstance(value, str):
                value = _make_forward_ref(value, is_argument=not isinstance(obj, types.ModuleType), is_class=False)
            value = typing._eval_type(value, globalns, localns)
            if name in defaults and defaults[name] is None:
                value = typing.Optional[value]
            hints[name] = value
        return hints if include_extras else {k: typing._strip_annotations(t) for (k, t) in hints.items()}
if sys.version_info < (3, 9):

    def evaluate_fwd_ref(ref: ForwardRef, globalns: dict[str, Any] | None=None, localns: dict[str, Any] | None=None) -> Any:
        if False:
            i = 10
            return i + 15
        return ref._evaluate(globalns=globalns, localns=localns)
else:

    def evaluate_fwd_ref(ref: ForwardRef, globalns: dict[str, Any] | None=None, localns: dict[str, Any] | None=None) -> Any:
        if False:
            print('Hello World!')
        return ref._evaluate(globalns=globalns, localns=localns, recursive_guard=frozenset())

def is_dataclass(_cls: type[Any]) -> TypeGuard[type[StandardDataclass]]:
    if False:
        while True:
            i = 10
    return dataclasses.is_dataclass(_cls)

def origin_is_type_alias_type(origin: Any) -> TypeGuard[TypeAliasType]:
    if False:
        i = 10
        return i + 15
    return isinstance(origin, TypeAliasType)
if sys.version_info >= (3, 10):

    def is_generic_alias(type_: type[Any]) -> bool:
        if False:
            i = 10
            return i + 15
        return isinstance(type_, (types.GenericAlias, typing._GenericAlias))
else:

    def is_generic_alias(type_: type[Any]) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return isinstance(type_, typing._GenericAlias)