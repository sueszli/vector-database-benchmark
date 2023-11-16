"""
These are keyword-only APIs that call `attr.s` and `attr.ib` with different
default values.
"""
from functools import partial
from . import setters
from ._funcs import asdict as _asdict
from ._funcs import astuple as _astuple
from ._make import NOTHING, _frozen_setattrs, _ng_default_on_setattr, attrib, attrs
from .exceptions import UnannotatedAttributeError

def define(maybe_cls=None, *, these=None, repr=None, unsafe_hash=None, hash=None, init=None, slots=True, frozen=False, weakref_slot=True, str=False, auto_attribs=None, kw_only=False, cache_hash=False, auto_exc=True, eq=None, order=False, auto_detect=True, getstate_setstate=None, on_setattr=None, field_transformer=None, match_args=True):
    if False:
        i = 10
        return i + 15
    '\n    Define an *attrs* class.\n\n    Differences to the classic `attr.s` that it uses underneath:\n\n    - Automatically detect whether or not *auto_attribs* should be `True` (c.f.\n      *auto_attribs* parameter).\n    - Converters and validators run when attributes are set by default -- if\n      *frozen* is `False`.\n    - *slots=True*\n\n      .. caution::\n\n         Usually this has only upsides and few visible effects in everyday\n         programming. But it *can* lead to some surprising behaviors, so please\n         make sure to read :term:`slotted classes`.\n    - *auto_exc=True*\n    - *auto_detect=True*\n    - *order=False*\n    - Some options that were only relevant on Python 2 or were kept around for\n      backwards-compatibility have been removed.\n\n    Please note that these are all defaults and you can change them as you\n    wish.\n\n    :param Optional[bool] auto_attribs: If set to `True` or `False`, it behaves\n       exactly like `attr.s`. If left `None`, `attr.s` will try to guess:\n\n       1. If any attributes are annotated and no unannotated `attrs.fields`\\ s\n          are found, it assumes *auto_attribs=True*.\n       2. Otherwise it assumes *auto_attribs=False* and tries to collect\n          `attrs.fields`\\ s.\n\n    For now, please refer to `attr.s` for the rest of the parameters.\n\n    .. versionadded:: 20.1.0\n    .. versionchanged:: 21.3.0 Converters are also run ``on_setattr``.\n    .. versionadded:: 22.2.0\n       *unsafe_hash* as an alias for *hash* (for :pep:`681` compliance).\n    '

    def do_it(cls, auto_attribs):
        if False:
            print('Hello World!')
        return attrs(maybe_cls=cls, these=these, repr=repr, hash=hash, unsafe_hash=unsafe_hash, init=init, slots=slots, frozen=frozen, weakref_slot=weakref_slot, str=str, auto_attribs=auto_attribs, kw_only=kw_only, cache_hash=cache_hash, auto_exc=auto_exc, eq=eq, order=order, auto_detect=auto_detect, collect_by_mro=True, getstate_setstate=getstate_setstate, on_setattr=on_setattr, field_transformer=field_transformer, match_args=match_args)

    def wrap(cls):
        if False:
            i = 10
            return i + 15
        '\n        Making this a wrapper ensures this code runs during class creation.\n\n        We also ensure that frozen-ness of classes is inherited.\n        '
        nonlocal frozen, on_setattr
        had_on_setattr = on_setattr not in (None, setters.NO_OP)
        if frozen is False and on_setattr is None:
            on_setattr = _ng_default_on_setattr
        for base_cls in cls.__bases__:
            if base_cls.__setattr__ is _frozen_setattrs:
                if had_on_setattr:
                    msg = "Frozen classes can't use on_setattr (frozen-ness was inherited)."
                    raise ValueError(msg)
                on_setattr = setters.NO_OP
                break
        if auto_attribs is not None:
            return do_it(cls, auto_attribs)
        try:
            return do_it(cls, True)
        except UnannotatedAttributeError:
            return do_it(cls, False)
    if maybe_cls is None:
        return wrap
    return wrap(maybe_cls)
mutable = define
frozen = partial(define, frozen=True, on_setattr=None)

def field(*, default=NOTHING, validator=None, repr=True, hash=None, init=True, metadata=None, type=None, converter=None, factory=None, kw_only=False, eq=None, order=None, on_setattr=None, alias=None):
    if False:
        while True:
            i = 10
    '\n    Identical to `attr.ib`, except keyword-only and with some arguments\n    removed.\n\n    .. versionadded:: 23.1.0\n       The *type* parameter has been re-added; mostly for `attrs.make_class`.\n       Please note that type checkers ignore this metadata.\n    .. versionadded:: 20.1.0\n    '
    return attrib(default=default, validator=validator, repr=repr, hash=hash, init=init, metadata=metadata, type=type, converter=converter, factory=factory, kw_only=kw_only, eq=eq, order=order, on_setattr=on_setattr, alias=alias)

def asdict(inst, *, recurse=True, filter=None, value_serializer=None):
    if False:
        return 10
    '\n    Same as `attr.asdict`, except that collections types are always retained\n    and dict is always used as *dict_factory*.\n\n    .. versionadded:: 21.3.0\n    '
    return _asdict(inst=inst, recurse=recurse, filter=filter, value_serializer=value_serializer, retain_collection_types=True)

def astuple(inst, *, recurse=True, filter=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Same as `attr.astuple`, except that collections types are always retained\n    and `tuple` is always used as the *tuple_factory*.\n\n    .. versionadded:: 21.3.0\n    '
    return _astuple(inst=inst, recurse=recurse, filter=filter, retain_collection_types=True)