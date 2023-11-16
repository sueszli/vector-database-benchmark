import contextlib
import copy
import enum
import inspect
import linecache
import sys
import types
import typing
from operator import itemgetter
from . import _compat, _config, setters
from ._compat import PY310, _AnnotationExtractor, get_generic_base
from .exceptions import DefaultAlreadySetError, FrozenInstanceError, NotAnAttrsClassError, UnannotatedAttributeError
_obj_setattr = object.__setattr__
_init_converter_pat = '__attr_converter_%s'
_init_factory_pat = '__attr_factory_%s'
_classvar_prefixes = ('typing.ClassVar', 't.ClassVar', 'ClassVar', 'typing_extensions.ClassVar')
_hash_cache_field = '_attrs_cached_hash'
_empty_metadata_singleton = types.MappingProxyType({})
_sentinel = object()
_ng_default_on_setattr = setters.pipe(setters.convert, setters.validate)

class _Nothing(enum.Enum):
    """
    Sentinel to indicate the lack of a value when ``None`` is ambiguous.

    If extending attrs, you can use ``typing.Literal[NOTHING]`` to show
    that a value may be ``NOTHING``.

    .. versionchanged:: 21.1.0 ``bool(NOTHING)`` is now False.
    .. versionchanged:: 22.2.0 ``NOTHING`` is now an ``enum.Enum`` variant.
    """
    NOTHING = enum.auto()

    def __repr__(self):
        if False:
            return 10
        return 'NOTHING'

    def __bool__(self):
        if False:
            print('Hello World!')
        return False
NOTHING = _Nothing.NOTHING
'\nSentinel to indicate the lack of a value when ``None`` is ambiguous.\n'

class _CacheHashWrapper(int):
    """
    An integer subclass that pickles / copies as None

    This is used for non-slots classes with ``cache_hash=True``, to avoid
    serializing a potentially (even likely) invalid hash value. Since ``None``
    is the default value for uncalculated hashes, whenever this is copied,
    the copy's value for the hash should automatically reset.

    See GH #613 for more details.
    """

    def __reduce__(self, _none_constructor=type(None), _args=()):
        if False:
            while True:
                i = 10
        return (_none_constructor, _args)

def attrib(default=NOTHING, validator=None, repr=True, cmp=None, hash=None, init=True, metadata=None, type=None, converter=None, factory=None, kw_only=False, eq=None, order=None, on_setattr=None, alias=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create a new attribute on a class.\n\n    ..  warning::\n\n        Does *not* do anything unless the class is also decorated with `attr.s`\n        / `attrs.define` / and so on!\n\n    Please consider using `attrs.field` in new code (``attr.ib`` will *never*\n    go away, though).\n\n    :param default: A value that is used if an *attrs*-generated ``__init__``\n        is used and no value is passed while instantiating or the attribute is\n        excluded using ``init=False``.\n\n        If the value is an instance of `attrs.Factory`, its callable will be\n        used to construct a new value (useful for mutable data types like lists\n        or dicts).\n\n        If a default is not set (or set manually to `attrs.NOTHING`), a value\n        *must* be supplied when instantiating; otherwise a `TypeError` will be\n        raised.\n\n        The default can also be set using decorator notation as shown below.\n\n        .. seealso:: `defaults`\n\n    :param callable factory: Syntactic sugar for\n        ``default=attr.Factory(factory)``.\n\n    :param validator: `callable` that is called by *attrs*-generated\n        ``__init__`` methods after the instance has been initialized.  They\n        receive the initialized instance, the :func:`~attrs.Attribute`, and the\n        passed value.\n\n        The return value is *not* inspected so the validator has to throw an\n        exception itself.\n\n        If a `list` is passed, its items are treated as validators and must all\n        pass.\n\n        Validators can be globally disabled and re-enabled using\n        `attrs.validators.get_disabled` / `attrs.validators.set_disabled`.\n\n        The validator can also be set using decorator notation as shown below.\n\n        .. seealso:: :ref:`validators`\n\n    :type validator: `callable` or a `list` of `callable`\\ s.\n\n    :param repr: Include this attribute in the generated ``__repr__`` method.\n        If ``True``, include the attribute; if ``False``, omit it. By default,\n        the built-in ``repr()`` function is used. To override how the attribute\n        value is formatted, pass a ``callable`` that takes a single value and\n        returns a string. Note that the resulting string is used as-is, i.e. it\n        will be used directly *instead* of calling ``repr()`` (the default).\n    :type repr: a `bool` or a `callable` to use a custom function.\n\n    :param eq: If ``True`` (default), include this attribute in the generated\n        ``__eq__`` and ``__ne__`` methods that check two instances for\n        equality. To override how the attribute value is compared, pass a\n        ``callable`` that takes a single value and returns the value to be\n        compared.\n\n        .. seealso:: `comparison`\n    :type eq: a `bool` or a `callable`.\n\n    :param order: If ``True`` (default), include this attributes in the\n        generated ``__lt__``, ``__le__``, ``__gt__`` and ``__ge__`` methods. To\n        override how the attribute value is ordered, pass a ``callable`` that\n        takes a single value and returns the value to be ordered.\n\n        .. seealso:: `comparison`\n    :type order: a `bool` or a `callable`.\n\n    :param cmp: Setting *cmp* is equivalent to setting *eq* and *order* to the\n        same value. Must not be mixed with *eq* or *order*.\n\n        .. seealso:: `comparison`\n    :type cmp: a `bool` or a `callable`.\n\n    :param bool | None hash: Include this attribute in the generated\n        ``__hash__`` method.  If ``None`` (default), mirror *eq*'s value.  This\n        is the correct behavior according the Python spec.  Setting this value\n        to anything else than ``None`` is *discouraged*.\n\n        .. seealso:: `hashing`\n    :param bool init: Include this attribute in the generated ``__init__``\n        method.  It is possible to set this to ``False`` and set a default\n        value.  In that case this attributed is unconditionally initialized\n        with the specified default value or factory.\n\n        .. seealso:: `init`\n    :param callable converter: `callable` that is called by *attrs*-generated\n        ``__init__`` methods to convert attribute's value to the desired\n        format.  It is given the passed-in value, and the returned value will\n        be used as the new value of the attribute.  The value is converted\n        before being passed to the validator, if any.\n\n        .. seealso:: :ref:`converters`\n    :param dict | None metadata: An arbitrary mapping, to be used by\n        third-party components.  See `extending-metadata`.\n\n    :param type: The type of the attribute. Nowadays, the preferred method to\n        specify the type is using a variable annotation (see :pep:`526`). This\n        argument is provided for backward compatibility. Regardless of the\n        approach used, the type will be stored on ``Attribute.type``.\n\n        Please note that *attrs* doesn't do anything with this metadata by\n        itself. You can use it as part of your own code or for `static type\n        checking <types>`.\n    :param bool kw_only: Make this attribute keyword-only in the generated\n        ``__init__`` (if ``init`` is ``False``, this parameter is ignored).\n    :param on_setattr: Allows to overwrite the *on_setattr* setting from\n        `attr.s`. If left `None`, the *on_setattr* value from `attr.s` is used.\n        Set to `attrs.setters.NO_OP` to run **no** `setattr` hooks for this\n        attribute -- regardless of the setting in `attr.s`.\n    :type on_setattr: `callable`, or a list of callables, or `None`, or\n        `attrs.setters.NO_OP`\n    :param str | None alias: Override this attribute's parameter name in the\n        generated ``__init__`` method. If left `None`, default to ``name``\n        stripped of leading underscores. See `private-attributes`.\n\n    .. versionadded:: 15.2.0 *convert*\n    .. versionadded:: 16.3.0 *metadata*\n    .. versionchanged:: 17.1.0 *validator* can be a ``list`` now.\n    .. versionchanged:: 17.1.0\n       *hash* is ``None`` and therefore mirrors *eq* by default.\n    .. versionadded:: 17.3.0 *type*\n    .. deprecated:: 17.4.0 *convert*\n    .. versionadded:: 17.4.0 *converter* as a replacement for the deprecated\n       *convert* to achieve consistency with other noun-based arguments.\n    .. versionadded:: 18.1.0\n       ``factory=f`` is syntactic sugar for ``default=attr.Factory(f)``.\n    .. versionadded:: 18.2.0 *kw_only*\n    .. versionchanged:: 19.2.0 *convert* keyword argument removed.\n    .. versionchanged:: 19.2.0 *repr* also accepts a custom callable.\n    .. deprecated:: 19.2.0 *cmp* Removal on or after 2021-06-01.\n    .. versionadded:: 19.2.0 *eq* and *order*\n    .. versionadded:: 20.1.0 *on_setattr*\n    .. versionchanged:: 20.3.0 *kw_only* backported to Python 2\n    .. versionchanged:: 21.1.0\n       *eq*, *order*, and *cmp* also accept a custom callable\n    .. versionchanged:: 21.1.0 *cmp* undeprecated\n    .. versionadded:: 22.2.0 *alias*\n    "
    (eq, eq_key, order, order_key) = _determine_attrib_eq_order(cmp, eq, order, True)
    if hash is not None and hash is not True and (hash is not False):
        msg = 'Invalid value for hash.  Must be True, False, or None.'
        raise TypeError(msg)
    if factory is not None:
        if default is not NOTHING:
            msg = 'The `default` and `factory` arguments are mutually exclusive.'
            raise ValueError(msg)
        if not callable(factory):
            msg = 'The `factory` argument must be a callable.'
            raise ValueError(msg)
        default = Factory(factory)
    if metadata is None:
        metadata = {}
    if isinstance(on_setattr, (list, tuple)):
        on_setattr = setters.pipe(*on_setattr)
    if validator and isinstance(validator, (list, tuple)):
        validator = and_(*validator)
    if converter and isinstance(converter, (list, tuple)):
        converter = pipe(*converter)
    return _CountingAttr(default=default, validator=validator, repr=repr, cmp=None, hash=hash, init=init, converter=converter, metadata=metadata, type=type, kw_only=kw_only, eq=eq, eq_key=eq_key, order=order, order_key=order_key, on_setattr=on_setattr, alias=alias)

def _compile_and_eval(script, globs, locs=None, filename=''):
    if False:
        for i in range(10):
            print('nop')
    '\n    "Exec" the script with the given global (globs) and local (locs) variables.\n    '
    bytecode = compile(script, filename, 'exec')
    eval(bytecode, globs, locs)

def _make_method(name, script, filename, globs):
    if False:
        print('Hello World!')
    '\n    Create the method with the script given and return the method object.\n    '
    locs = {}
    count = 1
    base_filename = filename
    while True:
        linecache_tuple = (len(script), None, script.splitlines(True), filename)
        old_val = linecache.cache.setdefault(filename, linecache_tuple)
        if old_val == linecache_tuple:
            break
        filename = f'{base_filename[:-1]}-{count}>'
        count += 1
    _compile_and_eval(script, globs, locs, filename)
    return locs[name]

def _make_attr_tuple_class(cls_name, attr_names):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a tuple subclass to hold `Attribute`s for an `attrs` class.\n\n    The subclass is a bare tuple with properties for names.\n\n    class MyClassAttributes(tuple):\n        __slots__ = ()\n        x = property(itemgetter(0))\n    '
    attr_class_name = f'{cls_name}Attributes'
    attr_class_template = [f'class {attr_class_name}(tuple):', '    __slots__ = ()']
    if attr_names:
        for (i, attr_name) in enumerate(attr_names):
            attr_class_template.append(f'    {attr_name} = _attrs_property(_attrs_itemgetter({i}))')
    else:
        attr_class_template.append('    pass')
    globs = {'_attrs_itemgetter': itemgetter, '_attrs_property': property}
    _compile_and_eval('\n'.join(attr_class_template), globs)
    return globs[attr_class_name]
_Attributes = _make_attr_tuple_class('_Attributes', ['attrs', 'base_attrs', 'base_attrs_map'])

def _is_class_var(annot):
    if False:
        print('Hello World!')
    '\n    Check whether *annot* is a typing.ClassVar.\n\n    The string comparison hack is used to avoid evaluating all string\n    annotations which would put attrs-based classes at a performance\n    disadvantage compared to plain old classes.\n    '
    annot = str(annot)
    if annot.startswith(("'", '"')) and annot.endswith(("'", '"')):
        annot = annot[1:-1]
    return annot.startswith(_classvar_prefixes)

def _has_own_attribute(cls, attrib_name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Check whether *cls* defines *attrib_name* (and doesn't just inherit it).\n    "
    attr = getattr(cls, attrib_name, _sentinel)
    if attr is _sentinel:
        return False
    for base_cls in cls.__mro__[1:]:
        a = getattr(base_cls, attrib_name, None)
        if attr is a:
            return False
    return True

def _get_annotations(cls):
    if False:
        return 10
    '\n    Get annotations for *cls*.\n    '
    if _has_own_attribute(cls, '__annotations__'):
        return cls.__annotations__
    return {}

def _collect_base_attrs(cls, taken_attr_names):
    if False:
        i = 10
        return i + 15
    '\n    Collect attr.ibs from base classes of *cls*, except *taken_attr_names*.\n    '
    base_attrs = []
    base_attr_map = {}
    for base_cls in reversed(cls.__mro__[1:-1]):
        for a in getattr(base_cls, '__attrs_attrs__', []):
            if a.inherited or a.name in taken_attr_names:
                continue
            a = a.evolve(inherited=True)
            base_attrs.append(a)
            base_attr_map[a.name] = base_cls
    filtered = []
    seen = set()
    for a in reversed(base_attrs):
        if a.name in seen:
            continue
        filtered.insert(0, a)
        seen.add(a.name)
    return (filtered, base_attr_map)

def _collect_base_attrs_broken(cls, taken_attr_names):
    if False:
        while True:
            i = 10
    '\n    Collect attr.ibs from base classes of *cls*, except *taken_attr_names*.\n\n    N.B. *taken_attr_names* will be mutated.\n\n    Adhere to the old incorrect behavior.\n\n    Notably it collects from the front and considers inherited attributes which\n    leads to the buggy behavior reported in #428.\n    '
    base_attrs = []
    base_attr_map = {}
    for base_cls in cls.__mro__[1:-1]:
        for a in getattr(base_cls, '__attrs_attrs__', []):
            if a.name in taken_attr_names:
                continue
            a = a.evolve(inherited=True)
            taken_attr_names.add(a.name)
            base_attrs.append(a)
            base_attr_map[a.name] = base_cls
    return (base_attrs, base_attr_map)

def _transform_attrs(cls, these, auto_attribs, kw_only, collect_by_mro, field_transformer):
    if False:
        for i in range(10):
            print('nop')
    "\n    Transform all `_CountingAttr`s on a class into `Attribute`s.\n\n    If *these* is passed, use that and don't look for them on the class.\n\n    *collect_by_mro* is True, collect them in the correct MRO order, otherwise\n    use the old -- incorrect -- order.  See #428.\n\n    Return an `_Attributes`.\n    "
    cd = cls.__dict__
    anns = _get_annotations(cls)
    if these is not None:
        ca_list = list(these.items())
    elif auto_attribs is True:
        ca_names = {name for (name, attr) in cd.items() if isinstance(attr, _CountingAttr)}
        ca_list = []
        annot_names = set()
        for (attr_name, type) in anns.items():
            if _is_class_var(type):
                continue
            annot_names.add(attr_name)
            a = cd.get(attr_name, NOTHING)
            if not isinstance(a, _CountingAttr):
                a = attrib() if a is NOTHING else attrib(default=a)
            ca_list.append((attr_name, a))
        unannotated = ca_names - annot_names
        if len(unannotated) > 0:
            raise UnannotatedAttributeError('The following `attr.ib`s lack a type annotation: ' + ', '.join(sorted(unannotated, key=lambda n: cd.get(n).counter)) + '.')
    else:
        ca_list = sorted(((name, attr) for (name, attr) in cd.items() if isinstance(attr, _CountingAttr)), key=lambda e: e[1].counter)
    own_attrs = [Attribute.from_counting_attr(name=attr_name, ca=ca, type=anns.get(attr_name)) for (attr_name, ca) in ca_list]
    if collect_by_mro:
        (base_attrs, base_attr_map) = _collect_base_attrs(cls, {a.name for a in own_attrs})
    else:
        (base_attrs, base_attr_map) = _collect_base_attrs_broken(cls, {a.name for a in own_attrs})
    if kw_only:
        own_attrs = [a.evolve(kw_only=True) for a in own_attrs]
        base_attrs = [a.evolve(kw_only=True) for a in base_attrs]
    attrs = base_attrs + own_attrs
    had_default = False
    for a in (a for a in attrs if a.init is not False and a.kw_only is False):
        if had_default is True and a.default is NOTHING:
            msg = f'No mandatory attributes allowed after an attribute with a default value or factory.  Attribute in question: {a!r}'
            raise ValueError(msg)
        if had_default is False and a.default is not NOTHING:
            had_default = True
    if field_transformer is not None:
        attrs = field_transformer(cls, attrs)
    attrs = [a.evolve(alias=_default_init_alias_for(a.name)) if not a.alias else a for a in attrs]
    attr_names = [a.name for a in attrs]
    AttrsClass = _make_attr_tuple_class(cls.__name__, attr_names)
    return _Attributes((AttrsClass(attrs), base_attrs, base_attr_map))

def _frozen_setattrs(self, name, value):
    if False:
        for i in range(10):
            print('nop')
    '\n    Attached to frozen classes as __setattr__.\n    '
    if isinstance(self, BaseException) and name in ('__cause__', '__context__', '__traceback__'):
        BaseException.__setattr__(self, name, value)
        return
    raise FrozenInstanceError()

def _frozen_delattrs(self, name):
    if False:
        i = 10
        return i + 15
    '\n    Attached to frozen classes as __delattr__.\n    '
    raise FrozenInstanceError()

class _ClassBuilder:
    """
    Iteratively build *one* class.
    """
    __slots__ = ('_attr_names', '_attrs', '_base_attr_map', '_base_names', '_cache_hash', '_cls', '_cls_dict', '_delete_attribs', '_frozen', '_has_pre_init', '_pre_init_has_args', '_has_post_init', '_is_exc', '_on_setattr', '_slots', '_weakref_slot', '_wrote_own_setattr', '_has_custom_setattr')

    def __init__(self, cls, these, slots, frozen, weakref_slot, getstate_setstate, auto_attribs, kw_only, cache_hash, is_exc, collect_by_mro, on_setattr, has_custom_setattr, field_transformer):
        if False:
            i = 10
            return i + 15
        (attrs, base_attrs, base_map) = _transform_attrs(cls, these, auto_attribs, kw_only, collect_by_mro, field_transformer)
        self._cls = cls
        self._cls_dict = dict(cls.__dict__) if slots else {}
        self._attrs = attrs
        self._base_names = {a.name for a in base_attrs}
        self._base_attr_map = base_map
        self._attr_names = tuple((a.name for a in attrs))
        self._slots = slots
        self._frozen = frozen
        self._weakref_slot = weakref_slot
        self._cache_hash = cache_hash
        self._has_pre_init = bool(getattr(cls, '__attrs_pre_init__', False))
        self._pre_init_has_args = False
        if self._has_pre_init:
            pre_init_func = cls.__attrs_pre_init__
            pre_init_signature = inspect.signature(pre_init_func)
            self._pre_init_has_args = len(pre_init_signature.parameters) > 1
        self._has_post_init = bool(getattr(cls, '__attrs_post_init__', False))
        self._delete_attribs = not bool(these)
        self._is_exc = is_exc
        self._on_setattr = on_setattr
        self._has_custom_setattr = has_custom_setattr
        self._wrote_own_setattr = False
        self._cls_dict['__attrs_attrs__'] = self._attrs
        if frozen:
            self._cls_dict['__setattr__'] = _frozen_setattrs
            self._cls_dict['__delattr__'] = _frozen_delattrs
            self._wrote_own_setattr = True
        elif on_setattr in (_ng_default_on_setattr, setters.validate, setters.convert):
            has_validator = has_converter = False
            for a in attrs:
                if a.validator is not None:
                    has_validator = True
                if a.converter is not None:
                    has_converter = True
                if has_validator and has_converter:
                    break
            if on_setattr == _ng_default_on_setattr and (not (has_validator or has_converter)) or (on_setattr == setters.validate and (not has_validator)) or (on_setattr == setters.convert and (not has_converter)):
                self._on_setattr = None
        if getstate_setstate:
            (self._cls_dict['__getstate__'], self._cls_dict['__setstate__']) = self._make_getstate_setstate()

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'<_ClassBuilder(cls={self._cls.__name__})>'
    if PY310:
        import abc

        def build_class(self):
            if False:
                i = 10
                return i + 15
            '\n            Finalize class based on the accumulated configuration.\n\n            Builder cannot be used after calling this method.\n            '
            if self._slots is True:
                return self._create_slots_class()
            return self.abc.update_abstractmethods(self._patch_original_class())
    else:

        def build_class(self):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Finalize class based on the accumulated configuration.\n\n            Builder cannot be used after calling this method.\n            '
            if self._slots is True:
                return self._create_slots_class()
            return self._patch_original_class()

    def _patch_original_class(self):
        if False:
            print('Hello World!')
        '\n        Apply accumulated methods and return the class.\n        '
        cls = self._cls
        base_names = self._base_names
        if self._delete_attribs:
            for name in self._attr_names:
                if name not in base_names and getattr(cls, name, _sentinel) is not _sentinel:
                    with contextlib.suppress(AttributeError):
                        delattr(cls, name)
        for (name, value) in self._cls_dict.items():
            setattr(cls, name, value)
        if not self._wrote_own_setattr and getattr(cls, '__attrs_own_setattr__', False):
            cls.__attrs_own_setattr__ = False
            if not self._has_custom_setattr:
                cls.__setattr__ = _obj_setattr
        return cls

    def _create_slots_class(self):
        if False:
            print('Hello World!')
        '\n        Build and return a new class with a `__slots__` attribute.\n        '
        cd = {k: v for (k, v) in self._cls_dict.items() if k not in (*tuple(self._attr_names), '__dict__', '__weakref__')}
        if not self._wrote_own_setattr:
            cd['__attrs_own_setattr__'] = False
            if not self._has_custom_setattr:
                for base_cls in self._cls.__bases__:
                    if base_cls.__dict__.get('__attrs_own_setattr__', False):
                        cd['__setattr__'] = _obj_setattr
                        break
        existing_slots = {}
        weakref_inherited = False
        for base_cls in self._cls.__mro__[1:-1]:
            if base_cls.__dict__.get('__weakref__', None) is not None:
                weakref_inherited = True
            existing_slots.update({name: getattr(base_cls, name) for name in getattr(base_cls, '__slots__', [])})
        base_names = set(self._base_names)
        names = self._attr_names
        if self._weakref_slot and '__weakref__' not in getattr(self._cls, '__slots__', ()) and ('__weakref__' not in names) and (not weakref_inherited):
            names += ('__weakref__',)
        slot_names = [name for name in names if name not in base_names]
        reused_slots = {slot: slot_descriptor for (slot, slot_descriptor) in existing_slots.items() if slot in slot_names}
        slot_names = [name for name in slot_names if name not in reused_slots]
        cd.update(reused_slots)
        if self._cache_hash:
            slot_names.append(_hash_cache_field)
        cd['__slots__'] = tuple(slot_names)
        cd['__qualname__'] = self._cls.__qualname__
        cls = type(self._cls)(self._cls.__name__, self._cls.__bases__, cd)
        for item in cls.__dict__.values():
            if isinstance(item, (classmethod, staticmethod)):
                closure_cells = getattr(item.__func__, '__closure__', None)
            elif isinstance(item, property):
                closure_cells = getattr(item.fget, '__closure__', None)
            else:
                closure_cells = getattr(item, '__closure__', None)
            if not closure_cells:
                continue
            for cell in closure_cells:
                try:
                    match = cell.cell_contents is self._cls
                except ValueError:
                    pass
                else:
                    if match:
                        cell.cell_contents = cls
        return cls

    def add_repr(self, ns):
        if False:
            while True:
                i = 10
        self._cls_dict['__repr__'] = self._add_method_dunders(_make_repr(self._attrs, ns, self._cls))
        return self

    def add_str(self):
        if False:
            print('Hello World!')
        repr = self._cls_dict.get('__repr__')
        if repr is None:
            msg = '__str__ can only be generated if a __repr__ exists.'
            raise ValueError(msg)

        def __str__(self):
            if False:
                i = 10
                return i + 15
            return self.__repr__()
        self._cls_dict['__str__'] = self._add_method_dunders(__str__)
        return self

    def _make_getstate_setstate(self):
        if False:
            i = 10
            return i + 15
        '\n        Create custom __setstate__ and __getstate__ methods.\n        '
        state_attr_names = tuple((an for an in self._attr_names if an != '__weakref__'))

        def slots_getstate(self):
            if False:
                print('Hello World!')
            '\n            Automatically created by attrs.\n            '
            return {name: getattr(self, name) for name in state_attr_names}
        hash_caching_enabled = self._cache_hash

        def slots_setstate(self, state):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Automatically created by attrs.\n            '
            __bound_setattr = _obj_setattr.__get__(self)
            if isinstance(state, tuple):
                for (name, value) in zip(state_attr_names, state):
                    __bound_setattr(name, value)
            else:
                for name in state_attr_names:
                    if name in state:
                        __bound_setattr(name, state[name])
            if hash_caching_enabled:
                __bound_setattr(_hash_cache_field, None)
        return (slots_getstate, slots_setstate)

    def make_unhashable(self):
        if False:
            for i in range(10):
                print('nop')
        self._cls_dict['__hash__'] = None
        return self

    def add_hash(self):
        if False:
            i = 10
            return i + 15
        self._cls_dict['__hash__'] = self._add_method_dunders(_make_hash(self._cls, self._attrs, frozen=self._frozen, cache_hash=self._cache_hash))
        return self

    def add_init(self):
        if False:
            print('Hello World!')
        self._cls_dict['__init__'] = self._add_method_dunders(_make_init(self._cls, self._attrs, self._has_pre_init, self._pre_init_has_args, self._has_post_init, self._frozen, self._slots, self._cache_hash, self._base_attr_map, self._is_exc, self._on_setattr, attrs_init=False))
        return self

    def add_match_args(self):
        if False:
            i = 10
            return i + 15
        self._cls_dict['__match_args__'] = tuple((field.name for field in self._attrs if field.init and (not field.kw_only)))

    def add_attrs_init(self):
        if False:
            i = 10
            return i + 15
        self._cls_dict['__attrs_init__'] = self._add_method_dunders(_make_init(self._cls, self._attrs, self._has_pre_init, self._pre_init_has_args, self._has_post_init, self._frozen, self._slots, self._cache_hash, self._base_attr_map, self._is_exc, self._on_setattr, attrs_init=True))
        return self

    def add_eq(self):
        if False:
            i = 10
            return i + 15
        cd = self._cls_dict
        cd['__eq__'] = self._add_method_dunders(_make_eq(self._cls, self._attrs))
        cd['__ne__'] = self._add_method_dunders(_make_ne())
        return self

    def add_order(self):
        if False:
            print('Hello World!')
        cd = self._cls_dict
        (cd['__lt__'], cd['__le__'], cd['__gt__'], cd['__ge__']) = (self._add_method_dunders(meth) for meth in _make_order(self._cls, self._attrs))
        return self

    def add_setattr(self):
        if False:
            print('Hello World!')
        if self._frozen:
            return self
        sa_attrs = {}
        for a in self._attrs:
            on_setattr = a.on_setattr or self._on_setattr
            if on_setattr and on_setattr is not setters.NO_OP:
                sa_attrs[a.name] = (a, on_setattr)
        if not sa_attrs:
            return self
        if self._has_custom_setattr:
            msg = "Can't combine custom __setattr__ with on_setattr hooks."
            raise ValueError(msg)

        def __setattr__(self, name, val):
            if False:
                return 10
            try:
                (a, hook) = sa_attrs[name]
            except KeyError:
                nval = val
            else:
                nval = hook(self, a, val)
            _obj_setattr(self, name, nval)
        self._cls_dict['__attrs_own_setattr__'] = True
        self._cls_dict['__setattr__'] = self._add_method_dunders(__setattr__)
        self._wrote_own_setattr = True
        return self

    def _add_method_dunders(self, method):
        if False:
            i = 10
            return i + 15
        '\n        Add __module__ and __qualname__ to a *method* if possible.\n        '
        with contextlib.suppress(AttributeError):
            method.__module__ = self._cls.__module__
        with contextlib.suppress(AttributeError):
            method.__qualname__ = f'{self._cls.__qualname__}.{method.__name__}'
        with contextlib.suppress(AttributeError):
            method.__doc__ = f'Method generated by attrs for class {self._cls.__qualname__}.'
        return method

def _determine_attrs_eq_order(cmp, eq, order, default_eq):
    if False:
        print('Hello World!')
    '\n    Validate the combination of *cmp*, *eq*, and *order*. Derive the effective\n    values of eq and order.  If *eq* is None, set it to *default_eq*.\n    '
    if cmp is not None and any((eq is not None, order is not None)):
        msg = "Don't mix `cmp` with `eq' and `order`."
        raise ValueError(msg)
    if cmp is not None:
        return (cmp, cmp)
    if eq is None:
        eq = default_eq
    if order is None:
        order = eq
    if eq is False and order is True:
        msg = '`order` can only be True if `eq` is True too.'
        raise ValueError(msg)
    return (eq, order)

def _determine_attrib_eq_order(cmp, eq, order, default_eq):
    if False:
        for i in range(10):
            print('nop')
    '\n    Validate the combination of *cmp*, *eq*, and *order*. Derive the effective\n    values of eq and order.  If *eq* is None, set it to *default_eq*.\n    '
    if cmp is not None and any((eq is not None, order is not None)):
        msg = "Don't mix `cmp` with `eq' and `order`."
        raise ValueError(msg)

    def decide_callable_or_boolean(value):
        if False:
            for i in range(10):
                print('nop')
        '\n        Decide whether a key function is used.\n        '
        if callable(value):
            (value, key) = (True, value)
        else:
            key = None
        return (value, key)
    if cmp is not None:
        (cmp, cmp_key) = decide_callable_or_boolean(cmp)
        return (cmp, cmp_key, cmp, cmp_key)
    if eq is None:
        (eq, eq_key) = (default_eq, None)
    else:
        (eq, eq_key) = decide_callable_or_boolean(eq)
    if order is None:
        (order, order_key) = (eq, eq_key)
    else:
        (order, order_key) = decide_callable_or_boolean(order)
    if eq is False and order is True:
        msg = '`order` can only be True if `eq` is True too.'
        raise ValueError(msg)
    return (eq, eq_key, order, order_key)

def _determine_whether_to_implement(cls, flag, auto_detect, dunders, default=True):
    if False:
        while True:
            i = 10
    "\n    Check whether we should implement a set of methods for *cls*.\n\n    *flag* is the argument passed into @attr.s like 'init', *auto_detect* the\n    same as passed into @attr.s and *dunders* is a tuple of attribute names\n    whose presence signal that the user has implemented it themselves.\n\n    Return *default* if no reason for either for or against is found.\n    "
    if flag is True or flag is False:
        return flag
    if flag is None and auto_detect is False:
        return default
    for dunder in dunders:
        if _has_own_attribute(cls, dunder):
            return False
    return default

def attrs(maybe_cls=None, these=None, repr_ns=None, repr=None, cmp=None, hash=None, init=None, slots=False, frozen=False, weakref_slot=True, str=False, auto_attribs=False, kw_only=False, cache_hash=False, auto_exc=False, eq=None, order=None, auto_detect=False, collect_by_mro=False, getstate_setstate=None, on_setattr=None, field_transformer=None, match_args=True, unsafe_hash=None):
    if False:
        i = 10
        return i + 15
    '\n    A class decorator that adds :term:`dunder methods` according to the\n    specified attributes using `attr.ib` or the *these* argument.\n\n    Please consider using `attrs.define` / `attrs.frozen` in new code\n    (``attr.s`` will *never* go away, though).\n\n    :param these: A dictionary of name to `attr.ib` mappings.  This is useful\n        to avoid the definition of your attributes within the class body\n        because you can\'t (e.g. if you want to add ``__repr__`` methods to\n        Django models) or don\'t want to.\n\n        If *these* is not ``None``, *attrs* will *not* search the class body\n        for attributes and will *not* remove any attributes from it.\n\n        The order is deduced from the order of the attributes inside *these*.\n\n    :type these: `dict` of `str` to `attr.ib`\n\n    :param str repr_ns: When using nested classes, there\'s no way in Python 2\n        to automatically detect that.  Therefore it\'s possible to set the\n        namespace explicitly for a more meaningful ``repr`` output.\n    :param bool auto_detect: Instead of setting the *init*, *repr*, *eq*,\n        *order*, and *hash* arguments explicitly, assume they are set to\n        ``True`` **unless any** of the involved methods for one of the\n        arguments is implemented in the *current* class (i.e. it is *not*\n        inherited from some base class).\n\n        So for example by implementing ``__eq__`` on a class yourself, *attrs*\n        will deduce ``eq=False`` and will create *neither* ``__eq__`` *nor*\n        ``__ne__`` (but Python classes come with a sensible ``__ne__`` by\n        default, so it *should* be enough to only implement ``__eq__`` in most\n        cases).\n\n        .. warning::\n\n           If you prevent *attrs* from creating the ordering methods for you\n           (``order=False``, e.g. by implementing ``__le__``), it becomes\n           *your* responsibility to make sure its ordering is sound. The best\n           way is to use the `functools.total_ordering` decorator.\n\n\n        Passing ``True`` or ``False`` to *init*, *repr*, *eq*, *order*, *cmp*,\n        or *hash* overrides whatever *auto_detect* would determine.\n\n    :param bool repr: Create a ``__repr__`` method with a human readable\n        representation of *attrs* attributes..\n    :param bool str: Create a ``__str__`` method that is identical to\n        ``__repr__``.  This is usually not necessary except for `Exception`\\ s.\n    :param bool | None eq: If ``True`` or ``None`` (default), add ``__eq__``\n        and ``__ne__`` methods that check two instances for equality.\n\n        They compare the instances as if they were tuples of their *attrs*\n        attributes if and only if the types of both classes are *identical*!\n\n        .. seealso:: `comparison`\n    :param bool | None order: If ``True``, add ``__lt__``, ``__le__``,\n        ``__gt__``, and ``__ge__`` methods that behave like *eq* above and\n        allow instances to be ordered. If ``None`` (default) mirror value of\n        *eq*.\n\n        .. seealso:: `comparison`\n    :param bool | None cmp: Setting *cmp* is equivalent to setting *eq* and\n        *order* to the same value. Must not be mixed with *eq* or *order*.\n\n        .. seealso:: `comparison`\n    :param bool | None unsafe_hash: If ``None`` (default), the ``__hash__``\n        method is generated according how *eq* and *frozen* are set.\n\n        1. If *both* are True, *attrs* will generate a ``__hash__`` for you.\n        2. If *eq* is True and *frozen* is False, ``__hash__`` will be set to\n           None, marking it unhashable (which it is).\n        3. If *eq* is False, ``__hash__`` will be left untouched meaning the\n           ``__hash__`` method of the base class will be used (if base class is\n           ``object``, this means it will fall back to id-based hashing.).\n\n        Although not recommended, you can decide for yourself and force *attrs*\n        to create one (e.g. if the class is immutable even though you didn\'t\n        freeze it programmatically) by passing ``True`` or not.  Both of these\n        cases are rather special and should be used carefully.\n\n        .. seealso::\n\n           - Our documentation on `hashing`,\n           - Python\'s documentation on `object.__hash__`,\n           - and the `GitHub issue that led to the default \\\n             behavior <https://github.com/python-attrs/attrs/issues/136>`_ for\n             more details.\n\n    :param bool | None hash: Alias for *unsafe_hash*. *unsafe_hash* takes\n        precedence.\n    :param bool init: Create a ``__init__`` method that initializes the *attrs*\n        attributes. Leading underscores are stripped for the argument name. If\n        a ``__attrs_pre_init__`` method exists on the class, it will be called\n        before the class is initialized. If a ``__attrs_post_init__`` method\n        exists on the class, it will be called after the class is fully\n        initialized.\n\n        If ``init`` is ``False``, an ``__attrs_init__`` method will be injected\n        instead. This allows you to define a custom ``__init__`` method that\n        can do pre-init work such as ``super().__init__()``, and then call\n        ``__attrs_init__()`` and ``__attrs_post_init__()``.\n\n        .. seealso:: `init`\n    :param bool slots: Create a :term:`slotted class <slotted classes>` that\'s\n        more memory-efficient. Slotted classes are generally superior to the\n        default dict classes, but have some gotchas you should know about, so\n        we encourage you to read the :term:`glossary entry <slotted classes>`.\n    :param bool frozen: Make instances immutable after initialization.  If\n        someone attempts to modify a frozen instance,\n        `attrs.exceptions.FrozenInstanceError` is raised.\n\n        .. note::\n\n            1. This is achieved by installing a custom ``__setattr__`` method\n               on your class, so you can\'t implement your own.\n\n            2. True immutability is impossible in Python.\n\n            3. This *does* have a minor a runtime performance `impact\n               <how-frozen>` when initializing new instances.  In other words:\n               ``__init__`` is slightly slower with ``frozen=True``.\n\n            4. If a class is frozen, you cannot modify ``self`` in\n               ``__attrs_post_init__`` or a self-written ``__init__``. You can\n               circumvent that limitation by using ``object.__setattr__(self,\n               "attribute_name", value)``.\n\n            5. Subclasses of a frozen class are frozen too.\n\n    :param bool weakref_slot: Make instances weak-referenceable.  This has no\n        effect unless ``slots`` is also enabled.\n    :param bool auto_attribs: If ``True``, collect :pep:`526`-annotated\n        attributes from the class body.\n\n        In this case, you **must** annotate every field.  If *attrs* encounters\n        a field that is set to an `attr.ib` but lacks a type annotation, an\n        `attr.exceptions.UnannotatedAttributeError` is raised.  Use\n        ``field_name: typing.Any = attr.ib(...)`` if you don\'t want to set a\n        type.\n\n        If you assign a value to those attributes (e.g. ``x: int = 42``), that\n        value becomes the default value like if it were passed using\n        ``attr.ib(default=42)``.  Passing an instance of `attrs.Factory` also\n        works as expected in most cases (see warning below).\n\n        Attributes annotated as `typing.ClassVar`, and attributes that are\n        neither annotated nor set to an `attr.ib` are **ignored**.\n\n        .. warning::\n           For features that use the attribute name to create decorators (e.g.\n           :ref:`validators <validators>`), you still *must* assign `attr.ib`\n           to them. Otherwise Python will either not find the name or try to\n           use the default value to call e.g. ``validator`` on it.\n\n           These errors can be quite confusing and probably the most common bug\n           report on our bug tracker.\n\n    :param bool kw_only: Make all attributes keyword-only in the generated\n        ``__init__`` (if ``init`` is ``False``, this parameter is ignored).\n    :param bool cache_hash: Ensure that the object\'s hash code is computed only\n        once and stored on the object.  If this is set to ``True``, hashing\n        must be either explicitly or implicitly enabled for this class.  If the\n        hash code is cached, avoid any reassignments of fields involved in hash\n        code computation or mutations of the objects those fields point to\n        after object creation.  If such changes occur, the behavior of the\n        object\'s hash code is undefined.\n    :param bool auto_exc: If the class subclasses `BaseException` (which\n        implicitly includes any subclass of any exception), the following\n        happens to behave like a well-behaved Python exceptions class:\n\n        - the values for *eq*, *order*, and *hash* are ignored and the\n          instances compare and hash by the instance\'s ids (N.B. *attrs* will\n          *not* remove existing implementations of ``__hash__`` or the equality\n          methods. It just won\'t add own ones.),\n        - all attributes that are either passed into ``__init__`` or have a\n          default value are additionally available as a tuple in the ``args``\n          attribute,\n        - the value of *str* is ignored leaving ``__str__`` to base classes.\n    :param bool collect_by_mro: Setting this to `True` fixes the way *attrs*\n       collects attributes from base classes.  The default behavior is\n       incorrect in certain cases of multiple inheritance.  It should be on by\n       default but is kept off for backward-compatibility.\n\n       .. seealso::\n          Issue `#428 <https://github.com/python-attrs/attrs/issues/428>`_\n\n    :param bool | None getstate_setstate:\n       .. note::\n          This is usually only interesting for slotted classes and you should\n          probably just set *auto_detect* to `True`.\n\n       If `True`, ``__getstate__`` and ``__setstate__`` are generated and\n       attached to the class. This is necessary for slotted classes to be\n       pickleable. If left `None`, it\'s `True` by default for slotted classes\n       and ``False`` for dict classes.\n\n       If *auto_detect* is `True`, and *getstate_setstate* is left `None`, and\n       **either** ``__getstate__`` or ``__setstate__`` is detected directly on\n       the class (i.e. not inherited), it is set to `False` (this is usually\n       what you want).\n\n    :param on_setattr: A callable that is run whenever the user attempts to set\n        an attribute (either by assignment like ``i.x = 42`` or by using\n        `setattr` like ``setattr(i, "x", 42)``). It receives the same arguments\n        as validators: the instance, the attribute that is being modified, and\n        the new value.\n\n        If no exception is raised, the attribute is set to the return value of\n        the callable.\n\n        If a list of callables is passed, they\'re automatically wrapped in an\n        `attrs.setters.pipe`.\n    :type on_setattr: `callable`, or a list of callables, or `None`, or\n        `attrs.setters.NO_OP`\n\n    :param callable | None field_transformer:\n        A function that is called with the original class object and all fields\n        right before *attrs* finalizes the class.  You can use this, e.g., to\n        automatically add converters or validators to fields based on their\n        types.\n\n        .. seealso:: `transform-fields`\n\n    :param bool match_args:\n        If `True` (default), set ``__match_args__`` on the class to support\n        :pep:`634` (Structural Pattern Matching). It is a tuple of all\n        non-keyword-only ``__init__`` parameter names on Python 3.10 and later.\n        Ignored on older Python versions.\n\n    .. versionadded:: 16.0.0 *slots*\n    .. versionadded:: 16.1.0 *frozen*\n    .. versionadded:: 16.3.0 *str*\n    .. versionadded:: 16.3.0 Support for ``__attrs_post_init__``.\n    .. versionchanged:: 17.1.0\n       *hash* supports ``None`` as value which is also the default now.\n    .. versionadded:: 17.3.0 *auto_attribs*\n    .. versionchanged:: 18.1.0\n       If *these* is passed, no attributes are deleted from the class body.\n    .. versionchanged:: 18.1.0 If *these* is ordered, the order is retained.\n    .. versionadded:: 18.2.0 *weakref_slot*\n    .. deprecated:: 18.2.0\n       ``__lt__``, ``__le__``, ``__gt__``, and ``__ge__`` now raise a\n       `DeprecationWarning` if the classes compared are subclasses of\n       each other. ``__eq`` and ``__ne__`` never tried to compared subclasses\n       to each other.\n    .. versionchanged:: 19.2.0\n       ``__lt__``, ``__le__``, ``__gt__``, and ``__ge__`` now do not consider\n       subclasses comparable anymore.\n    .. versionadded:: 18.2.0 *kw_only*\n    .. versionadded:: 18.2.0 *cache_hash*\n    .. versionadded:: 19.1.0 *auto_exc*\n    .. deprecated:: 19.2.0 *cmp* Removal on or after 2021-06-01.\n    .. versionadded:: 19.2.0 *eq* and *order*\n    .. versionadded:: 20.1.0 *auto_detect*\n    .. versionadded:: 20.1.0 *collect_by_mro*\n    .. versionadded:: 20.1.0 *getstate_setstate*\n    .. versionadded:: 20.1.0 *on_setattr*\n    .. versionadded:: 20.3.0 *field_transformer*\n    .. versionchanged:: 21.1.0\n       ``init=False`` injects ``__attrs_init__``\n    .. versionchanged:: 21.1.0 Support for ``__attrs_pre_init__``\n    .. versionchanged:: 21.1.0 *cmp* undeprecated\n    .. versionadded:: 21.3.0 *match_args*\n    .. versionadded:: 22.2.0\n       *unsafe_hash* as an alias for *hash* (for :pep:`681` compliance).\n    '
    (eq_, order_) = _determine_attrs_eq_order(cmp, eq, order, None)
    if unsafe_hash is not None:
        hash = unsafe_hash
    if isinstance(on_setattr, (list, tuple)):
        on_setattr = setters.pipe(*on_setattr)

    def wrap(cls):
        if False:
            for i in range(10):
                print('nop')
        is_frozen = frozen or _has_frozen_base_class(cls)
        is_exc = auto_exc is True and issubclass(cls, BaseException)
        has_own_setattr = auto_detect and _has_own_attribute(cls, '__setattr__')
        if has_own_setattr and is_frozen:
            msg = "Can't freeze a class with a custom __setattr__."
            raise ValueError(msg)
        builder = _ClassBuilder(cls, these, slots, is_frozen, weakref_slot, _determine_whether_to_implement(cls, getstate_setstate, auto_detect, ('__getstate__', '__setstate__'), default=slots), auto_attribs, kw_only, cache_hash, is_exc, collect_by_mro, on_setattr, has_own_setattr, field_transformer)
        if _determine_whether_to_implement(cls, repr, auto_detect, ('__repr__',)):
            builder.add_repr(repr_ns)
        if str is True:
            builder.add_str()
        eq = _determine_whether_to_implement(cls, eq_, auto_detect, ('__eq__', '__ne__'))
        if not is_exc and eq is True:
            builder.add_eq()
        if not is_exc and _determine_whether_to_implement(cls, order_, auto_detect, ('__lt__', '__le__', '__gt__', '__ge__')):
            builder.add_order()
        builder.add_setattr()
        nonlocal hash
        if hash is None and auto_detect is True and _has_own_attribute(cls, '__hash__'):
            hash = False
        if hash is not True and hash is not False and (hash is not None):
            msg = 'Invalid value for hash.  Must be True, False, or None.'
            raise TypeError(msg)
        if hash is False or (hash is None and eq is False) or is_exc:
            if cache_hash:
                msg = 'Invalid value for cache_hash.  To use hash caching, hashing must be either explicitly or implicitly enabled.'
                raise TypeError(msg)
        elif hash is True or (hash is None and eq is True and (is_frozen is True)):
            builder.add_hash()
        else:
            if cache_hash:
                msg = 'Invalid value for cache_hash.  To use hash caching, hashing must be either explicitly or implicitly enabled.'
                raise TypeError(msg)
            builder.make_unhashable()
        if _determine_whether_to_implement(cls, init, auto_detect, ('__init__',)):
            builder.add_init()
        else:
            builder.add_attrs_init()
            if cache_hash:
                msg = 'Invalid value for cache_hash.  To use hash caching, init must be True.'
                raise TypeError(msg)
        if PY310 and match_args and (not _has_own_attribute(cls, '__match_args__')):
            builder.add_match_args()
        return builder.build_class()
    if maybe_cls is None:
        return wrap
    return wrap(maybe_cls)
_attrs = attrs
'\nInternal alias so we can use it in functions that take an argument called\n*attrs*.\n'

def _has_frozen_base_class(cls):
    if False:
        return 10
    '\n    Check whether *cls* has a frozen ancestor by looking at its\n    __setattr__.\n    '
    return cls.__setattr__ is _frozen_setattrs

def _generate_unique_filename(cls, func_name):
    if False:
        while True:
            i = 10
    '\n    Create a "filename" suitable for a function being generated.\n    '
    return f"<attrs generated {func_name} {cls.__module__}.{getattr(cls, '__qualname__', cls.__name__)}>"

def _make_hash(cls, attrs, frozen, cache_hash):
    if False:
        i = 10
        return i + 15
    attrs = tuple((a for a in attrs if a.hash is True or (a.hash is None and a.eq is True)))
    tab = '        '
    unique_filename = _generate_unique_filename(cls, 'hash')
    type_hash = hash(unique_filename)
    globs = {}
    hash_def = 'def __hash__(self'
    hash_func = 'hash(('
    closing_braces = '))'
    if not cache_hash:
        hash_def += '):'
    else:
        hash_def += ', *'
        hash_def += ", _cache_wrapper=__import__('attr._make')._make._CacheHashWrapper):"
        hash_func = '_cache_wrapper(' + hash_func
        closing_braces += ')'
    method_lines = [hash_def]

    def append_hash_computation_lines(prefix, indent):
        if False:
            while True:
                i = 10
        '\n        Generate the code for actually computing the hash code.\n        Below this will either be returned directly or used to compute\n        a value which is then cached, depending on the value of cache_hash\n        '
        method_lines.extend([indent + prefix + hash_func, indent + f'        {type_hash},'])
        for a in attrs:
            if a.eq_key:
                cmp_name = f'_{a.name}_key'
                globs[cmp_name] = a.eq_key
                method_lines.append(indent + f'        {cmp_name}(self.{a.name}),')
            else:
                method_lines.append(indent + f'        self.{a.name},')
        method_lines.append(indent + '    ' + closing_braces)
    if cache_hash:
        method_lines.append(tab + f'if self.{_hash_cache_field} is None:')
        if frozen:
            append_hash_computation_lines(f"object.__setattr__(self, '{_hash_cache_field}', ", tab * 2)
            method_lines.append(tab * 2 + ')')
        else:
            append_hash_computation_lines(f'self.{_hash_cache_field} = ', tab * 2)
        method_lines.append(tab + f'return self.{_hash_cache_field}')
    else:
        append_hash_computation_lines('return ', tab)
    script = '\n'.join(method_lines)
    return _make_method('__hash__', script, unique_filename, globs)

def _add_hash(cls, attrs):
    if False:
        i = 10
        return i + 15
    '\n    Add a hash method to *cls*.\n    '
    cls.__hash__ = _make_hash(cls, attrs, frozen=False, cache_hash=False)
    return cls

def _make_ne():
    if False:
        return 10
    '\n    Create __ne__ method.\n    '

    def __ne__(self, other):
        if False:
            return 10
        '\n        Check equality and either forward a NotImplemented or\n        return the result negated.\n        '
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result
    return __ne__

def _make_eq(cls, attrs):
    if False:
        print('Hello World!')
    '\n    Create __eq__ method for *cls* with *attrs*.\n    '
    attrs = [a for a in attrs if a.eq]
    unique_filename = _generate_unique_filename(cls, 'eq')
    lines = ['def __eq__(self, other):', '    if other.__class__ is not self.__class__:', '        return NotImplemented']
    globs = {}
    if attrs:
        lines.append('    return  (')
        others = ['    ) == (']
        for a in attrs:
            if a.eq_key:
                cmp_name = f'_{a.name}_key'
                globs[cmp_name] = a.eq_key
                lines.append(f'        {cmp_name}(self.{a.name}),')
                others.append(f'        {cmp_name}(other.{a.name}),')
            else:
                lines.append(f'        self.{a.name},')
                others.append(f'        other.{a.name},')
        lines += [*others, '    )']
    else:
        lines.append('    return True')
    script = '\n'.join(lines)
    return _make_method('__eq__', script, unique_filename, globs)

def _make_order(cls, attrs):
    if False:
        while True:
            i = 10
    '\n    Create ordering methods for *cls* with *attrs*.\n    '
    attrs = [a for a in attrs if a.order]

    def attrs_to_tuple(obj):
        if False:
            return 10
        '\n        Save us some typing.\n        '
        return tuple((key(value) if key else value for (value, key) in ((getattr(obj, a.name), a.order_key) for a in attrs)))

    def __lt__(self, other):
        if False:
            print('Hello World!')
        '\n        Automatically created by attrs.\n        '
        if other.__class__ is self.__class__:
            return attrs_to_tuple(self) < attrs_to_tuple(other)
        return NotImplemented

    def __le__(self, other):
        if False:
            return 10
        '\n        Automatically created by attrs.\n        '
        if other.__class__ is self.__class__:
            return attrs_to_tuple(self) <= attrs_to_tuple(other)
        return NotImplemented

    def __gt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Automatically created by attrs.\n        '
        if other.__class__ is self.__class__:
            return attrs_to_tuple(self) > attrs_to_tuple(other)
        return NotImplemented

    def __ge__(self, other):
        if False:
            i = 10
            return i + 15
        '\n        Automatically created by attrs.\n        '
        if other.__class__ is self.__class__:
            return attrs_to_tuple(self) >= attrs_to_tuple(other)
        return NotImplemented
    return (__lt__, __le__, __gt__, __ge__)

def _add_eq(cls, attrs=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add equality methods to *cls* with *attrs*.\n    '
    if attrs is None:
        attrs = cls.__attrs_attrs__
    cls.__eq__ = _make_eq(cls, attrs)
    cls.__ne__ = _make_ne()
    return cls

def _make_repr(attrs, ns, cls):
    if False:
        print('Hello World!')
    unique_filename = _generate_unique_filename(cls, 'repr')
    attr_names_with_reprs = tuple(((a.name, repr if a.repr is True else a.repr, a.init) for a in attrs if a.repr is not False))
    globs = {name + '_repr': r for (name, r, _) in attr_names_with_reprs if r != repr}
    globs['_compat'] = _compat
    globs['AttributeError'] = AttributeError
    globs['NOTHING'] = NOTHING
    attribute_fragments = []
    for (name, r, i) in attr_names_with_reprs:
        accessor = 'self.' + name if i else 'getattr(self, "' + name + '", NOTHING)'
        fragment = '%s={%s!r}' % (name, accessor) if r == repr else '%s={%s_repr(%s)}' % (name, name, accessor)
        attribute_fragments.append(fragment)
    repr_fragment = ', '.join(attribute_fragments)
    if ns is None:
        cls_name_fragment = '{self.__class__.__qualname__.rsplit(">.", 1)[-1]}'
    else:
        cls_name_fragment = ns + '.{self.__class__.__name__}'
    lines = ['def __repr__(self):', '  try:', '    already_repring = _compat.repr_context.already_repring', '  except AttributeError:', '    already_repring = {id(self),}', '    _compat.repr_context.already_repring = already_repring', '  else:', '    if id(self) in already_repring:', "      return '...'", '    else:', '      already_repring.add(id(self))', '  try:', f"    return f'{cls_name_fragment}({repr_fragment})'", '  finally:', '    already_repring.remove(id(self))']
    return _make_method('__repr__', '\n'.join(lines), unique_filename, globs=globs)

def _add_repr(cls, ns=None, attrs=None):
    if False:
        while True:
            i = 10
    '\n    Add a repr method to *cls*.\n    '
    if attrs is None:
        attrs = cls.__attrs_attrs__
    cls.__repr__ = _make_repr(attrs, ns, cls)
    return cls

def fields(cls):
    if False:
        while True:
            i = 10
    '\n    Return the tuple of *attrs* attributes for a class.\n\n    The tuple also allows accessing the fields by their names (see below for\n    examples).\n\n    :param type cls: Class to introspect.\n\n    :raise TypeError: If *cls* is not a class.\n    :raise attrs.exceptions.NotAnAttrsClassError: If *cls* is not an *attrs*\n        class.\n\n    :rtype: tuple (with name accessors) of `attrs.Attribute`\n\n    .. versionchanged:: 16.2.0 Returned tuple allows accessing the fields\n       by name.\n    .. versionchanged:: 23.1.0 Add support for generic classes.\n    '
    generic_base = get_generic_base(cls)
    if generic_base is None and (not isinstance(cls, type)):
        msg = 'Passed object must be a class.'
        raise TypeError(msg)
    attrs = getattr(cls, '__attrs_attrs__', None)
    if attrs is None:
        if generic_base is not None:
            attrs = getattr(generic_base, '__attrs_attrs__', None)
            if attrs is not None:
                cls.__attrs_attrs__ = attrs
                return attrs
        msg = f'{cls!r} is not an attrs-decorated class.'
        raise NotAnAttrsClassError(msg)
    return attrs

def fields_dict(cls):
    if False:
        i = 10
        return i + 15
    '\n    Return an ordered dictionary of *attrs* attributes for a class, whose\n    keys are the attribute names.\n\n    :param type cls: Class to introspect.\n\n    :raise TypeError: If *cls* is not a class.\n    :raise attrs.exceptions.NotAnAttrsClassError: If *cls* is not an *attrs*\n        class.\n\n    :rtype: dict\n\n    .. versionadded:: 18.1.0\n    '
    if not isinstance(cls, type):
        msg = 'Passed object must be a class.'
        raise TypeError(msg)
    attrs = getattr(cls, '__attrs_attrs__', None)
    if attrs is None:
        msg = f'{cls!r} is not an attrs-decorated class.'
        raise NotAnAttrsClassError(msg)
    return {a.name: a for a in attrs}

def validate(inst):
    if False:
        i = 10
        return i + 15
    '\n    Validate all attributes on *inst* that have a validator.\n\n    Leaves all exceptions through.\n\n    :param inst: Instance of a class with *attrs* attributes.\n    '
    if _config._run_validators is False:
        return
    for a in fields(inst.__class__):
        v = a.validator
        if v is not None:
            v(inst, a, getattr(inst, a.name))

def _is_slot_cls(cls):
    if False:
        while True:
            i = 10
    return '__slots__' in cls.__dict__

def _is_slot_attr(a_name, base_attr_map):
    if False:
        i = 10
        return i + 15
    '\n    Check if the attribute name comes from a slot class.\n    '
    return a_name in base_attr_map and _is_slot_cls(base_attr_map[a_name])

def _make_init(cls, attrs, pre_init, pre_init_has_args, post_init, frozen, slots, cache_hash, base_attr_map, is_exc, cls_on_setattr, attrs_init):
    if False:
        while True:
            i = 10
    has_cls_on_setattr = cls_on_setattr is not None and cls_on_setattr is not setters.NO_OP
    if frozen and has_cls_on_setattr:
        msg = "Frozen classes can't use on_setattr."
        raise ValueError(msg)
    needs_cached_setattr = cache_hash or frozen
    filtered_attrs = []
    attr_dict = {}
    for a in attrs:
        if not a.init and a.default is NOTHING:
            continue
        filtered_attrs.append(a)
        attr_dict[a.name] = a
        if a.on_setattr is not None:
            if frozen is True:
                msg = "Frozen classes can't use on_setattr."
                raise ValueError(msg)
            needs_cached_setattr = True
        elif has_cls_on_setattr and a.on_setattr is not setters.NO_OP:
            needs_cached_setattr = True
    unique_filename = _generate_unique_filename(cls, 'init')
    (script, globs, annotations) = _attrs_to_init_script(filtered_attrs, frozen, slots, pre_init, pre_init_has_args, post_init, cache_hash, base_attr_map, is_exc, needs_cached_setattr, has_cls_on_setattr, attrs_init)
    if cls.__module__ in sys.modules:
        globs.update(sys.modules[cls.__module__].__dict__)
    globs.update({'NOTHING': NOTHING, 'attr_dict': attr_dict})
    if needs_cached_setattr:
        globs['_cached_setattr_get'] = _obj_setattr.__get__
    init = _make_method('__attrs_init__' if attrs_init else '__init__', script, unique_filename, globs)
    init.__annotations__ = annotations
    return init

def _setattr(attr_name, value_var, has_on_setattr):
    if False:
        for i in range(10):
            print('nop')
    '\n    Use the cached object.setattr to set *attr_name* to *value_var*.\n    '
    return f"_setattr('{attr_name}', {value_var})"

def _setattr_with_converter(attr_name, value_var, has_on_setattr):
    if False:
        print('Hello World!')
    '\n    Use the cached object.setattr to set *attr_name* to *value_var*, but run\n    its converter first.\n    '
    return "_setattr('%s', %s(%s))" % (attr_name, _init_converter_pat % (attr_name,), value_var)

def _assign(attr_name, value, has_on_setattr):
    if False:
        print('Hello World!')
    '\n    Unless *attr_name* has an on_setattr hook, use normal assignment. Otherwise\n    relegate to _setattr.\n    '
    if has_on_setattr:
        return _setattr(attr_name, value, True)
    return f'self.{attr_name} = {value}'

def _assign_with_converter(attr_name, value_var, has_on_setattr):
    if False:
        while True:
            i = 10
    '\n    Unless *attr_name* has an on_setattr hook, use normal assignment after\n    conversion. Otherwise relegate to _setattr_with_converter.\n    '
    if has_on_setattr:
        return _setattr_with_converter(attr_name, value_var, True)
    return 'self.%s = %s(%s)' % (attr_name, _init_converter_pat % (attr_name,), value_var)

def _attrs_to_init_script(attrs, frozen, slots, pre_init, pre_init_has_args, post_init, cache_hash, base_attr_map, is_exc, needs_cached_setattr, has_cls_on_setattr, attrs_init):
    if False:
        return 10
    '\n    Return a script of an initializer for *attrs* and a dict of globals.\n\n    The globals are expected by the generated script.\n\n    If *frozen* is True, we cannot set the attributes directly so we use\n    a cached ``object.__setattr__``.\n    '
    lines = []
    if pre_init:
        lines.append('self.__attrs_pre_init__()')
    if needs_cached_setattr:
        lines.append('_setattr = _cached_setattr_get(self)')
    if frozen is True:
        if slots is True:
            fmt_setter = _setattr
            fmt_setter_with_converter = _setattr_with_converter
        else:
            lines.append('_inst_dict = self.__dict__')

            def fmt_setter(attr_name, value_var, has_on_setattr):
                if False:
                    for i in range(10):
                        print('nop')
                if _is_slot_attr(attr_name, base_attr_map):
                    return _setattr(attr_name, value_var, has_on_setattr)
                return f"_inst_dict['{attr_name}'] = {value_var}"

            def fmt_setter_with_converter(attr_name, value_var, has_on_setattr):
                if False:
                    print('Hello World!')
                if has_on_setattr or _is_slot_attr(attr_name, base_attr_map):
                    return _setattr_with_converter(attr_name, value_var, has_on_setattr)
                return "_inst_dict['%s'] = %s(%s)" % (attr_name, _init_converter_pat % (attr_name,), value_var)
    else:
        fmt_setter = _assign
        fmt_setter_with_converter = _assign_with_converter
    args = []
    kw_only_args = []
    attrs_to_validate = []
    names_for_globals = {}
    annotations = {'return': None}
    for a in attrs:
        if a.validator:
            attrs_to_validate.append(a)
        attr_name = a.name
        has_on_setattr = a.on_setattr is not None or (a.on_setattr is not setters.NO_OP and has_cls_on_setattr)
        arg_name = a.alias
        has_factory = isinstance(a.default, Factory)
        maybe_self = 'self' if has_factory and a.default.takes_self else ''
        if a.init is False:
            if has_factory:
                init_factory_name = _init_factory_pat % (a.name,)
                if a.converter is not None:
                    lines.append(fmt_setter_with_converter(attr_name, init_factory_name + f'({maybe_self})', has_on_setattr))
                    conv_name = _init_converter_pat % (a.name,)
                    names_for_globals[conv_name] = a.converter
                else:
                    lines.append(fmt_setter(attr_name, init_factory_name + f'({maybe_self})', has_on_setattr))
                names_for_globals[init_factory_name] = a.default.factory
            elif a.converter is not None:
                lines.append(fmt_setter_with_converter(attr_name, f"attr_dict['{attr_name}'].default", has_on_setattr))
                conv_name = _init_converter_pat % (a.name,)
                names_for_globals[conv_name] = a.converter
            else:
                lines.append(fmt_setter(attr_name, f"attr_dict['{attr_name}'].default", has_on_setattr))
        elif a.default is not NOTHING and (not has_factory):
            arg = f"{arg_name}=attr_dict['{attr_name}'].default"
            if a.kw_only:
                kw_only_args.append(arg)
            else:
                args.append(arg)
            if a.converter is not None:
                lines.append(fmt_setter_with_converter(attr_name, arg_name, has_on_setattr))
                names_for_globals[_init_converter_pat % (a.name,)] = a.converter
            else:
                lines.append(fmt_setter(attr_name, arg_name, has_on_setattr))
        elif has_factory:
            arg = f'{arg_name}=NOTHING'
            if a.kw_only:
                kw_only_args.append(arg)
            else:
                args.append(arg)
            lines.append(f'if {arg_name} is not NOTHING:')
            init_factory_name = _init_factory_pat % (a.name,)
            if a.converter is not None:
                lines.append('    ' + fmt_setter_with_converter(attr_name, arg_name, has_on_setattr))
                lines.append('else:')
                lines.append('    ' + fmt_setter_with_converter(attr_name, init_factory_name + '(' + maybe_self + ')', has_on_setattr))
                names_for_globals[_init_converter_pat % (a.name,)] = a.converter
            else:
                lines.append('    ' + fmt_setter(attr_name, arg_name, has_on_setattr))
                lines.append('else:')
                lines.append('    ' + fmt_setter(attr_name, init_factory_name + '(' + maybe_self + ')', has_on_setattr))
            names_for_globals[init_factory_name] = a.default.factory
        else:
            if a.kw_only:
                kw_only_args.append(arg_name)
            else:
                args.append(arg_name)
            if a.converter is not None:
                lines.append(fmt_setter_with_converter(attr_name, arg_name, has_on_setattr))
                names_for_globals[_init_converter_pat % (a.name,)] = a.converter
            else:
                lines.append(fmt_setter(attr_name, arg_name, has_on_setattr))
        if a.init is True:
            if a.type is not None and a.converter is None:
                annotations[arg_name] = a.type
            elif a.converter is not None:
                t = _AnnotationExtractor(a.converter).get_first_param_type()
                if t:
                    annotations[arg_name] = t
    if attrs_to_validate:
        names_for_globals['_config'] = _config
        lines.append('if _config._run_validators is True:')
        for a in attrs_to_validate:
            val_name = '__attr_validator_' + a.name
            attr_name = '__attr_' + a.name
            lines.append(f'    {val_name}(self, {attr_name}, self.{a.name})')
            names_for_globals[val_name] = a.validator
            names_for_globals[attr_name] = a
    if post_init:
        lines.append('self.__attrs_post_init__()')
    if cache_hash:
        if frozen:
            if slots:
                init_hash_cache = "_setattr('%s', %s)"
            else:
                init_hash_cache = "_inst_dict['%s'] = %s"
        else:
            init_hash_cache = 'self.%s = %s'
        lines.append(init_hash_cache % (_hash_cache_field, 'None'))
    if is_exc:
        vals = ','.join((f'self.{a.name}' for a in attrs if a.init))
        lines.append(f'BaseException.__init__(self, {vals})')
    args = ', '.join(args)
    pre_init_args = args
    if kw_only_args:
        args += '%s*, %s' % (', ' if args else '', ', '.join(kw_only_args))
        pre_init_kw_only_args = ', '.join(['%s=%s' % (kw_arg, kw_arg) for kw_arg in kw_only_args])
        pre_init_args += ', ' if pre_init_args else ''
        pre_init_args += pre_init_kw_only_args
    if pre_init and pre_init_has_args:
        lines[0] = 'self.__attrs_pre_init__(%s)' % pre_init_args
    return ('def %s(self, %s):\n    %s\n' % ('__attrs_init__' if attrs_init else '__init__', args, '\n    '.join(lines) if lines else 'pass'), names_for_globals, annotations)

def _default_init_alias_for(name: str) -> str:
    if False:
        print('Hello World!')
    '\n    The default __init__ parameter name for a field.\n\n    This performs private-name adjustment via leading-unscore stripping,\n    and is the default value of Attribute.alias if not provided.\n    '
    return name.lstrip('_')

class Attribute:
    """
    *Read-only* representation of an attribute.

    .. warning::

       You should never instantiate this class yourself.

    The class has *all* arguments of `attr.ib` (except for ``factory``
    which is only syntactic sugar for ``default=Factory(...)`` plus the
    following:

    - ``name`` (`str`): The name of the attribute.
    - ``alias`` (`str`): The __init__ parameter name of the attribute, after
      any explicit overrides and default private-attribute-name handling.
    - ``inherited`` (`bool`): Whether or not that attribute has been inherited
      from a base class.
    - ``eq_key`` and ``order_key`` (`typing.Callable` or `None`): The callables
      that are used for comparing and ordering objects by this attribute,
      respectively. These are set by passing a callable to `attr.ib`'s ``eq``,
      ``order``, or ``cmp`` arguments. See also :ref:`comparison customization
      <custom-comparison>`.

    Instances of this class are frequently used for introspection purposes
    like:

    - `fields` returns a tuple of them.
    - Validators get them passed as the first argument.
    - The :ref:`field transformer <transform-fields>` hook receives a list of
      them.
    - The ``alias`` property exposes the __init__ parameter name of the field,
      with any overrides and default private-attribute handling applied.


    .. versionadded:: 20.1.0 *inherited*
    .. versionadded:: 20.1.0 *on_setattr*
    .. versionchanged:: 20.2.0 *inherited* is not taken into account for
        equality checks and hashing anymore.
    .. versionadded:: 21.1.0 *eq_key* and *order_key*
    .. versionadded:: 22.2.0 *alias*

    For the full version history of the fields, see `attr.ib`.
    """
    __slots__ = ('name', 'default', 'validator', 'repr', 'eq', 'eq_key', 'order', 'order_key', 'hash', 'init', 'metadata', 'type', 'converter', 'kw_only', 'inherited', 'on_setattr', 'alias')

    def __init__(self, name, default, validator, repr, cmp, hash, init, inherited, metadata=None, type=None, converter=None, kw_only=False, eq=None, eq_key=None, order=None, order_key=None, on_setattr=None, alias=None):
        if False:
            return 10
        (eq, eq_key, order, order_key) = _determine_attrib_eq_order(cmp, eq_key or eq, order_key or order, True)
        bound_setattr = _obj_setattr.__get__(self)
        bound_setattr('name', name)
        bound_setattr('default', default)
        bound_setattr('validator', validator)
        bound_setattr('repr', repr)
        bound_setattr('eq', eq)
        bound_setattr('eq_key', eq_key)
        bound_setattr('order', order)
        bound_setattr('order_key', order_key)
        bound_setattr('hash', hash)
        bound_setattr('init', init)
        bound_setattr('converter', converter)
        bound_setattr('metadata', types.MappingProxyType(dict(metadata)) if metadata else _empty_metadata_singleton)
        bound_setattr('type', type)
        bound_setattr('kw_only', kw_only)
        bound_setattr('inherited', inherited)
        bound_setattr('on_setattr', on_setattr)
        bound_setattr('alias', alias)

    def __setattr__(self, name, value):
        if False:
            i = 10
            return i + 15
        raise FrozenInstanceError()

    @classmethod
    def from_counting_attr(cls, name, ca, type=None):
        if False:
            return 10
        if type is None:
            type = ca.type
        elif ca.type is not None:
            msg = 'Type annotation and type argument cannot both be present'
            raise ValueError(msg)
        inst_dict = {k: getattr(ca, k) for k in Attribute.__slots__ if k not in ('name', 'validator', 'default', 'type', 'inherited')}
        return cls(name=name, validator=ca._validator, default=ca._default, type=type, cmp=None, inherited=False, **inst_dict)

    def evolve(self, **changes):
        if False:
            i = 10
            return i + 15
        '\n        Copy *self* and apply *changes*.\n\n        This works similarly to `attrs.evolve` but that function does not work\n        with `Attribute`.\n\n        It is mainly meant to be used for `transform-fields`.\n\n        .. versionadded:: 20.3.0\n        '
        new = copy.copy(self)
        new._setattrs(changes.items())
        return new

    def __getstate__(self):
        if False:
            while True:
                i = 10
        '\n        Play nice with pickle.\n        '
        return tuple((getattr(self, name) if name != 'metadata' else dict(self.metadata) for name in self.__slots__))

    def __setstate__(self, state):
        if False:
            return 10
        '\n        Play nice with pickle.\n        '
        self._setattrs(zip(self.__slots__, state))

    def _setattrs(self, name_values_pairs):
        if False:
            i = 10
            return i + 15
        bound_setattr = _obj_setattr.__get__(self)
        for (name, value) in name_values_pairs:
            if name != 'metadata':
                bound_setattr(name, value)
            else:
                bound_setattr(name, types.MappingProxyType(dict(value)) if value else _empty_metadata_singleton)
_a = [Attribute(name=name, default=NOTHING, validator=None, repr=True, cmp=None, eq=True, order=False, hash=name != 'metadata', init=True, inherited=False, alias=_default_init_alias_for(name)) for name in Attribute.__slots__]
Attribute = _add_hash(_add_eq(_add_repr(Attribute, attrs=_a), attrs=[a for a in _a if a.name != 'inherited']), attrs=[a for a in _a if a.hash and a.name != 'inherited'])

class _CountingAttr:
    """
    Intermediate representation of attributes that uses a counter to preserve
    the order in which the attributes have been defined.

    *Internal* data structure of the attrs library.  Running into is most
    likely the result of a bug like a forgotten `@attr.s` decorator.
    """
    __slots__ = ('counter', '_default', 'repr', 'eq', 'eq_key', 'order', 'order_key', 'hash', 'init', 'metadata', '_validator', 'converter', 'type', 'kw_only', 'on_setattr', 'alias')
    __attrs_attrs__ = (*tuple((Attribute(name=name, alias=_default_init_alias_for(name), default=NOTHING, validator=None, repr=True, cmp=None, hash=True, init=True, kw_only=False, eq=True, eq_key=None, order=False, order_key=None, inherited=False, on_setattr=None) for name in ('counter', '_default', 'repr', 'eq', 'order', 'hash', 'init', 'on_setattr', 'alias'))), Attribute(name='metadata', alias='metadata', default=None, validator=None, repr=True, cmp=None, hash=False, init=True, kw_only=False, eq=True, eq_key=None, order=False, order_key=None, inherited=False, on_setattr=None))
    cls_counter = 0

    def __init__(self, default, validator, repr, cmp, hash, init, converter, metadata, type, kw_only, eq, eq_key, order, order_key, on_setattr, alias):
        if False:
            i = 10
            return i + 15
        _CountingAttr.cls_counter += 1
        self.counter = _CountingAttr.cls_counter
        self._default = default
        self._validator = validator
        self.converter = converter
        self.repr = repr
        self.eq = eq
        self.eq_key = eq_key
        self.order = order
        self.order_key = order_key
        self.hash = hash
        self.init = init
        self.metadata = metadata
        self.type = type
        self.kw_only = kw_only
        self.on_setattr = on_setattr
        self.alias = alias

    def validator(self, meth):
        if False:
            while True:
                i = 10
        '\n        Decorator that adds *meth* to the list of validators.\n\n        Returns *meth* unchanged.\n\n        .. versionadded:: 17.1.0\n        '
        if self._validator is None:
            self._validator = meth
        else:
            self._validator = and_(self._validator, meth)
        return meth

    def default(self, meth):
        if False:
            for i in range(10):
                print('nop')
        '\n        Decorator that allows to set the default for an attribute.\n\n        Returns *meth* unchanged.\n\n        :raises DefaultAlreadySetError: If default has been set before.\n\n        .. versionadded:: 17.1.0\n        '
        if self._default is not NOTHING:
            raise DefaultAlreadySetError()
        self._default = Factory(meth, takes_self=True)
        return meth
_CountingAttr = _add_eq(_add_repr(_CountingAttr))

class Factory:
    """
    Stores a factory callable.

    If passed as the default value to `attrs.field`, the factory is used to
    generate a new value.

    :param callable factory: A callable that takes either none or exactly one
        mandatory positional argument depending on *takes_self*.
    :param bool takes_self: Pass the partially initialized instance that is
        being initialized as a positional argument.

    .. versionadded:: 17.1.0  *takes_self*
    """
    __slots__ = ('factory', 'takes_self')

    def __init__(self, factory, takes_self=False):
        if False:
            i = 10
            return i + 15
        self.factory = factory
        self.takes_self = takes_self

    def __getstate__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Play nice with pickle.\n        '
        return tuple((getattr(self, name) for name in self.__slots__))

    def __setstate__(self, state):
        if False:
            while True:
                i = 10
        '\n        Play nice with pickle.\n        '
        for (name, value) in zip(self.__slots__, state):
            setattr(self, name, value)
_f = [Attribute(name=name, default=NOTHING, validator=None, repr=True, cmp=None, eq=True, order=False, hash=True, init=True, inherited=False) for name in Factory.__slots__]
Factory = _add_hash(_add_eq(_add_repr(Factory, attrs=_f), attrs=_f), attrs=_f)

def make_class(name, attrs, bases=(object,), **attributes_arguments):
    if False:
        while True:
            i = 10
    '\n    A quick way to create a new class called *name* with *attrs*.\n\n    :param str name: The name for the new class.\n\n    :param attrs: A list of names or a dictionary of mappings of names to\n        `attr.ib`\\ s / `attrs.field`\\ s.\n\n        The order is deduced from the order of the names or attributes inside\n        *attrs*.  Otherwise the order of the definition of the attributes is\n        used.\n    :type attrs: `list` or `dict`\n\n    :param tuple bases: Classes that the new class will subclass.\n\n    :param attributes_arguments: Passed unmodified to `attr.s`.\n\n    :return: A new class with *attrs*.\n    :rtype: type\n\n    .. versionadded:: 17.1.0 *bases*\n    .. versionchanged:: 18.1.0 If *attrs* is ordered, the order is retained.\n    '
    if isinstance(attrs, dict):
        cls_dict = attrs
    elif isinstance(attrs, (list, tuple)):
        cls_dict = {a: attrib() for a in attrs}
    else:
        msg = 'attrs argument must be a dict or a list.'
        raise TypeError(msg)
    pre_init = cls_dict.pop('__attrs_pre_init__', None)
    post_init = cls_dict.pop('__attrs_post_init__', None)
    user_init = cls_dict.pop('__init__', None)
    body = {}
    if pre_init is not None:
        body['__attrs_pre_init__'] = pre_init
    if post_init is not None:
        body['__attrs_post_init__'] = post_init
    if user_init is not None:
        body['__init__'] = user_init
    type_ = types.new_class(name, bases, {}, lambda ns: ns.update(body))
    with contextlib.suppress(AttributeError, ValueError):
        type_.__module__ = sys._getframe(1).f_globals.get('__name__', '__main__')
    cmp = attributes_arguments.pop('cmp', None)
    (attributes_arguments['eq'], attributes_arguments['order']) = _determine_attrs_eq_order(cmp, attributes_arguments.get('eq'), attributes_arguments.get('order'), True)
    return _attrs(these=cls_dict, **attributes_arguments)(type_)

@attrs(slots=True, hash=True)
class _AndValidator:
    """
    Compose many validators to a single one.
    """
    _validators = attrib()

    def __call__(self, inst, attr, value):
        if False:
            i = 10
            return i + 15
        for v in self._validators:
            v(inst, attr, value)

def and_(*validators):
    if False:
        print('Hello World!')
    '\n    A validator that composes multiple validators into one.\n\n    When called on a value, it runs all wrapped validators.\n\n    :param callables validators: Arbitrary number of validators.\n\n    .. versionadded:: 17.1.0\n    '
    vals = []
    for validator in validators:
        vals.extend(validator._validators if isinstance(validator, _AndValidator) else [validator])
    return _AndValidator(tuple(vals))

def pipe(*converters):
    if False:
        for i in range(10):
            print('nop')
    "\n    A converter that composes multiple converters into one.\n\n    When called on a value, it runs all wrapped converters, returning the\n    *last* value.\n\n    Type annotations will be inferred from the wrapped converters', if\n    they have any.\n\n    :param callables converters: Arbitrary number of converters.\n\n    .. versionadded:: 20.1.0\n    "

    def pipe_converter(val):
        if False:
            print('Hello World!')
        for converter in converters:
            val = converter(val)
        return val
    if not converters:
        A = typing.TypeVar('A')
        pipe_converter.__annotations__ = {'val': A, 'return': A}
    else:
        t = _AnnotationExtractor(converters[0]).get_first_param_type()
        if t:
            pipe_converter.__annotations__['val'] = t
        rt = _AnnotationExtractor(converters[-1]).get_return_type()
        if rt:
            pipe_converter.__annotations__['return'] = rt
    return pipe_converter