from __future__ import annotations
import enum
from itertools import zip_longest
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from .visitors import anon_map
from .visitors import HasTraversalDispatch
from .visitors import HasTraverseInternals
from .visitors import InternalTraversal
from .visitors import prefix_anon_map
from .. import util
from ..inspection import inspect
from ..util import HasMemoized
from ..util.typing import Literal
from ..util.typing import Protocol
if typing.TYPE_CHECKING:
    from .elements import BindParameter
    from .elements import ClauseElement
    from .visitors import _TraverseInternalsType
    from ..engine.interfaces import _CoreSingleExecuteParams

class _CacheKeyTraversalDispatchType(Protocol):

    def __call__(s, self: HasCacheKey, visitor: _CacheKeyTraversal) -> _CacheKeyTraversalDispatchTypeReturn:
        if False:
            for i in range(10):
                print('nop')
        ...

class CacheConst(enum.Enum):
    NO_CACHE = 0
NO_CACHE = CacheConst.NO_CACHE
_CacheKeyTraversalType = Union['_TraverseInternalsType', Literal[CacheConst.NO_CACHE], Literal[None]]

class CacheTraverseTarget(enum.Enum):
    CACHE_IN_PLACE = 0
    CALL_GEN_CACHE_KEY = 1
    STATIC_CACHE_KEY = 2
    PROPAGATE_ATTRS = 3
    ANON_NAME = 4
(CACHE_IN_PLACE, CALL_GEN_CACHE_KEY, STATIC_CACHE_KEY, PROPAGATE_ATTRS, ANON_NAME) = tuple(CacheTraverseTarget)
_CacheKeyTraversalDispatchTypeReturn = Sequence[Tuple[str, Any, Union[Callable[..., Tuple[Any, ...]], CacheTraverseTarget, InternalTraversal]]]

class HasCacheKey:
    """Mixin for objects which can produce a cache key.

    This class is usually in a hierarchy that starts with the
    :class:`.HasTraverseInternals` base, but this is optional.  Currently,
    the class should be able to work on its own without including
    :class:`.HasTraverseInternals`.

    .. seealso::

        :class:`.CacheKey`

        :ref:`sql_caching`

    """
    __slots__ = ()
    _cache_key_traversal: _CacheKeyTraversalType = NO_CACHE
    _is_has_cache_key = True
    _hierarchy_supports_caching = True
    'private attribute which may be set to False to prevent the\n    inherit_cache warning from being emitted for a hierarchy of subclasses.\n\n    Currently applies to the :class:`.ExecutableDDLElement` hierarchy which\n    does not implement caching.\n\n    '
    inherit_cache: Optional[bool] = None
    'Indicate if this :class:`.HasCacheKey` instance should make use of the\n    cache key generation scheme used by its immediate superclass.\n\n    The attribute defaults to ``None``, which indicates that a construct has\n    not yet taken into account whether or not its appropriate for it to\n    participate in caching; this is functionally equivalent to setting the\n    value to ``False``, except that a warning is also emitted.\n\n    This flag can be set to ``True`` on a particular class, if the SQL that\n    corresponds to the object does not change based on attributes which\n    are local to this class, and not its superclass.\n\n    .. seealso::\n\n        :ref:`compilerext_caching` - General guideslines for setting the\n        :attr:`.HasCacheKey.inherit_cache` attribute for third-party or user\n        defined SQL constructs.\n\n    '
    __slots__ = ()
    _generated_cache_key_traversal: Any

    @classmethod
    def _generate_cache_attrs(cls) -> Union[_CacheKeyTraversalDispatchType, Literal[CacheConst.NO_CACHE]]:
        if False:
            i = 10
            return i + 15
        'generate cache key dispatcher for a new class.\n\n        This sets the _generated_cache_key_traversal attribute once called\n        so should only be called once per class.\n\n        '
        inherit_cache = cls.__dict__.get('inherit_cache', None)
        inherit = bool(inherit_cache)
        if inherit:
            _cache_key_traversal = getattr(cls, '_cache_key_traversal', None)
            if _cache_key_traversal is None:
                try:
                    assert issubclass(cls, HasTraverseInternals)
                    _cache_key_traversal = cls._traverse_internals
                except AttributeError:
                    cls._generated_cache_key_traversal = NO_CACHE
                    return NO_CACHE
            assert _cache_key_traversal is not NO_CACHE, f'class {cls} has _cache_key_traversal=NO_CACHE, which conflicts with inherit_cache=True'
            return _cache_key_traversal_visitor.generate_dispatch(cls, _cache_key_traversal, '_generated_cache_key_traversal')
        else:
            _cache_key_traversal = cls.__dict__.get('_cache_key_traversal', None)
            if _cache_key_traversal is None:
                _cache_key_traversal = cls.__dict__.get('_traverse_internals', None)
                if _cache_key_traversal is None:
                    cls._generated_cache_key_traversal = NO_CACHE
                    if inherit_cache is None and cls._hierarchy_supports_caching:
                        util.warn("Class %s will not make use of SQL compilation caching as it does not set the 'inherit_cache' attribute to ``True``.  This can have significant performance implications including some performance degradations in comparison to prior SQLAlchemy versions.  Set this attribute to True if this object can make use of the cache key generated by the superclass.  Alternatively, this attribute may be set to False which will disable this warning." % cls.__name__, code='cprf')
                    return NO_CACHE
            return _cache_key_traversal_visitor.generate_dispatch(cls, _cache_key_traversal, '_generated_cache_key_traversal')

    @util.preload_module('sqlalchemy.sql.elements')
    def _gen_cache_key(self, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Optional[Tuple[Any, ...]]:
        if False:
            print('Hello World!')
        'return an optional cache key.\n\n        The cache key is a tuple which can contain any series of\n        objects that are hashable and also identifies\n        this object uniquely within the presence of a larger SQL expression\n        or statement, for the purposes of caching the resulting query.\n\n        The cache key should be based on the SQL compiled structure that would\n        ultimately be produced.   That is, two structures that are composed in\n        exactly the same way should produce the same cache key; any difference\n        in the structures that would affect the SQL string or the type handlers\n        should result in a different cache key.\n\n        If a structure cannot produce a useful cache key, the NO_CACHE\n        symbol should be added to the anon_map and the method should\n        return None.\n\n        '
        cls = self.__class__
        (id_, found) = anon_map.get_anon(self)
        if found:
            return (id_, cls)
        dispatcher: Union[Literal[CacheConst.NO_CACHE], _CacheKeyTraversalDispatchType]
        try:
            dispatcher = cls.__dict__['_generated_cache_key_traversal']
        except KeyError:
            dispatcher = cls._generate_cache_attrs()
        if dispatcher is NO_CACHE:
            anon_map[NO_CACHE] = True
            return None
        result: Tuple[Any, ...] = (id_, cls)
        for (attrname, obj, meth) in dispatcher(self, _cache_key_traversal_visitor):
            if obj is not None:
                if meth is STATIC_CACHE_KEY:
                    sck = obj._static_cache_key
                    if sck is NO_CACHE:
                        anon_map[NO_CACHE] = True
                        return None
                    result += (attrname, sck)
                elif meth is ANON_NAME:
                    elements = util.preloaded.sql_elements
                    if isinstance(obj, elements._anonymous_label):
                        obj = obj.apply_map(anon_map)
                    result += (attrname, obj)
                elif meth is CALL_GEN_CACHE_KEY:
                    result += (attrname, obj._gen_cache_key(anon_map, bindparams))
                elif obj:
                    if meth is CACHE_IN_PLACE:
                        result += (attrname, obj)
                    elif meth is PROPAGATE_ATTRS:
                        result += (attrname, obj['compile_state_plugin'], obj['plugin_subject']._gen_cache_key(anon_map, bindparams) if obj['plugin_subject'] else None)
                    elif meth is InternalTraversal.dp_annotations_key:
                        if self._gen_static_annotations_cache_key:
                            result += self._annotations_cache_key
                        else:
                            result += self._gen_annotations_cache_key(anon_map)
                    elif meth is InternalTraversal.dp_clauseelement_list or meth is InternalTraversal.dp_clauseelement_tuple or meth is InternalTraversal.dp_memoized_select_entities:
                        result += (attrname, tuple([elem._gen_cache_key(anon_map, bindparams) for elem in obj]))
                    else:
                        result += meth(attrname, obj, self, anon_map, bindparams)
        return result

    def _generate_cache_key(self) -> Optional[CacheKey]:
        if False:
            print('Hello World!')
        'return a cache key.\n\n        The cache key is a tuple which can contain any series of\n        objects that are hashable and also identifies\n        this object uniquely within the presence of a larger SQL expression\n        or statement, for the purposes of caching the resulting query.\n\n        The cache key should be based on the SQL compiled structure that would\n        ultimately be produced.   That is, two structures that are composed in\n        exactly the same way should produce the same cache key; any difference\n        in the structures that would affect the SQL string or the type handlers\n        should result in a different cache key.\n\n        The cache key returned by this method is an instance of\n        :class:`.CacheKey`, which consists of a tuple representing the\n        cache key, as well as a list of :class:`.BindParameter` objects\n        which are extracted from the expression.   While two expressions\n        that produce identical cache key tuples will themselves generate\n        identical SQL strings, the list of :class:`.BindParameter` objects\n        indicates the bound values which may have different values in\n        each one; these bound parameters must be consulted in order to\n        execute the statement with the correct parameters.\n\n        a :class:`_expression.ClauseElement` structure that does not implement\n        a :meth:`._gen_cache_key` method and does not implement a\n        :attr:`.traverse_internals` attribute will not be cacheable; when\n        such an element is embedded into a larger structure, this method\n        will return None, indicating no cache key is available.\n\n        '
        bindparams: List[BindParameter[Any]] = []
        _anon_map = anon_map()
        key = self._gen_cache_key(_anon_map, bindparams)
        if NO_CACHE in _anon_map:
            return None
        else:
            assert key is not None
            return CacheKey(key, bindparams)

    @classmethod
    def _generate_cache_key_for_object(cls, obj: HasCacheKey) -> Optional[CacheKey]:
        if False:
            while True:
                i = 10
        bindparams: List[BindParameter[Any]] = []
        _anon_map = anon_map()
        key = obj._gen_cache_key(_anon_map, bindparams)
        if NO_CACHE in _anon_map:
            return None
        else:
            assert key is not None
            return CacheKey(key, bindparams)

class HasCacheKeyTraverse(HasTraverseInternals, HasCacheKey):
    pass

class MemoizedHasCacheKey(HasCacheKey, HasMemoized):
    __slots__ = ()

    @HasMemoized.memoized_instancemethod
    def _generate_cache_key(self) -> Optional[CacheKey]:
        if False:
            for i in range(10):
                print('nop')
        return HasCacheKey._generate_cache_key(self)

class SlotsMemoizedHasCacheKey(HasCacheKey, util.MemoizedSlots):
    __slots__ = ()

    def _memoized_method__generate_cache_key(self) -> Optional[CacheKey]:
        if False:
            print('Hello World!')
        return HasCacheKey._generate_cache_key(self)

class CacheKey(NamedTuple):
    """The key used to identify a SQL statement construct in the
    SQL compilation cache.

    .. seealso::

        :ref:`sql_caching`

    """
    key: Tuple[Any, ...]
    bindparams: Sequence[BindParameter[Any]]

    def __hash__(self) -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        'CacheKey itself is not hashable - hash the .key portion'
        return None

    def to_offline_string(self, statement_cache: MutableMapping[Any, str], statement: ClauseElement, parameters: _CoreSingleExecuteParams) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Generate an "offline string" form of this :class:`.CacheKey`\n\n        The "offline string" is basically the string SQL for the\n        statement plus a repr of the bound parameter values in series.\n        Whereas the :class:`.CacheKey` object is dependent on in-memory\n        identities in order to work as a cache key, the "offline" version\n        is suitable for a cache that will work for other processes as well.\n\n        The given ``statement_cache`` is a dictionary-like object where the\n        string form of the statement itself will be cached.  This dictionary\n        should be in a longer lived scope in order to reduce the time spent\n        stringifying statements.\n\n\n        '
        if self.key not in statement_cache:
            statement_cache[self.key] = sql_str = str(statement)
        else:
            sql_str = statement_cache[self.key]
        if not self.bindparams:
            param_tuple = tuple((parameters[key] for key in sorted(parameters)))
        else:
            param_tuple = tuple((parameters.get(bindparam.key, bindparam.value) for bindparam in self.bindparams))
        return repr((sql_str, param_tuple))

    def __eq__(self, other: Any) -> bool:
        if False:
            i = 10
            return i + 15
        return bool(self.key == other.key)

    def __ne__(self, other: Any) -> bool:
        if False:
            print('Hello World!')
        return not self.key == other.key

    @classmethod
    def _diff_tuples(cls, left: CacheKey, right: CacheKey) -> str:
        if False:
            while True:
                i = 10
        ck1 = CacheKey(left, [])
        ck2 = CacheKey(right, [])
        return ck1._diff(ck2)

    def _whats_different(self, other: CacheKey) -> Iterator[str]:
        if False:
            i = 10
            return i + 15
        k1 = self.key
        k2 = other.key
        stack: List[int] = []
        pickup_index = 0
        while True:
            (s1, s2) = (k1, k2)
            for idx in stack:
                s1 = s1[idx]
                s2 = s2[idx]
            for (idx, (e1, e2)) in enumerate(zip_longest(s1, s2)):
                if idx < pickup_index:
                    continue
                if e1 != e2:
                    if isinstance(e1, tuple) and isinstance(e2, tuple):
                        stack.append(idx)
                        break
                    else:
                        yield ('key%s[%d]:  %s != %s' % (''.join(('[%d]' % id_ for id_ in stack)), idx, e1, e2))
            else:
                pickup_index = stack.pop(-1)
                break

    def _diff(self, other: CacheKey) -> str:
        if False:
            print('Hello World!')
        return ', '.join(self._whats_different(other))

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        stack: List[Union[Tuple[Any, ...], HasCacheKey]] = [self.key]
        output = []
        sentinel = object()
        indent = -1
        while stack:
            elem = stack.pop(0)
            if elem is sentinel:
                output.append(' ' * (indent * 2) + '),')
                indent -= 1
            elif isinstance(elem, tuple):
                if not elem:
                    output.append(' ' * ((indent + 1) * 2) + '()')
                else:
                    indent += 1
                    stack = list(elem) + [sentinel] + stack
                    output.append(' ' * (indent * 2) + '(')
            else:
                if isinstance(elem, HasCacheKey):
                    repr_ = '<%s object at %s>' % (type(elem).__name__, hex(id(elem)))
                else:
                    repr_ = repr(elem)
                output.append(' ' * (indent * 2) + '  ' + repr_ + ', ')
        return 'CacheKey(key=%s)' % ('\n'.join(output),)

    def _generate_param_dict(self) -> Dict[str, Any]:
        if False:
            return 10
        'used for testing'
        _anon_map = prefix_anon_map()
        return {b.key % _anon_map: b.effective_value for b in self.bindparams}

    def _apply_params_to_element(self, original_cache_key: CacheKey, target_element: ClauseElement) -> ClauseElement:
        if False:
            i = 10
            return i + 15
        if target_element._is_immutable:
            return target_element
        translate = {k.key: v.value for (k, v) in zip(original_cache_key.bindparams, self.bindparams)}
        return target_element.params(translate)

def _ad_hoc_cache_key_from_args(tokens: Tuple[Any, ...], traverse_args: Iterable[Tuple[str, InternalTraversal]], args: Iterable[Any]) -> Tuple[Any, ...]:
    if False:
        print('Hello World!')
    'a quick cache key generator used by reflection.flexi_cache.'
    bindparams: List[BindParameter[Any]] = []
    _anon_map = anon_map()
    tup = tokens
    for ((attrname, sym), arg) in zip(traverse_args, args):
        key = sym.name
        visit_key = key.replace('dp_', 'visit_')
        if arg is None:
            tup += (attrname, None)
            continue
        meth = getattr(_cache_key_traversal_visitor, visit_key)
        if meth is CACHE_IN_PLACE:
            tup += (attrname, arg)
        elif meth in (CALL_GEN_CACHE_KEY, STATIC_CACHE_KEY, ANON_NAME, PROPAGATE_ATTRS):
            raise NotImplementedError(f"Haven't implemented symbol {meth} for ad-hoc key from args")
        else:
            tup += meth(attrname, arg, None, _anon_map, bindparams)
    return tup

class _CacheKeyTraversal(HasTraversalDispatch):
    visit_has_cache_key = visit_clauseelement = CALL_GEN_CACHE_KEY
    visit_clauseelement_list = InternalTraversal.dp_clauseelement_list
    visit_annotations_key = InternalTraversal.dp_annotations_key
    visit_clauseelement_tuple = InternalTraversal.dp_clauseelement_tuple
    visit_memoized_select_entities = InternalTraversal.dp_memoized_select_entities
    visit_string = visit_boolean = visit_operator = visit_plain_obj = CACHE_IN_PLACE
    visit_statement_hint_list = CACHE_IN_PLACE
    visit_type = STATIC_CACHE_KEY
    visit_anon_name = ANON_NAME
    visit_propagate_attrs = PROPAGATE_ATTRS

    def visit_with_context_options(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if False:
            return 10
        return tuple(((fn.__code__, c_key) for (fn, c_key) in obj))

    def visit_inspectable(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if False:
            for i in range(10):
                print('nop')
        return (attrname, inspect(obj)._gen_cache_key(anon_map, bindparams))

    def visit_string_list(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if False:
            while True:
                i = 10
        return tuple(obj)

    def visit_multi(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if False:
            i = 10
            return i + 15
        return (attrname, obj._gen_cache_key(anon_map, bindparams) if isinstance(obj, HasCacheKey) else obj)

    def visit_multi_list(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if False:
            while True:
                i = 10
        return (attrname, tuple((elem._gen_cache_key(anon_map, bindparams) if isinstance(elem, HasCacheKey) else elem for elem in obj)))

    def visit_has_cache_key_tuples(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if False:
            for i in range(10):
                print('nop')
        if not obj:
            return ()
        return (attrname, tuple((tuple((elem._gen_cache_key(anon_map, bindparams) for elem in tup_elem)) for tup_elem in obj)))

    def visit_has_cache_key_list(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if False:
            return 10
        if not obj:
            return ()
        return (attrname, tuple((elem._gen_cache_key(anon_map, bindparams) for elem in obj)))

    def visit_executable_options(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if False:
            return 10
        if not obj:
            return ()
        return (attrname, tuple((elem._gen_cache_key(anon_map, bindparams) for elem in obj if elem._is_has_cache_key)))

    def visit_inspectable_list(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if False:
            return 10
        return self.visit_has_cache_key_list(attrname, [inspect(o) for o in obj], parent, anon_map, bindparams)

    def visit_clauseelement_tuples(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if False:
            while True:
                i = 10
        return self.visit_has_cache_key_tuples(attrname, obj, parent, anon_map, bindparams)

    def visit_fromclause_ordered_set(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if False:
            while True:
                i = 10
        if not obj:
            return ()
        return (attrname, tuple([elem._gen_cache_key(anon_map, bindparams) for elem in obj]))

    def visit_clauseelement_unordered_set(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if False:
            i = 10
            return i + 15
        if not obj:
            return ()
        cache_keys = [elem._gen_cache_key(anon_map, bindparams) for elem in obj]
        return (attrname, tuple(sorted(cache_keys)))

    def visit_named_ddl_element(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if False:
            for i in range(10):
                print('nop')
        return (attrname, obj.name)

    def visit_prefix_sequence(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if False:
            return 10
        if not obj:
            return ()
        return (attrname, tuple([(clause._gen_cache_key(anon_map, bindparams), strval) for (clause, strval) in obj]))

    def visit_setup_join_tuple(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if False:
            return 10
        return tuple(((target._gen_cache_key(anon_map, bindparams), onclause._gen_cache_key(anon_map, bindparams) if onclause is not None else None, from_._gen_cache_key(anon_map, bindparams) if from_ is not None else None, tuple([(key, flags[key]) for key in sorted(flags)])) for (target, onclause, from_, flags) in obj))

    def visit_table_hint_list(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if False:
            while True:
                i = 10
        if not obj:
            return ()
        return (attrname, tuple([(clause._gen_cache_key(anon_map, bindparams), dialect_name, text) for ((clause, dialect_name), text) in obj.items()]))

    def visit_plain_dict(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if False:
            for i in range(10):
                print('nop')
        return (attrname, tuple([(key, obj[key]) for key in sorted(obj)]))

    def visit_dialect_options(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if False:
            return 10
        return (attrname, tuple(((dialect_name, tuple([(key, obj[dialect_name][key]) for key in sorted(obj[dialect_name])])) for dialect_name in sorted(obj))))

    def visit_string_clauseelement_dict(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if False:
            print('Hello World!')
        return (attrname, tuple(((key, obj[key]._gen_cache_key(anon_map, bindparams)) for key in sorted(obj))))

    def visit_string_multi_dict(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if False:
            while True:
                i = 10
        return (attrname, tuple(((key, value._gen_cache_key(anon_map, bindparams) if isinstance(value, HasCacheKey) else value) for (key, value) in [(key, obj[key]) for key in sorted(obj)])))

    def visit_fromclause_canonical_column_collection(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if False:
            for i in range(10):
                print('nop')
        return (attrname, tuple((col._gen_cache_key(anon_map, bindparams) for (k, col, _) in obj._collection)))

    def visit_unknown_structure(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if False:
            i = 10
            return i + 15
        anon_map[NO_CACHE] = True
        return ()

    def visit_dml_ordered_values(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if False:
            print('Hello World!')
        return (attrname, tuple(((key._gen_cache_key(anon_map, bindparams) if hasattr(key, '__clause_element__') else key, value._gen_cache_key(anon_map, bindparams)) for (key, value) in obj)))

    def visit_dml_values(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if False:
            for i in range(10):
                print('nop')
        return (attrname, tuple(((k._gen_cache_key(anon_map, bindparams) if hasattr(k, '__clause_element__') else k, obj[k]._gen_cache_key(anon_map, bindparams)) for k in obj)))

    def visit_dml_multi_values(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if False:
            print('Hello World!')
        anon_map[NO_CACHE] = True
        return ()
_cache_key_traversal_visitor = _CacheKeyTraversal()