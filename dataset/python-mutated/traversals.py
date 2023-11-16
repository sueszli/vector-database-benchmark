from __future__ import annotations
from collections import deque
import collections.abc as collections_abc
import itertools
from itertools import zip_longest
import operator
import typing
from typing import Any
from typing import Callable
from typing import Deque
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from . import operators
from .cache_key import HasCacheKey
from .visitors import _TraverseInternalsType
from .visitors import anon_map
from .visitors import ExternallyTraversible
from .visitors import HasTraversalDispatch
from .visitors import HasTraverseInternals
from .. import util
from ..util import langhelpers
from ..util.typing import Self
SKIP_TRAVERSE = util.symbol('skip_traverse')
COMPARE_FAILED = False
COMPARE_SUCCEEDED = True

def compare(obj1: Any, obj2: Any, **kw: Any) -> bool:
    if False:
        print('Hello World!')
    strategy: TraversalComparatorStrategy
    if kw.get('use_proxies', False):
        strategy = ColIdentityComparatorStrategy()
    else:
        strategy = TraversalComparatorStrategy()
    return strategy.compare(obj1, obj2, **kw)

def _preconfigure_traversals(target_hierarchy: Type[Any]) -> None:
    if False:
        for i in range(10):
            print('nop')
    for cls in util.walk_subclasses(target_hierarchy):
        if hasattr(cls, '_generate_cache_attrs') and hasattr(cls, '_traverse_internals'):
            cls._generate_cache_attrs()
            _copy_internals.generate_dispatch(cls, cls._traverse_internals, '_generated_copy_internals_traversal')
            _get_children.generate_dispatch(cls, cls._traverse_internals, '_generated_get_children_traversal')

class HasShallowCopy(HasTraverseInternals):
    """attribute-wide operations that are useful for classes that use
    __slots__ and therefore can't operate on their attributes in a dictionary.


    """
    __slots__ = ()
    if typing.TYPE_CHECKING:

        def _generated_shallow_copy_traversal(self, other: Self) -> None:
            if False:
                while True:
                    i = 10
            ...

        def _generated_shallow_from_dict_traversal(self, d: Dict[str, Any]) -> None:
            if False:
                i = 10
                return i + 15
            ...

        def _generated_shallow_to_dict_traversal(self) -> Dict[str, Any]:
            if False:
                for i in range(10):
                    print('nop')
            ...

    @classmethod
    def _generate_shallow_copy(cls, internal_dispatch: _TraverseInternalsType, method_name: str) -> Callable[[Self, Self], None]:
        if False:
            print('Hello World!')
        code = '\n'.join((f'    other.{attrname} = self.{attrname}' for (attrname, _) in internal_dispatch))
        meth_text = f'def {method_name}(self, other):\n{code}\n'
        return langhelpers._exec_code_in_env(meth_text, {}, method_name)

    @classmethod
    def _generate_shallow_to_dict(cls, internal_dispatch: _TraverseInternalsType, method_name: str) -> Callable[[Self], Dict[str, Any]]:
        if False:
            return 10
        code = ',\n'.join((f"    '{attrname}': self.{attrname}" for (attrname, _) in internal_dispatch))
        meth_text = f'def {method_name}(self):\n    return {{{code}}}\n'
        return langhelpers._exec_code_in_env(meth_text, {}, method_name)

    @classmethod
    def _generate_shallow_from_dict(cls, internal_dispatch: _TraverseInternalsType, method_name: str) -> Callable[[Self, Dict[str, Any]], None]:
        if False:
            i = 10
            return i + 15
        code = '\n'.join((f"    self.{attrname} = d['{attrname}']" for (attrname, _) in internal_dispatch))
        meth_text = f'def {method_name}(self, d):\n{code}\n'
        return langhelpers._exec_code_in_env(meth_text, {}, method_name)

    def _shallow_from_dict(self, d: Dict[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        cls = self.__class__
        shallow_from_dict: Callable[[HasShallowCopy, Dict[str, Any]], None]
        try:
            shallow_from_dict = cls.__dict__['_generated_shallow_from_dict_traversal']
        except KeyError:
            shallow_from_dict = self._generate_shallow_from_dict(cls._traverse_internals, '_generated_shallow_from_dict_traversal')
            cls._generated_shallow_from_dict_traversal = shallow_from_dict
        shallow_from_dict(self, d)

    def _shallow_to_dict(self) -> Dict[str, Any]:
        if False:
            return 10
        cls = self.__class__
        shallow_to_dict: Callable[[HasShallowCopy], Dict[str, Any]]
        try:
            shallow_to_dict = cls.__dict__['_generated_shallow_to_dict_traversal']
        except KeyError:
            shallow_to_dict = self._generate_shallow_to_dict(cls._traverse_internals, '_generated_shallow_to_dict_traversal')
            cls._generated_shallow_to_dict_traversal = shallow_to_dict
        return shallow_to_dict(self)

    def _shallow_copy_to(self, other: Self) -> None:
        if False:
            i = 10
            return i + 15
        cls = self.__class__
        shallow_copy: Callable[[Self, Self], None]
        try:
            shallow_copy = cls.__dict__['_generated_shallow_copy_traversal']
        except KeyError:
            shallow_copy = self._generate_shallow_copy(cls._traverse_internals, '_generated_shallow_copy_traversal')
            cls._generated_shallow_copy_traversal = shallow_copy
        shallow_copy(self, other)

    def _clone(self, **kw: Any) -> Self:
        if False:
            i = 10
            return i + 15
        'Create a shallow copy'
        c = self.__class__.__new__(self.__class__)
        self._shallow_copy_to(c)
        return c

class GenerativeOnTraversal(HasShallowCopy):
    """Supplies Generative behavior but making use of traversals to shallow
    copy.

    .. seealso::

        :class:`sqlalchemy.sql.base.Generative`


    """
    __slots__ = ()

    def _generate(self) -> Self:
        if False:
            while True:
                i = 10
        cls = self.__class__
        s = cls.__new__(cls)
        self._shallow_copy_to(s)
        return s

def _clone(element, **kw):
    if False:
        for i in range(10):
            print('nop')
    return element._clone()

class HasCopyInternals(HasTraverseInternals):
    __slots__ = ()

    def _clone(self, **kw):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def _copy_internals(self, *, omit_attrs: Iterable[str]=(), **kw: Any) -> None:
        if False:
            while True:
                i = 10
        'Reassign internal elements to be clones of themselves.\n\n        Called during a copy-and-traverse operation on newly\n        shallow-copied elements to create a deep copy.\n\n        The given clone function should be used, which may be applying\n        additional transformations to the element (i.e. replacement\n        traversal, cloned traversal, annotations).\n\n        '
        try:
            traverse_internals = self._traverse_internals
        except AttributeError:
            return
        for (attrname, obj, meth) in _copy_internals.run_generated_dispatch(self, traverse_internals, '_generated_copy_internals_traversal'):
            if attrname in omit_attrs:
                continue
            if obj is not None:
                result = meth(attrname, self, obj, **kw)
                if result is not None:
                    setattr(self, attrname, result)

class _CopyInternalsTraversal(HasTraversalDispatch):
    """Generate a _copy_internals internal traversal dispatch for classes
    with a _traverse_internals collection."""

    def visit_clauseelement(self, attrname, parent, element, clone=_clone, **kw):
        if False:
            i = 10
            return i + 15
        return clone(element, **kw)

    def visit_clauseelement_list(self, attrname, parent, element, clone=_clone, **kw):
        if False:
            print('Hello World!')
        return [clone(clause, **kw) for clause in element]

    def visit_clauseelement_tuple(self, attrname, parent, element, clone=_clone, **kw):
        if False:
            i = 10
            return i + 15
        return tuple([clone(clause, **kw) for clause in element])

    def visit_executable_options(self, attrname, parent, element, clone=_clone, **kw):
        if False:
            for i in range(10):
                print('nop')
        return tuple([clone(clause, **kw) for clause in element])

    def visit_clauseelement_unordered_set(self, attrname, parent, element, clone=_clone, **kw):
        if False:
            i = 10
            return i + 15
        return {clone(clause, **kw) for clause in element}

    def visit_clauseelement_tuples(self, attrname, parent, element, clone=_clone, **kw):
        if False:
            return 10
        return [tuple((clone(tup_elem, **kw) for tup_elem in elem)) for elem in element]

    def visit_string_clauseelement_dict(self, attrname, parent, element, clone=_clone, **kw):
        if False:
            print('Hello World!')
        return {key: clone(value, **kw) for (key, value) in element.items()}

    def visit_setup_join_tuple(self, attrname, parent, element, clone=_clone, **kw):
        if False:
            while True:
                i = 10
        return tuple(((clone(target, **kw) if target is not None else None, clone(onclause, **kw) if onclause is not None else None, clone(from_, **kw) if from_ is not None else None, flags) for (target, onclause, from_, flags) in element))

    def visit_memoized_select_entities(self, attrname, parent, element, **kw):
        if False:
            while True:
                i = 10
        return self.visit_clauseelement_tuple(attrname, parent, element, **kw)

    def visit_dml_ordered_values(self, attrname, parent, element, clone=_clone, **kw):
        if False:
            while True:
                i = 10
        return [(clone(key, **kw) if hasattr(key, '__clause_element__') else key, clone(value, **kw)) for (key, value) in element]

    def visit_dml_values(self, attrname, parent, element, clone=_clone, **kw):
        if False:
            return 10
        return {clone(key, **kw) if hasattr(key, '__clause_element__') else key: clone(value, **kw) for (key, value) in element.items()}

    def visit_dml_multi_values(self, attrname, parent, element, clone=_clone, **kw):
        if False:
            i = 10
            return i + 15

        def copy(elem):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(elem, (list, tuple)):
                return [clone(value, **kw) if hasattr(value, '__clause_element__') else value for value in elem]
            elif isinstance(elem, dict):
                return {clone(key, **kw) if hasattr(key, '__clause_element__') else key: clone(value, **kw) if hasattr(value, '__clause_element__') else value for (key, value) in elem.items()}
            else:
                assert False
        return [[copy(sub_element) for sub_element in sequence] for sequence in element]

    def visit_propagate_attrs(self, attrname, parent, element, clone=_clone, **kw):
        if False:
            print('Hello World!')
        return element
_copy_internals = _CopyInternalsTraversal()

def _flatten_clauseelement(element):
    if False:
        return 10
    while hasattr(element, '__clause_element__') and (not getattr(element, 'is_clause_element', False)):
        element = element.__clause_element__()
    return element

class _GetChildrenTraversal(HasTraversalDispatch):
    """Generate a _children_traversal internal traversal dispatch for classes
    with a _traverse_internals collection."""

    def visit_has_cache_key(self, element, **kw):
        if False:
            print('Hello World!')
        return ()

    def visit_clauseelement(self, element, **kw):
        if False:
            while True:
                i = 10
        return (element,)

    def visit_clauseelement_list(self, element, **kw):
        if False:
            return 10
        return element

    def visit_clauseelement_tuple(self, element, **kw):
        if False:
            return 10
        return element

    def visit_clauseelement_tuples(self, element, **kw):
        if False:
            while True:
                i = 10
        return itertools.chain.from_iterable(element)

    def visit_fromclause_canonical_column_collection(self, element, **kw):
        if False:
            for i in range(10):
                print('nop')
        return ()

    def visit_string_clauseelement_dict(self, element, **kw):
        if False:
            i = 10
            return i + 15
        return element.values()

    def visit_fromclause_ordered_set(self, element, **kw):
        if False:
            print('Hello World!')
        return element

    def visit_clauseelement_unordered_set(self, element, **kw):
        if False:
            print('Hello World!')
        return element

    def visit_setup_join_tuple(self, element, **kw):
        if False:
            print('Hello World!')
        for (target, onclause, from_, flags) in element:
            if from_ is not None:
                yield from_
            if not isinstance(target, str):
                yield _flatten_clauseelement(target)
            if onclause is not None and (not isinstance(onclause, str)):
                yield _flatten_clauseelement(onclause)

    def visit_memoized_select_entities(self, element, **kw):
        if False:
            i = 10
            return i + 15
        return self.visit_clauseelement_tuple(element, **kw)

    def visit_dml_ordered_values(self, element, **kw):
        if False:
            return 10
        for (k, v) in element:
            if hasattr(k, '__clause_element__'):
                yield k
            yield v

    def visit_dml_values(self, element, **kw):
        if False:
            while True:
                i = 10
        expr_values = {k for k in element if hasattr(k, '__clause_element__')}
        str_values = expr_values.symmetric_difference(element)
        for k in sorted(str_values):
            yield element[k]
        for k in expr_values:
            yield k
            yield element[k]

    def visit_dml_multi_values(self, element, **kw):
        if False:
            i = 10
            return i + 15
        return ()

    def visit_propagate_attrs(self, element, **kw):
        if False:
            for i in range(10):
                print('nop')
        return ()
_get_children = _GetChildrenTraversal()

@util.preload_module('sqlalchemy.sql.elements')
def _resolve_name_for_compare(element, name, anon_map, **kw):
    if False:
        while True:
            i = 10
    if isinstance(name, util.preloaded.sql_elements._anonymous_label):
        name = name.apply_map(anon_map)
    return name

class TraversalComparatorStrategy(HasTraversalDispatch, util.MemoizedSlots):
    __slots__ = ('stack', 'cache', 'anon_map')

    def __init__(self):
        if False:
            return 10
        self.stack: Deque[Tuple[Optional[ExternallyTraversible], Optional[ExternallyTraversible]]] = deque()
        self.cache = set()

    def _memoized_attr_anon_map(self):
        if False:
            while True:
                i = 10
        return (anon_map(), anon_map())

    def compare(self, obj1: ExternallyTraversible, obj2: ExternallyTraversible, **kw: Any) -> bool:
        if False:
            while True:
                i = 10
        stack = self.stack
        cache = self.cache
        compare_annotations = kw.get('compare_annotations', False)
        stack.append((obj1, obj2))
        while stack:
            (left, right) = stack.popleft()
            if left is right:
                continue
            elif left is None or right is None:
                return False
            elif (left, right) in cache:
                continue
            cache.add((left, right))
            visit_name = left.__visit_name__
            if visit_name != right.__visit_name__:
                return False
            meth = getattr(self, 'compare_%s' % visit_name, None)
            if meth:
                attributes_compared = meth(left, right, **kw)
                if attributes_compared is COMPARE_FAILED:
                    return False
                elif attributes_compared is SKIP_TRAVERSE:
                    continue
            else:
                attributes_compared = ()
            for ((left_attrname, left_visit_sym), (right_attrname, right_visit_sym)) in zip_longest(left._traverse_internals, right._traverse_internals, fillvalue=(None, None)):
                if not compare_annotations and (left_attrname == '_annotations' or right_attrname == '_annotations'):
                    continue
                if left_attrname != right_attrname or left_visit_sym is not right_visit_sym:
                    return False
                elif left_attrname in attributes_compared:
                    continue
                assert left_visit_sym is not None
                assert left_attrname is not None
                assert right_attrname is not None
                dispatch = self.dispatch(left_visit_sym)
                assert dispatch is not None, f"{self.__class__} has no dispatch for '{self._dispatch_lookup[left_visit_sym]}'"
                left_child = operator.attrgetter(left_attrname)(left)
                right_child = operator.attrgetter(right_attrname)(right)
                if left_child is None:
                    if right_child is not None:
                        return False
                    else:
                        continue
                comparison = dispatch(left_attrname, left, left_child, right, right_child, **kw)
                if comparison is COMPARE_FAILED:
                    return False
        return True

    def compare_inner(self, obj1, obj2, **kw):
        if False:
            print('Hello World!')
        comparator = self.__class__()
        return comparator.compare(obj1, obj2, **kw)

    def visit_has_cache_key(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            for i in range(10):
                print('nop')
        if left._gen_cache_key(self.anon_map[0], []) != right._gen_cache_key(self.anon_map[1], []):
            return COMPARE_FAILED

    def visit_propagate_attrs(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            for i in range(10):
                print('nop')
        return self.compare_inner(left.get('plugin_subject', None), right.get('plugin_subject', None))

    def visit_has_cache_key_list(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            while True:
                i = 10
        for (l, r) in zip_longest(left, right, fillvalue=None):
            if l is None:
                if r is not None:
                    return COMPARE_FAILED
                else:
                    continue
            elif r is None:
                return COMPARE_FAILED
            if l._gen_cache_key(self.anon_map[0], []) != r._gen_cache_key(self.anon_map[1], []):
                return COMPARE_FAILED

    def visit_executable_options(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            while True:
                i = 10
        for (l, r) in zip_longest(left, right, fillvalue=None):
            if l is None:
                if r is not None:
                    return COMPARE_FAILED
                else:
                    continue
            elif r is None:
                return COMPARE_FAILED
            if (l._gen_cache_key(self.anon_map[0], []) if l._is_has_cache_key else l) != (r._gen_cache_key(self.anon_map[1], []) if r._is_has_cache_key else r):
                return COMPARE_FAILED

    def visit_clauseelement(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            while True:
                i = 10
        self.stack.append((left, right))

    def visit_fromclause_canonical_column_collection(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            i = 10
            return i + 15
        for (lcol, rcol) in zip_longest(left, right, fillvalue=None):
            self.stack.append((lcol, rcol))

    def visit_fromclause_derived_column_collection(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            i = 10
            return i + 15
        pass

    def visit_string_clauseelement_dict(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            while True:
                i = 10
        for (lstr, rstr) in zip_longest(sorted(left), sorted(right), fillvalue=None):
            if lstr != rstr:
                return COMPARE_FAILED
            self.stack.append((left[lstr], right[rstr]))

    def visit_clauseelement_tuples(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            print('Hello World!')
        for (ltup, rtup) in zip_longest(left, right, fillvalue=None):
            if ltup is None or rtup is None:
                return COMPARE_FAILED
            for (l, r) in zip_longest(ltup, rtup, fillvalue=None):
                self.stack.append((l, r))

    def visit_clauseelement_list(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            return 10
        for (l, r) in zip_longest(left, right, fillvalue=None):
            self.stack.append((l, r))

    def visit_clauseelement_tuple(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            for i in range(10):
                print('nop')
        for (l, r) in zip_longest(left, right, fillvalue=None):
            self.stack.append((l, r))

    def _compare_unordered_sequences(self, seq1, seq2, **kw):
        if False:
            while True:
                i = 10
        if seq1 is None:
            return seq2 is None
        completed: Set[object] = set()
        for clause in seq1:
            for other_clause in set(seq2).difference(completed):
                if self.compare_inner(clause, other_clause, **kw):
                    completed.add(other_clause)
                    break
        return len(completed) == len(seq1) == len(seq2)

    def visit_clauseelement_unordered_set(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            while True:
                i = 10
        return self._compare_unordered_sequences(left, right, **kw)

    def visit_fromclause_ordered_set(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            i = 10
            return i + 15
        for (l, r) in zip_longest(left, right, fillvalue=None):
            self.stack.append((l, r))

    def visit_string(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            i = 10
            return i + 15
        return left == right

    def visit_string_list(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            while True:
                i = 10
        return left == right

    def visit_string_multi_dict(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            while True:
                i = 10
        for (lk, rk) in zip_longest(sorted(left.keys()), sorted(right.keys()), fillvalue=(None, None)):
            if lk != rk:
                return COMPARE_FAILED
            (lv, rv) = (left[lk], right[rk])
            lhc = isinstance(left, HasCacheKey)
            rhc = isinstance(right, HasCacheKey)
            if lhc and rhc:
                if lv._gen_cache_key(self.anon_map[0], []) != rv._gen_cache_key(self.anon_map[1], []):
                    return COMPARE_FAILED
            elif lhc != rhc:
                return COMPARE_FAILED
            elif lv != rv:
                return COMPARE_FAILED

    def visit_multi(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            print('Hello World!')
        lhc = isinstance(left, HasCacheKey)
        rhc = isinstance(right, HasCacheKey)
        if lhc and rhc:
            if left._gen_cache_key(self.anon_map[0], []) != right._gen_cache_key(self.anon_map[1], []):
                return COMPARE_FAILED
        elif lhc != rhc:
            return COMPARE_FAILED
        else:
            return left == right

    def visit_anon_name(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            print('Hello World!')
        return _resolve_name_for_compare(left_parent, left, self.anon_map[0], **kw) == _resolve_name_for_compare(right_parent, right, self.anon_map[1], **kw)

    def visit_boolean(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            for i in range(10):
                print('nop')
        return left == right

    def visit_operator(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            while True:
                i = 10
        return left == right

    def visit_type(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            return 10
        return left._compare_type_affinity(right)

    def visit_plain_dict(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            while True:
                i = 10
        return left == right

    def visit_dialect_options(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            while True:
                i = 10
        return left == right

    def visit_annotations_key(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            print('Hello World!')
        if left and right:
            return left_parent._annotations_cache_key == right_parent._annotations_cache_key
        else:
            return left == right

    def visit_with_context_options(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            for i in range(10):
                print('nop')
        return tuple(((fn.__code__, c_key) for (fn, c_key) in left)) == tuple(((fn.__code__, c_key) for (fn, c_key) in right))

    def visit_plain_obj(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            while True:
                i = 10
        return left == right

    def visit_named_ddl_element(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            i = 10
            return i + 15
        if left is None:
            if right is not None:
                return COMPARE_FAILED
        return left.name == right.name

    def visit_prefix_sequence(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            while True:
                i = 10
        for ((l_clause, l_str), (r_clause, r_str)) in zip_longest(left, right, fillvalue=(None, None)):
            if l_str != r_str:
                return COMPARE_FAILED
            else:
                self.stack.append((l_clause, r_clause))

    def visit_setup_join_tuple(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            i = 10
            return i + 15
        for ((l_target, l_onclause, l_from, l_flags), (r_target, r_onclause, r_from, r_flags)) in zip_longest(left, right, fillvalue=(None, None, None, None)):
            if l_flags != r_flags:
                return COMPARE_FAILED
            self.stack.append((l_target, r_target))
            self.stack.append((l_onclause, r_onclause))
            self.stack.append((l_from, r_from))

    def visit_memoized_select_entities(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            print('Hello World!')
        return self.visit_clauseelement_tuple(attrname, left_parent, left, right_parent, right, **kw)

    def visit_table_hint_list(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            return 10
        left_keys = sorted(left, key=lambda elem: (elem[0].fullname, elem[1]))
        right_keys = sorted(right, key=lambda elem: (elem[0].fullname, elem[1]))
        for ((ltable, ldialect), (rtable, rdialect)) in zip_longest(left_keys, right_keys, fillvalue=(None, None)):
            if ldialect != rdialect:
                return COMPARE_FAILED
            elif left[ltable, ldialect] != right[rtable, rdialect]:
                return COMPARE_FAILED
            else:
                self.stack.append((ltable, rtable))

    def visit_statement_hint_list(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            for i in range(10):
                print('nop')
        return left == right

    def visit_unknown_structure(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def visit_dml_ordered_values(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            i = 10
            return i + 15
        for ((lk, lv), (rk, rv)) in zip_longest(left, right, fillvalue=(None, None)):
            if not self._compare_dml_values_or_ce(lk, rk, **kw):
                return COMPARE_FAILED

    def _compare_dml_values_or_ce(self, lv, rv, **kw):
        if False:
            for i in range(10):
                print('nop')
        lvce = hasattr(lv, '__clause_element__')
        rvce = hasattr(rv, '__clause_element__')
        if lvce != rvce:
            return False
        elif lvce and (not self.compare_inner(lv, rv, **kw)):
            return False
        elif not lvce and lv != rv:
            return False
        elif not self.compare_inner(lv, rv, **kw):
            return False
        return True

    def visit_dml_values(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            print('Hello World!')
        if left is None or right is None or len(left) != len(right):
            return COMPARE_FAILED
        if isinstance(left, collections_abc.Sequence):
            for (lv, rv) in zip(left, right):
                if not self._compare_dml_values_or_ce(lv, rv, **kw):
                    return COMPARE_FAILED
        elif isinstance(right, collections_abc.Sequence):
            return COMPARE_FAILED
        else:
            for ((lk, lv), (rk, rv)) in zip(left.items(), right.items()):
                if not self._compare_dml_values_or_ce(lk, rk, **kw):
                    return COMPARE_FAILED
                if not self._compare_dml_values_or_ce(lv, rv, **kw):
                    return COMPARE_FAILED

    def visit_dml_multi_values(self, attrname, left_parent, left, right_parent, right, **kw):
        if False:
            i = 10
            return i + 15
        for (lseq, rseq) in zip_longest(left, right, fillvalue=None):
            if lseq is None or rseq is None:
                return COMPARE_FAILED
            for (ld, rd) in zip_longest(lseq, rseq, fillvalue=None):
                if self.visit_dml_values(attrname, left_parent, ld, right_parent, rd, **kw) is COMPARE_FAILED:
                    return COMPARE_FAILED

    def compare_expression_clauselist(self, left, right, **kw):
        if False:
            i = 10
            return i + 15
        if left.operator is right.operator:
            if operators.is_associative(left.operator):
                if self._compare_unordered_sequences(left.clauses, right.clauses, **kw):
                    return ['operator', 'clauses']
                else:
                    return COMPARE_FAILED
            else:
                return ['operator']
        else:
            return COMPARE_FAILED

    def compare_clauselist(self, left, right, **kw):
        if False:
            while True:
                i = 10
        return self.compare_expression_clauselist(left, right, **kw)

    def compare_binary(self, left, right, **kw):
        if False:
            while True:
                i = 10
        if left.operator == right.operator:
            if operators.is_commutative(left.operator):
                if self.compare_inner(left.left, right.left, **kw) and self.compare_inner(left.right, right.right, **kw) or (self.compare_inner(left.left, right.right, **kw) and self.compare_inner(left.right, right.left, **kw)):
                    return ['operator', 'negate', 'left', 'right']
                else:
                    return COMPARE_FAILED
            else:
                return ['operator', 'negate']
        else:
            return COMPARE_FAILED

    def compare_bindparam(self, left, right, **kw):
        if False:
            return 10
        compare_keys = kw.pop('compare_keys', True)
        compare_values = kw.pop('compare_values', True)
        if compare_values:
            omit = []
        else:
            omit = ['callable', 'value']
        if not compare_keys:
            omit.append('key')
        return omit

class ColIdentityComparatorStrategy(TraversalComparatorStrategy):

    def compare_column_element(self, left, right, use_proxies=True, equivalents=(), **kw):
        if False:
            for i in range(10):
                print('nop')
        'Compare ColumnElements using proxies and equivalent collections.\n\n        This is a comparison strategy specific to the ORM.\n        '
        to_compare = (right,)
        if equivalents and right in equivalents:
            to_compare = equivalents[right].union(to_compare)
        for oth in to_compare:
            if use_proxies and left.shares_lineage(oth):
                return SKIP_TRAVERSE
            elif hash(left) == hash(right):
                return SKIP_TRAVERSE
        else:
            return COMPARE_FAILED

    def compare_column(self, left, right, **kw):
        if False:
            print('Hello World!')
        return self.compare_column_element(left, right, **kw)

    def compare_label(self, left, right, **kw):
        if False:
            print('Hello World!')
        return self.compare_column_element(left, right, **kw)

    def compare_table(self, left, right, **kw):
        if False:
            i = 10
            return i + 15
        return SKIP_TRAVERSE if left is right else COMPARE_FAILED