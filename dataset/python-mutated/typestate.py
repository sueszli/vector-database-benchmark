"""
A shared state for all TypeInfos that holds global cache and dependency information,
and potentially other mutable TypeInfo state. This module contains mutable global state.
"""
from __future__ import annotations
from typing import Dict, Final, Set, Tuple
from typing_extensions import TypeAlias as _TypeAlias
from mypy.nodes import TypeInfo
from mypy.server.trigger import make_trigger
from mypy.types import Instance, Type, TypeVarId, get_proper_type
MAX_NEGATIVE_CACHE_TYPES: Final = 1000
MAX_NEGATIVE_CACHE_ENTRIES: Final = 10000
SubtypeRelationship: _TypeAlias = Tuple[Instance, Instance]
SubtypeKind: _TypeAlias = Tuple[bool, ...]
SubtypeCache: _TypeAlias = Dict[TypeInfo, Dict[SubtypeKind, Set[SubtypeRelationship]]]

class TypeState:
    """This class provides subtype caching to improve performance of subtype checks.
    It also holds protocol fine grained dependencies.

    Note: to avoid leaking global state, 'reset_all_subtype_caches()' should be called
    after a build has finished and after a daemon shutdown. This subtype cache only exists for
    performance reasons, resetting subtype caches for a class has no semantic effect.
    The protocol dependencies however are only stored here, and shouldn't be deleted unless
    not needed any more (e.g. during daemon shutdown).
    """
    _subtype_caches: Final[SubtypeCache]
    _negative_subtype_caches: Final[SubtypeCache]
    proto_deps: dict[str, set[str]] | None
    _attempted_protocols: Final[dict[str, set[str]]]
    _checked_against_members: Final[dict[str, set[str]]]
    _rechecked_types: Final[set[TypeInfo]]
    _assuming: Final[list[tuple[Type, Type]]]
    _assuming_proper: Final[list[tuple[Type, Type]]]
    inferring: Final[list[tuple[Type, Type]]]
    infer_unions: bool
    infer_polymorphic: bool

    def __init__(self) -> None:
        if False:
            return 10
        self._subtype_caches = {}
        self._negative_subtype_caches = {}
        self.proto_deps = {}
        self._attempted_protocols = {}
        self._checked_against_members = {}
        self._rechecked_types = set()
        self._assuming = []
        self._assuming_proper = []
        self.inferring = []
        self.infer_unions = False
        self.infer_polymorphic = False

    def is_assumed_subtype(self, left: Type, right: Type) -> bool:
        if False:
            return 10
        for (l, r) in reversed(self._assuming):
            if get_proper_type(l) == get_proper_type(left) and get_proper_type(r) == get_proper_type(right):
                return True
        return False

    def is_assumed_proper_subtype(self, left: Type, right: Type) -> bool:
        if False:
            return 10
        for (l, r) in reversed(self._assuming_proper):
            if get_proper_type(l) == get_proper_type(left) and get_proper_type(r) == get_proper_type(right):
                return True
        return False

    def get_assumptions(self, is_proper: bool) -> list[tuple[Type, Type]]:
        if False:
            while True:
                i = 10
        if is_proper:
            return self._assuming_proper
        return self._assuming

    def reset_all_subtype_caches(self) -> None:
        if False:
            i = 10
            return i + 15
        'Completely reset all known subtype caches.'
        self._subtype_caches.clear()
        self._negative_subtype_caches.clear()

    def reset_subtype_caches_for(self, info: TypeInfo) -> None:
        if False:
            return 10
        'Reset subtype caches (if any) for a given supertype TypeInfo.'
        if info in self._subtype_caches:
            self._subtype_caches[info].clear()
        if info in self._negative_subtype_caches:
            self._negative_subtype_caches[info].clear()

    def reset_all_subtype_caches_for(self, info: TypeInfo) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Reset subtype caches (if any) for a given supertype TypeInfo and its MRO.'
        for item in info.mro:
            self.reset_subtype_caches_for(item)

    def is_cached_subtype_check(self, kind: SubtypeKind, left: Instance, right: Instance) -> bool:
        if False:
            i = 10
            return i + 15
        if left.last_known_value is not None or right.last_known_value is not None:
            return False
        info = right.type
        cache = self._subtype_caches.get(info)
        if cache is None:
            return False
        subcache = cache.get(kind)
        if subcache is None:
            return False
        return (left, right) in subcache

    def is_cached_negative_subtype_check(self, kind: SubtypeKind, left: Instance, right: Instance) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if left.last_known_value is not None or right.last_known_value is not None:
            return False
        info = right.type
        cache = self._negative_subtype_caches.get(info)
        if cache is None:
            return False
        subcache = cache.get(kind)
        if subcache is None:
            return False
        return (left, right) in subcache

    def record_subtype_cache_entry(self, kind: SubtypeKind, left: Instance, right: Instance) -> None:
        if False:
            for i in range(10):
                print('nop')
        if left.last_known_value is not None or right.last_known_value is not None:
            return
        cache = self._subtype_caches.setdefault(right.type, dict())
        cache.setdefault(kind, set()).add((left, right))

    def record_negative_subtype_cache_entry(self, kind: SubtypeKind, left: Instance, right: Instance) -> None:
        if False:
            i = 10
            return i + 15
        if left.last_known_value is not None or right.last_known_value is not None:
            return
        if len(self._negative_subtype_caches) > MAX_NEGATIVE_CACHE_TYPES:
            self._negative_subtype_caches.clear()
        cache = self._negative_subtype_caches.setdefault(right.type, dict())
        subcache = cache.setdefault(kind, set())
        if len(subcache) > MAX_NEGATIVE_CACHE_ENTRIES:
            subcache.clear()
        cache.setdefault(kind, set()).add((left, right))

    def reset_protocol_deps(self) -> None:
        if False:
            while True:
                i = 10
        'Reset dependencies after a full run or before a daemon shutdown.'
        self.proto_deps = {}
        self._attempted_protocols.clear()
        self._checked_against_members.clear()
        self._rechecked_types.clear()

    def record_protocol_subtype_check(self, left_type: TypeInfo, right_type: TypeInfo) -> None:
        if False:
            while True:
                i = 10
        assert right_type.is_protocol
        self._rechecked_types.add(left_type)
        self._attempted_protocols.setdefault(left_type.fullname, set()).add(right_type.fullname)
        self._checked_against_members.setdefault(left_type.fullname, set()).update(right_type.protocol_members)

    def _snapshot_protocol_deps(self) -> dict[str, set[str]]:
        if False:
            while True:
                i = 10
        "Collect protocol attribute dependencies found so far from registered subtype checks.\n\n        There are three kinds of protocol dependencies. For example, after a subtype check:\n\n            x: Proto = C()\n\n        the following dependencies will be generated:\n            1. ..., <SuperProto[wildcard]>, <Proto[wildcard]> -> <Proto>\n            2. ..., <B.attr>, <C.attr> -> <C> [for every attr in Proto members]\n            3. <C> -> Proto  # this one to invalidate the subtype cache\n\n        The first kind is generated immediately per-module in deps.py (see also an example there\n        for motivation why it is needed). While two other kinds are generated here after all\n        modules are type checked and we have recorded all the subtype checks. To understand these\n        two kinds, consider a simple example:\n\n            class A:\n                def __iter__(self) -> Iterator[int]:\n                    ...\n\n            it: Iterable[int] = A()\n\n        We add <a.A.__iter__> -> <a.A> to invalidate the assignment (module target in this case),\n        whenever the signature of a.A.__iter__ changes. We also add <a.A> -> typing.Iterable,\n        to invalidate the subtype caches of the latter. (Note that the same logic applies to\n        proper subtype checks, and calculating meets and joins, if this involves calling\n        'subtypes.is_protocol_implementation').\n        "
        deps: dict[str, set[str]] = {}
        for info in self._rechecked_types:
            for attr in self._checked_against_members[info.fullname]:
                for base_info in info.mro[:-1]:
                    trigger = make_trigger(f'{base_info.fullname}.{attr}')
                    if 'typing' in trigger or 'builtins' in trigger:
                        continue
                    deps.setdefault(trigger, set()).add(make_trigger(info.fullname))
            for proto in self._attempted_protocols[info.fullname]:
                trigger = make_trigger(info.fullname)
                if 'typing' in trigger or 'builtins' in trigger:
                    continue
                deps.setdefault(trigger, set()).add(proto)
        return deps

    def update_protocol_deps(self, second_map: dict[str, set[str]] | None=None) -> None:
        if False:
            print('Hello World!')
        'Update global protocol dependency map.\n\n        We update the global map incrementally, using a snapshot only from recently\n        type checked types. If second_map is given, update it as well. This is currently used\n        by FineGrainedBuildManager that maintains normal (non-protocol) dependencies.\n        '
        assert self.proto_deps is not None, 'This should not be called after failed cache load'
        new_deps = self._snapshot_protocol_deps()
        for (trigger, targets) in new_deps.items():
            self.proto_deps.setdefault(trigger, set()).update(targets)
        if second_map is not None:
            for (trigger, targets) in new_deps.items():
                second_map.setdefault(trigger, set()).update(targets)
        self._rechecked_types.clear()
        self._attempted_protocols.clear()
        self._checked_against_members.clear()

    def add_all_protocol_deps(self, deps: dict[str, set[str]]) -> None:
        if False:
            return 10
        'Add all known protocol dependencies to deps.\n\n        This is used by tests and debug output, and also when collecting\n        all collected or loaded dependencies as part of build.\n        '
        self.update_protocol_deps()
        if self.proto_deps is not None:
            for (trigger, targets) in self.proto_deps.items():
                deps.setdefault(trigger, set()).update(targets)
type_state: Final = TypeState()

def reset_global_state() -> None:
    if False:
        print('Hello World!')
    'Reset most existing global state.\n\n    Currently most of it is in this module. Few exceptions are strict optional status\n    and functools.lru_cache.\n    '
    type_state.reset_all_subtype_caches()
    type_state.reset_protocol_deps()
    TypeVarId.next_raw_id = 1