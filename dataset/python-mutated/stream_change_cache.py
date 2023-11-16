import logging
import math
from typing import Collection, Dict, FrozenSet, List, Mapping, Optional, Set, Union
import attr
from sortedcontainers import SortedDict
from synapse.util import caches
logger = logging.getLogger(__name__)
EntityType = str

@attr.s(auto_attribs=True, frozen=True, slots=True)
class AllEntitiesChangedResult:
    """Return type of `get_all_entities_changed`.

    Callers must check that there was a cache hit, via `result.hit`, before
    using the entities in `result.entities`.

    This specifically does *not* implement helpers such as `__bool__` to ensure
    that callers do the correct checks.
    """
    _entities: Optional[List[EntityType]]

    @property
    def hit(self) -> bool:
        if False:
            while True:
                i = 10
        return self._entities is not None

    @property
    def entities(self) -> List[EntityType]:
        if False:
            for i in range(10):
                print('nop')
        assert self._entities is not None
        return self._entities

class StreamChangeCache:
    """
    Keeps track of the stream positions of the latest change in a set of entities.

    The entity will is typically a room ID or user ID, but can be any string.

    Can be queried for whether a specific entity has changed after a stream position
    or for a list of changed entities after a stream position. See the individual
    methods for more information.

    Only tracks to a maximum cache size, any position earlier than the earliest
    known stream position must be treated as unknown.
    """

    def __init__(self, name: str, current_stream_pos: int, max_size: int=10000, prefilled_cache: Optional[Mapping[EntityType, int]]=None) -> None:
        if False:
            print('Hello World!')
        self._original_max_size: int = max_size
        self._max_size = math.floor(max_size)
        self._cache: SortedDict[int, Set[EntityType]] = SortedDict()
        self._entity_to_key: Dict[EntityType, int] = {}
        self._earliest_known_stream_pos = current_stream_pos
        self.name = name
        self.metrics = caches.register_cache('cache', self.name, self._cache, resize_callback=self.set_cache_factor)
        if prefilled_cache:
            for (entity, stream_pos) in prefilled_cache.items():
                self.entity_has_changed(entity, stream_pos)

    def set_cache_factor(self, factor: float) -> bool:
        if False:
            return 10
        '\n        Set the cache factor for this individual cache.\n\n        This will trigger a resize if it changes, which may require evicting\n        items from the cache.\n\n        Returns:\n            Whether the cache changed size or not.\n        '
        new_size = math.floor(self._original_max_size * factor)
        if new_size != self._max_size:
            self.max_size = new_size
            self._evict()
            return True
        return False

    def has_entity_changed(self, entity: EntityType, stream_pos: int) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Returns True if the entity may have been updated after stream_pos.\n\n        Args:\n            entity: The entity to check for changes.\n            stream_pos: The stream position to check for changes after.\n\n        Return:\n            True if the entity may have been updated, this happens if:\n                * The given stream position is at or earlier than the earliest\n                  known stream position.\n                * The given stream position is earlier than the latest change for\n                  the entity.\n\n            False otherwise:\n                * The entity is unknown.\n                * The given stream position is at or later than the latest change\n                  for the entity.\n        '
        assert isinstance(stream_pos, int)
        if stream_pos <= self._earliest_known_stream_pos:
            self.metrics.inc_misses()
            return True
        latest_entity_change_pos = self._entity_to_key.get(entity, None)
        if latest_entity_change_pos is None:
            self.metrics.inc_hits()
            return False
        if stream_pos < latest_entity_change_pos:
            self.metrics.inc_misses()
            return True
        self.metrics.inc_hits()
        return False

    def get_entities_changed(self, entities: Collection[EntityType], stream_pos: int) -> Union[Set[EntityType], FrozenSet[EntityType]]:
        if False:
            return 10
        '\n        Returns the subset of the given entities that have had changes after the given position.\n\n        Entities unknown to the cache will be returned.\n\n        If the position is too old it will just return the given list.\n\n        Args:\n            entities: Entities to check for changes.\n            stream_pos: The stream position to check for changes after.\n\n        Return:\n            A subset of entities which have changed after the given stream position.\n\n            This will be all entities if the given stream position is at or earlier\n            than the earliest known stream position.\n        '
        cache_result = self.get_all_entities_changed(stream_pos)
        if cache_result.hit:
            if isinstance(entities, (set, frozenset)):
                result = entities.intersection(cache_result.entities)
            elif len(cache_result.entities) < len(entities):
                result = set(cache_result.entities).intersection(entities)
            else:
                result = set(entities).intersection(cache_result.entities)
            self.metrics.inc_hits()
        else:
            result = set(entities)
            self.metrics.inc_misses()
        return result

    def has_any_entity_changed(self, stream_pos: int) -> bool:
        if False:
            while True:
                i = 10
        '\n        Returns true if any entity has changed after the given stream position.\n\n        Args:\n            stream_pos: The stream position to check for changes after.\n\n        Return:\n            True if any entity has changed after the given stream position or\n            if the given stream position is at or earlier than the earliest\n            known stream position.\n\n            False otherwise.\n        '
        assert isinstance(stream_pos, int)
        if stream_pos <= self._earliest_known_stream_pos:
            self.metrics.inc_misses()
            return True
        if not self._cache:
            self.metrics.inc_misses()
            return False
        self.metrics.inc_hits()
        return stream_pos < self._cache.peekitem()[0]

    def get_all_entities_changed(self, stream_pos: int) -> AllEntitiesChangedResult:
        if False:
            return 10
        '\n        Returns all entities that have had changes after the given position.\n\n        If the stream change cache does not go far enough back, i.e. the\n        position is too old, it will return None.\n\n        Returns the entities in the order that they were changed.\n\n        Args:\n            stream_pos: The stream position to check for changes after.\n\n        Return:\n            A class indicating if we have the requested data cached, and if so\n            includes the entities in the order they were changed.\n        '
        assert isinstance(stream_pos, int)
        if stream_pos <= self._earliest_known_stream_pos:
            return AllEntitiesChangedResult(None)
        changed_entities: List[EntityType] = []
        for k in self._cache.islice(start=self._cache.bisect_right(stream_pos)):
            changed_entities.extend(self._cache[k])
        return AllEntitiesChangedResult(changed_entities)

    def entity_has_changed(self, entity: EntityType, stream_pos: int) -> None:
        if False:
            print('Hello World!')
        '\n        Informs the cache that the entity has been changed at the given position.\n\n        Args:\n            entity: The entity to mark as changed.\n            stream_pos: The stream position to update the entity to.\n        '
        assert isinstance(stream_pos, int)
        if stream_pos <= self._earliest_known_stream_pos:
            return
        old_pos = self._entity_to_key.get(entity, None)
        if old_pos is not None:
            if old_pos >= stream_pos:
                return
            e = self._cache[old_pos]
            e.remove(entity)
            if not e:
                del self._cache[old_pos]
        e1 = self._cache.get(stream_pos)
        if e1 is None:
            e1 = self._cache[stream_pos] = set()
        e1.add(entity)
        self._entity_to_key[entity] = stream_pos
        self._evict()

    def _evict(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Ensure the cache has not exceeded the maximum size.\n\n        Evicts entries until it is at the maximum size.\n        '
        while len(self._cache) > self._max_size:
            (k, r) = self._cache.popitem(0)
            self._earliest_known_stream_pos = max(k, self._earliest_known_stream_pos)
            for entity in r:
                self._entity_to_key.pop(entity, None)

    def get_max_pos_of_last_change(self, entity: EntityType) -> int:
        if False:
            while True:
                i = 10
        'Returns an upper bound of the stream id of the last change to an\n        entity.\n\n        Args:\n            entity: The entity to check.\n\n        Return:\n            The stream position of the latest change for the given entity or\n            the earliest known stream position if the entitiy is unknown.\n        '
        return self._entity_to_key.get(entity, self._earliest_known_stream_pos)