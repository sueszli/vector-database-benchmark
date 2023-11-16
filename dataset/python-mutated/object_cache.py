from collections import defaultdict, Counter
from typing import Dict, Generator, List, Optional, TypeVar
T = TypeVar('T')
U = TypeVar('U')

class _ObjectCache:
    """Cache up to some maximum count given a grouping key.

    This object cache can e.g. be used to cache Ray Tune trainable actors
    given their resource requirements (reuse_actors=True).

    If the max number of cached objects for a grouping key is reached,
    no more objects for this group will be cached.

    However, if `may_keep_one=True`, one object (globally across all grouping
    keys) may be cached, even if the max number of objects is 0. This is to
    allow to cache an object if the max number of objects of this key
    will increase shortly after (as is the case e.g. in the Ray Tune control
    loop).

    Args:
        may_keep_one: If True, one object (globally) may be cached if no desired
            maximum objects are defined.

    """

    def __init__(self, may_keep_one: bool=True):
        if False:
            for i in range(10):
                print('nop')
        self._num_cached_objects: int = 0
        self._cached_objects: Dict[T, List[U]] = defaultdict(list)
        self._max_num_objects: Counter[T] = Counter()
        self._may_keep_one = may_keep_one

    @property
    def num_cached_objects(self):
        if False:
            for i in range(10):
                print('nop')
        return self._num_cached_objects

    @property
    def total_max_objects(self):
        if False:
            for i in range(10):
                print('nop')
        return sum(self._max_num_objects.values())

    def increase_max(self, key: T, by: int=1) -> None:
        if False:
            print('Hello World!')
        'Increase number of max objects for this key.\n\n        Args:\n            key: Group key.\n            by: Decrease by this amount.\n        '
        self._max_num_objects[key] += by

    def decrease_max(self, key: T, by: int=1) -> None:
        if False:
            print('Hello World!')
        'Decrease number of max objects for this key.\n\n        Args:\n            key: Group key.\n            by: Decrease by this amount.\n        '
        self._max_num_objects[key] -= by

    def has_cached_object(self, key: T) -> bool:
        if False:
            print('Hello World!')
        'Return True if at least one cached object exists for this key.\n\n        Args:\n            key: Group key.\n\n        Returns:\n            True if at least one cached object exists for this key.\n        '
        return bool(self._cached_objects[key])

    def cache_object(self, key: T, obj: U) -> bool:
        if False:
            i = 10
            return i + 15
        'Cache object for a given key.\n\n        This will put the object into a cache, assuming the number\n        of cached objects for this key is less than the number of\n        max objects for this key.\n\n        An exception is made if `max_keep_one=True` and no other\n        objects are cached globally. In that case, the object can\n        still be cached.\n\n        Args:\n            key: Group key.\n            obj: Object to cache.\n\n        Returns:\n            True if the object has been cached. False otherwise.\n\n        '
        if len(self._cached_objects[key]) >= self._max_num_objects[key]:
            if not self._may_keep_one:
                return False
            if self._num_cached_objects > 0:
                return False
            if any((v for v in self._max_num_objects.values())):
                return False
        self._cached_objects[key].append(obj)
        self._num_cached_objects += 1
        return True

    def pop_cached_object(self, key: T) -> Optional[U]:
        if False:
            while True:
                i = 10
        'Get one cached object for a key.\n\n        This will remove the object from the cache.\n\n        Args:\n            key: Group key.\n\n        Returns:\n            Cached object.\n        '
        if not self.has_cached_object(key):
            return None
        self._num_cached_objects -= 1
        return self._cached_objects[key].pop(0)

    def flush_cached_objects(self, force_all: bool=False) -> Generator[U, None, None]:
        if False:
            while True:
                i = 10
        'Return a generator over cached objects evicted from the cache.\n\n        This method yields all cached objects that should be evicted from the\n        cache for cleanup by the caller.\n\n        If the number of max objects is lower than the number of\n        cached objects for a given key, objects are evicted until\n        the numbers are equal.\n\n        If `max_keep_one=True` (and ``force_all=False``), one cached object\n        may be retained.\n\n        Objects are evicted FIFO.\n\n        If ``force_all=True``, all objects are evicted.\n\n        Args:\n            force_all: If True, all objects are flushed. This takes precedence\n                over ``keep_one``.\n\n        Yields:\n            Evicted objects to be cleaned up by caller.\n\n        '
        keep_one = self._may_keep_one and (not force_all)
        for (key, objs) in self._cached_objects.items():
            max_cached = self._max_num_objects[key] if not force_all else 0
            if self._num_cached_objects == 1 and keep_one and (not any((v for v in self._max_num_objects.values()))):
                break
            while len(objs) > max_cached:
                self._num_cached_objects -= 1
                yield objs.pop(0)