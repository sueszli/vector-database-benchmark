import logging
from typing import Generic, Hashable, List, Set, TypeVar
import attr
logger = logging.getLogger(__name__)
T = TypeVar('T', bound=Hashable)

@attr.s(slots=True, frozen=True, auto_attribs=True)
class _Entry(Generic[T]):
    end_key: int
    elements: Set[T] = attr.Factory(set)

class WheelTimer(Generic[T]):
    """Stores arbitrary objects that will be returned after their timers have
    expired.
    """

    def __init__(self, bucket_size: int=5000) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            bucket_size: Size of buckets in ms. Corresponds roughly to the\n                accuracy of the timer.\n        '
        self.bucket_size: int = bucket_size
        self.entries: List[_Entry[T]] = []
        self.current_tick: int = 0

    def insert(self, now: int, obj: T, then: int) -> None:
        if False:
            return 10
        'Inserts object into timer.\n\n        Args:\n            now: Current time in msec\n            obj: Object to be inserted\n            then: When to return the object strictly after.\n        '
        then_key = int(then / self.bucket_size) + 1
        now_key = int(now / self.bucket_size)
        if self.entries:
            min_key = self.entries[0].end_key
            max_key = self.entries[-1].end_key
            if min_key < now_key - 10:
                logger.warning("Inserting into a wheel timer that hasn't been read from recently. Item: %s", obj)
            if then_key <= max_key:
                self.entries[max(min_key, then_key) - min_key].elements.add(obj)
                return
        next_key = now_key + 1
        if self.entries:
            last_key = self.entries[-1].end_key
        else:
            last_key = next_key
        then_key = max(last_key, then_key)
        self.entries.extend((_Entry(key) for key in range(last_key, then_key + 1)))
        self.entries[-1].elements.add(obj)

    def fetch(self, now: int) -> List[T]:
        if False:
            return 10
        'Fetch any objects that have timed out\n\n        Args:\n            now: Current time in msec\n\n        Returns:\n            List of objects that have timed out\n        '
        now_key = int(now / self.bucket_size)
        ret: List[T] = []
        while self.entries and self.entries[0].end_key <= now_key:
            ret.extend(self.entries.pop(0).elements)
        return ret

    def __len__(self) -> int:
        if False:
            while True:
                i = 10
        return sum((len(entry.elements) for entry in self.entries))