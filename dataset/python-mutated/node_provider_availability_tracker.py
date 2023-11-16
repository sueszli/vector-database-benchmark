import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
from ray.autoscaler._private.constants import AUTOSCALER_NODE_AVAILABILITY_MAX_STALENESS_S
from ray.autoscaler.node_launch_exception import NodeLaunchException

@dataclass
class UnavailableNodeInformation:
    category: str
    description: str

@dataclass
class NodeAvailabilityRecord:
    node_type: str
    is_available: bool
    last_checked_timestamp: float
    unavailable_node_information: Optional[UnavailableNodeInformation]

@dataclass
class NodeAvailabilitySummary:
    node_availabilities: Dict[str, NodeAvailabilityRecord]

    @classmethod
    def from_fields(cls, **fields) -> Optional['NodeAvailabilitySummary']:
        if False:
            i = 10
            return i + 15
        "Implement marshalling from nested fields. pydantic isn't a core dependency\n        so we're implementing this by hand instead."
        parsed = {}
        node_availabilites_dict = fields.get('node_availabilities', {})
        for (node_type, node_availability_record_dict) in node_availabilites_dict.items():
            unavailable_information_dict = node_availability_record_dict.pop('unavailable_node_information', None)
            unavaiable_information = None
            if unavailable_information_dict is not None:
                unavaiable_information = UnavailableNodeInformation(**unavailable_information_dict)
            parsed[node_type] = NodeAvailabilityRecord(unavailable_node_information=unavaiable_information, **node_availability_record_dict)
        return NodeAvailabilitySummary(node_availabilities=parsed)

    def __eq__(self, other: 'NodeAvailabilitySummary'):
        if False:
            return 10
        return self.node_availabilities == other.node_availabilities

    def __bool__(self) -> bool:
        if False:
            while True:
                i = 10
        return bool(self.node_availabilities)

class NodeProviderAvailabilityTracker:
    """A thread safe, TTL cache of node provider availability. We don't use
    cachetools.TTLCache because it always sets the expiration time relative to
    insertion time, but in our case, we want entries to expire relative to when
    the node creation was attempted (and entries aren't necessarily added in
    order). We want the entries to expire because the information grows stale
    over time.
    """

    def __init__(self, timer: Callable[[], float]=time.time, ttl: float=AUTOSCALER_NODE_AVAILABILITY_MAX_STALENESS_S):
        if False:
            for i in range(10):
                print('nop')
        'A cache that tracks the availability of nodes and throw away\n        entries which have grown too stale.\n\n        Args:\n          timer: A function that returns the current time in seconds.\n          ttl: The ttl from the insertion timestamp of an entry.\n        '
        self.timer = timer
        self.ttl = ttl
        self.store: Dict[str, Tuple[float, NodeAvailabilityRecord]] = {}
        self.lock = threading.RLock()

    def _update_node_availability_requires_lock(self, node_type: str, timestamp: int, node_launch_exception: Optional[NodeLaunchException]) -> None:
        if False:
            i = 10
            return i + 15
        if node_launch_exception is None:
            record = NodeAvailabilityRecord(node_type=node_type, is_available=True, last_checked_timestamp=timestamp, unavailable_node_information=None)
        else:
            info = UnavailableNodeInformation(category=node_launch_exception.category, description=node_launch_exception.description)
            record = NodeAvailabilityRecord(node_type=node_type, is_available=False, last_checked_timestamp=timestamp, unavailable_node_information=info)
        expiration_time = timestamp + self.ttl
        self.store[node_type] = (expiration_time, record)
        self._remove_old_entries()

    def update_node_availability(self, node_type: str, timestamp: int, node_launch_exception: Optional[NodeLaunchException]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Update the availability and details of a single ndoe type.\n\n        Args:\n          node_type: The node type.\n          timestamp: The timestamp that this information is accurate as of.\n          node_launch_exception: Details about why the node launch failed. If\n            empty, the node type will be considered available.'
        with self.lock:
            self._update_node_availability_requires_lock(node_type, timestamp, node_launch_exception)

    def summary(self) -> NodeAvailabilitySummary:
        if False:
            while True:
                i = 10
        '\n        Returns a summary of node availabilities and their staleness.\n\n        Returns\n            A summary of node availabilities and their staleness.\n        '
        with self.lock:
            self._remove_old_entries()
            return NodeAvailabilitySummary({node_type: record for (node_type, (_, record)) in self.store.items()})

    def _remove_old_entries(self):
        if False:
            while True:
                i = 10
        'Remove any expired entries from the cache.'
        cur_time = self.timer()
        with self.lock:
            for (key, (expiration_time, _)) in list(self.store.items()):
                if expiration_time < cur_time:
                    del self.store[key]