from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, MutableMapping, Optional

class ConcurrencyCompatibleStateType(Enum):
    date_range = 'date-range'

class ConcurrentStreamStateConverter(ABC):
    START_KEY = 'start'
    END_KEY = 'end'

    def get_concurrent_stream_state(self, state: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        if False:
            while True:
                i = 10
        if self.is_state_message_compatible(state):
            return state
        return self.convert_from_sequential_state(state)

    @staticmethod
    def is_state_message_compatible(state: MutableMapping[str, Any]) -> bool:
        if False:
            print('Hello World!')
        return state.get('state_type') in [t.value for t in ConcurrencyCompatibleStateType]

    @abstractmethod
    def convert_from_sequential_state(self, stream_state: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        if False:
            while True:
                i = 10
        '\n        Convert the state message to the format required by the ThreadBasedConcurrentStream.\n\n        e.g.\n        {\n            "state_type": ConcurrencyCompatibleStateType.date_range.value,\n            "metadata": { … },\n            "slices": [\n                {starts: 0, end: 1617030403, finished_processing: true}]\n        }\n        '
        ...

    @abstractmethod
    def convert_to_sequential_state(self, stream_state: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert the state message from the concurrency-compatible format to the stream\'s original format.\n\n        e.g.\n        { "created": 1617030403 }\n        '
        ...

    def _get_latest_complete_time(self, slices: List[MutableMapping[str, Any]]) -> Optional[Any]:
        if False:
            print('Hello World!')
        '\n        Get the latest time before which all records have been processed.\n        '
        if slices:
            first_interval = self.merge_intervals(slices)[0][self.END_KEY]
            return first_interval
        else:
            return None

    @staticmethod
    @abstractmethod
    def increment(timestamp: Any) -> Any:
        if False:
            i = 10
            return i + 15
        '\n        Increment a timestamp by a single unit.\n        '
        ...

    @classmethod
    def merge_intervals(cls, intervals: List[MutableMapping[str, Any]]) -> List[MutableMapping[str, Any]]:
        if False:
            i = 10
            return i + 15
        sorted_intervals = sorted(intervals, key=lambda x: (x[cls.START_KEY], x[cls.END_KEY]))
        if len(sorted_intervals) > 0:
            merged_intervals = [sorted_intervals[0]]
        else:
            return []
        for interval in sorted_intervals[1:]:
            if interval[cls.START_KEY] <= cls.increment(merged_intervals[-1][cls.END_KEY]):
                merged_intervals[-1][cls.END_KEY] = interval[cls.END_KEY]
            else:
                merged_intervals.append(interval)
        return merged_intervals

class EpochValueConcurrentStreamStateConverter(ConcurrentStreamStateConverter):

    def __init__(self, cursor_field: str):
        if False:
            while True:
                i = 10
        self._cursor_field = cursor_field

    def convert_from_sequential_state(self, stream_state: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        if False:
            while True:
                i = 10
        '\n        e.g.\n        { "created": 1617030403 }\n        =>\n        {\n            "state_type": "date-range",\n            "metadata": { … },\n            "slices": [\n                {starts: 0, end: 1617030403, finished_processing: true}\n            ]\n        }\n        '
        if self.is_state_message_compatible(stream_state):
            return stream_state
        if self._cursor_field in stream_state:
            slices = [{self.START_KEY: 0, self.END_KEY: stream_state[self._cursor_field]}]
        else:
            slices = []
        return {'state_type': ConcurrencyCompatibleStateType.date_range.value, 'slices': slices, 'legacy': stream_state}

    def convert_to_sequential_state(self, stream_state: MutableMapping[str, Any]) -> Any:
        if False:
            while True:
                i = 10
        '\n        e.g.\n        {\n            "state_type": "date-range",\n            "metadata": { … },\n            "slices": [\n                {starts: 0, end: 1617030403, finished_processing: true}\n            ]\n        }\n        =>\n        { "created": 1617030403 }\n        '
        if self.is_state_message_compatible(stream_state):
            legacy_state = stream_state.get('legacy', {})
            if (slices := stream_state.pop('slices', None)):
                legacy_state.update({self._cursor_field: self._get_latest_complete_time(slices)})
            return legacy_state
        else:
            return stream_state

    @staticmethod
    def increment(timestamp: Any) -> Any:
        if False:
            print('Hello World!')
        return timestamp + 1