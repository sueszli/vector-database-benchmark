"""Support managing EventTypes."""
from __future__ import annotations
from collections.abc import Iterable
from typing import TYPE_CHECKING, cast
from lru import LRU
from sqlalchemy.orm.session import Session
from homeassistant.core import Event
from ..db_schema import EventTypes
from ..queries import find_event_type_ids
from ..tasks import RefreshEventTypesTask
from ..util import chunked, execute_stmt_lambda_element
from . import BaseLRUTableManager
if TYPE_CHECKING:
    from ..core import Recorder
CACHE_SIZE = 2048

class EventTypeManager(BaseLRUTableManager[EventTypes]):
    """Manage the EventTypes table."""

    def __init__(self, recorder: Recorder) -> None:
        if False:
            i = 10
            return i + 15
        'Initialize the event type manager.'
        super().__init__(recorder, CACHE_SIZE)
        self._non_existent_event_types: LRU = LRU(CACHE_SIZE)

    def load(self, events: list[Event], session: Session) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Load the event_type to event_type_ids mapping into memory.\n\n        This call is not thread-safe and must be called from the\n        recorder thread.\n        '
        self.get_many({event.event_type for event in events if event.event_type is not None}, session, True)

    def get(self, event_type: str, session: Session, from_recorder: bool=False) -> int | None:
        if False:
            while True:
                i = 10
        'Resolve event_type to the event_type_id.\n\n        This call is not thread-safe and must be called from the\n        recorder thread.\n        '
        return self.get_many((event_type,), session)[event_type]

    def get_many(self, event_types: Iterable[str], session: Session, from_recorder: bool=False) -> dict[str, int | None]:
        if False:
            while True:
                i = 10
        'Resolve event_types to event_type_ids.\n\n        This call is not thread-safe and must be called from the\n        recorder thread.\n        '
        results: dict[str, int | None] = {}
        missing: list[str] = []
        non_existent: list[str] = []
        for event_type in event_types:
            if (event_type_id := self._id_map.get(event_type)) is None:
                if event_type in self._non_existent_event_types:
                    results[event_type] = None
                else:
                    missing.append(event_type)
            results[event_type] = event_type_id
        if not missing:
            return results
        with session.no_autoflush:
            for missing_chunk in chunked(missing, self.recorder.max_bind_vars):
                for (event_type_id, event_type) in execute_stmt_lambda_element(session, find_event_type_ids(missing_chunk), orm_rows=False):
                    results[event_type] = self._id_map[event_type] = cast(int, event_type_id)
        if (non_existent := [event_type for event_type in missing if results[event_type] is None]):
            if from_recorder:
                for event_type in non_existent:
                    self._non_existent_event_types[event_type] = None
            else:
                self.recorder.queue_task(RefreshEventTypesTask(non_existent))
        return results

    def add_pending(self, db_event_type: EventTypes) -> None:
        if False:
            print('Hello World!')
        'Add a pending EventTypes that will be committed at the next interval.\n\n        This call is not thread-safe and must be called from the\n        recorder thread.\n        '
        assert db_event_type.event_type is not None
        event_type: str = db_event_type.event_type
        self._pending[event_type] = db_event_type

    def post_commit_pending(self) -> None:
        if False:
            while True:
                i = 10
        'Call after commit to load the event_type_ids of the new EventTypes into the LRU.\n\n        This call is not thread-safe and must be called from the\n        recorder thread.\n        '
        for (event_type, db_event_types) in self._pending.items():
            self._id_map[event_type] = db_event_types.event_type_id
            self.clear_non_existent(event_type)
        self._pending.clear()

    def clear_non_existent(self, event_type: str) -> None:
        if False:
            print('Hello World!')
        'Clear a non-existent event type from the cache.\n\n        This call is not thread-safe and must be called from the\n        recorder thread.\n        '
        self._non_existent_event_types.pop(event_type, None)

    def evict_purged(self, event_types: Iterable[str]) -> None:
        if False:
            while True:
                i = 10
        'Evict purged event_types from the cache when they are no longer used.\n\n        This call is not thread-safe and must be called from the\n        recorder thread.\n        '
        for event_type in event_types:
            self._id_map.pop(event_type, None)