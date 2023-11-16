"""Support managing StatesMeta."""
from __future__ import annotations
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, cast
from sqlalchemy.orm.session import Session
from homeassistant.core import Event
from ..db_schema import StatesMeta
from ..queries import find_all_states_metadata_ids, find_states_metadata_ids
from ..util import chunked, execute_stmt_lambda_element
from . import BaseLRUTableManager
if TYPE_CHECKING:
    from ..core import Recorder
CACHE_SIZE = 8192

class StatesMetaManager(BaseLRUTableManager[StatesMeta]):
    """Manage the StatesMeta table."""

    def __init__(self, recorder: Recorder) -> None:
        if False:
            i = 10
            return i + 15
        'Initialize the states meta manager.'
        self._did_first_load = False
        super().__init__(recorder, CACHE_SIZE)

    def load(self, events: list[Event], session: Session) -> None:
        if False:
            while True:
                i = 10
        'Load the entity_id to metadata_id mapping into memory.\n\n        This call is not thread-safe and must be called from the\n        recorder thread.\n        '
        self._did_first_load = True
        self.get_many({event.data['new_state'].entity_id for event in events if event.data.get('new_state') is not None}, session, True)

    def get(self, entity_id: str, session: Session, from_recorder: bool) -> int | None:
        if False:
            while True:
                i = 10
        'Resolve entity_id to the metadata_id.\n\n        This call is not thread-safe after startup since\n        purge can remove all references to an entity_id.\n\n        When calling this method from the recorder thread, set\n        from_recorder to True to ensure any missing entity_ids\n        are added to the cache.\n        '
        return self.get_many((entity_id,), session, from_recorder)[entity_id]

    def get_metadata_id_to_entity_id(self, session: Session) -> dict[int, str]:
        if False:
            return 10
        'Resolve all entity_ids to metadata_ids.\n\n        This call is always thread-safe.\n        '
        with session.no_autoflush:
            return dict(cast(Sequence[tuple[int, str]], execute_stmt_lambda_element(session, find_all_states_metadata_ids(), orm_rows=False)))

    def get_many(self, entity_ids: Iterable[str], session: Session, from_recorder: bool) -> dict[str, int | None]:
        if False:
            i = 10
            return i + 15
        'Resolve entity_id to metadata_id.\n\n        This call is not thread-safe after startup since\n        purge can remove all references to an entity_id.\n\n        When calling this method from the recorder thread, set\n        from_recorder to True to ensure any missing entity_ids\n        are added to the cache.\n        '
        results: dict[str, int | None] = {}
        missing: list[str] = []
        for entity_id in entity_ids:
            if (metadata_id := self._id_map.get(entity_id)) is None:
                missing.append(entity_id)
            results[entity_id] = metadata_id
        if not missing:
            return results
        update_cache = from_recorder or not self._did_first_load
        with session.no_autoflush:
            for missing_chunk in chunked(missing, self.recorder.max_bind_vars):
                for (metadata_id, entity_id) in execute_stmt_lambda_element(session, find_states_metadata_ids(missing_chunk)):
                    metadata_id = cast(int, metadata_id)
                    results[entity_id] = metadata_id
                    if update_cache:
                        self._id_map[entity_id] = metadata_id
        return results

    def add_pending(self, db_states_meta: StatesMeta) -> None:
        if False:
            return 10
        'Add a pending StatesMeta that will be committed at the next interval.\n\n        This call is not thread-safe and must be called from the\n        recorder thread.\n        '
        assert db_states_meta.entity_id is not None
        entity_id: str = db_states_meta.entity_id
        self._pending[entity_id] = db_states_meta

    def post_commit_pending(self) -> None:
        if False:
            return 10
        'Call after commit to load the metadata_ids of the new StatesMeta into the LRU.\n\n        This call is not thread-safe and must be called from the\n        recorder thread.\n        '
        for (entity_id, db_states_meta) in self._pending.items():
            self._id_map[entity_id] = db_states_meta.metadata_id
        self._pending.clear()

    def evict_purged(self, entity_ids: Iterable[str]) -> None:
        if False:
            i = 10
            return i + 15
        'Evict purged event_types from the cache when they are no longer used.\n\n        This call is not thread-safe and must be called from the\n        recorder thread.\n        '
        for entity_id in entity_ids:
            self._id_map.pop(entity_id, None)

    def update_metadata(self, session: Session, entity_id: str, new_entity_id: str) -> bool:
        if False:
            return 10
        'Update states metadata for an entity_id.'
        if self.get(new_entity_id, session, True) is not None:
            return False
        session.query(StatesMeta).filter(StatesMeta.entity_id == entity_id).update({StatesMeta.entity_id: new_entity_id})
        self._id_map.pop(entity_id, None)
        return True