import logging
import os
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Mapping, Optional, Sequence
import dagster._check as check
from dagster import DagsterEventType
from dagster._core.instance import DagsterInstance
from dagster._core.storage.dagster_run import RunRecord, RunsFilter
from dagster._core.workspace.context import IWorkspaceProcessContext
from ..daemon import IntervalDaemon
from .auto_run_reexecution import consume_new_runs_for_automatic_reexecution
if TYPE_CHECKING:
    from dagster._core.events.log import EventLogEntry
_INTERVAL_SECONDS = int(os.environ.get('DAGSTER_EVENT_LOG_CONSUMER_DAEMON_INTERVAL_SECONDS', 5))
_EVENT_LOG_FETCH_LIMIT = int(os.environ.get('DAGSTER_EVENT_LOG_CONSUMER_DAEMON_FETCH_LIMIT', 500))
DAGSTER_EVENT_TYPES = [DagsterEventType.RUN_FAILURE, DagsterEventType.RUN_SUCCESS]

class EventLogConsumerDaemon(IntervalDaemon):

    def __init__(self, interval_seconds: int=_INTERVAL_SECONDS, event_log_fetch_limit: int=_EVENT_LOG_FETCH_LIMIT):
        if False:
            return 10
        super(EventLogConsumerDaemon, self).__init__(interval_seconds=interval_seconds)
        self._event_log_fetch_limit = event_log_fetch_limit

    @classmethod
    def daemon_type(cls) -> str:
        if False:
            while True:
                i = 10
        return 'EVENT_LOG_CONSUMER'

    @property
    def handle_updated_runs_fns(self) -> Sequence[Callable[[IWorkspaceProcessContext, Sequence[RunRecord]], Iterator]]:
        if False:
            while True:
                i = 10
        'List of functions that will be called with the list of run records that have new events.'
        return [consume_new_runs_for_automatic_reexecution]

    def run_iteration(self, workspace_process_context: IWorkspaceProcessContext):
        if False:
            while True:
                i = 10
        instance = workspace_process_context.instance
        persisted_cursors = _fetch_persisted_cursors(instance, DAGSTER_EVENT_TYPES, self._logger)
        overall_max_event_id = instance.event_log_storage.get_maximum_record_id()
        events: List[EventLogEntry] = []
        new_cursors: Dict[DagsterEventType, int] = {}
        for event_type in DAGSTER_EVENT_TYPES:
            yield
            cursor = persisted_cursors[event_type]
            if cursor is None:
                cursor = overall_max_event_id or 0
            events_by_log_id_for_type = instance.event_log_storage.get_logs_for_all_runs_by_log_id(after_cursor=cursor, dagster_event_type={event_type}, limit=self._event_log_fetch_limit)
            events.extend(events_by_log_id_for_type.values())
            new_cursors[event_type] = get_new_cursor(cursor, overall_max_event_id, self._event_log_fetch_limit, list(events_by_log_id_for_type.keys()))
        if events:
            run_ids = list({event.run_id for event in events})
            run_records = instance.get_run_records(filters=RunsFilter(run_ids=run_ids))
            for fn in self.handle_updated_runs_fns:
                try:
                    yield from fn(workspace_process_context, run_records)
                except Exception:
                    self._logger.exception(f'Error calling event event log consumer handler: {fn.__name__}')
        _persist_cursors(instance, new_cursors)

def _create_cursor_key(event_type: DagsterEventType) -> str:
    if False:
        print('Hello World!')
    check.inst_param(event_type, 'event_type', DagsterEventType)
    return f'EVENT_LOG_CONSUMER_CURSOR-{event_type.value}'

def _fetch_persisted_cursors(instance: DagsterInstance, event_types: Sequence[DagsterEventType], logger: logging.Logger) -> Dict[DagsterEventType, Optional[int]]:
    if False:
        while True:
            i = 10
    check.inst_param(instance, 'instance', DagsterInstance)
    check.sequence_param(event_types, 'event_types', of_type=DagsterEventType)
    persisted_cursors = instance.daemon_cursor_storage.get_cursor_values({_create_cursor_key(event_type) for event_type in event_types})
    fetched_cursors: Dict[DagsterEventType, Optional[int]] = {}
    for event_type in event_types:
        raw_cursor_value = persisted_cursors.get(_create_cursor_key(event_type))
        if raw_cursor_value is None:
            logger.warn(f'No cursor for event type {event_type}, ignoring older events')
            fetched_cursors[event_type] = None
        else:
            try:
                cursor_value = int(raw_cursor_value)
            except ValueError:
                logger.exception(f'Invalid cursor for event_type {event_type}: {raw_cursor_value}')
                raise
            fetched_cursors[event_type] = cursor_value
    return fetched_cursors

def _persist_cursors(instance: DagsterInstance, cursors: Mapping[DagsterEventType, int]) -> None:
    if False:
        i = 10
        return i + 15
    check.inst_param(instance, 'instance', DagsterInstance)
    check.mapping_param(cursors, 'cursors', key_type=DagsterEventType, value_type=int)
    if cursors:
        instance.daemon_cursor_storage.set_cursor_values({_create_cursor_key(event_type): str(cursor_value) for (event_type, cursor_value) in cursors.items()})

def get_new_cursor(persisted_cursor: int, overall_max_event_id: Optional[int], fetch_limit: int, new_event_ids: Sequence[int]) -> int:
    if False:
        return 10
    "Return the new cursor value for an event type, or None if one shouldn't be persisted.\n\n    The cursor is guaranteed to be:\n\n    - greater than or equal to any id in new_event_ids (otherwise we could process an event twice)\n    - less than the id of any event of the desired type that hasn't been fetched yet (otherwise we\n      could skip events)\n\n    This method optimizes for moving the cursor as far forward as possible, using\n    overall_max_event_id.\n    "
    check.int_param(persisted_cursor, 'persisted_cursor')
    check.opt_int_param(overall_max_event_id, 'overall_max_event_id')
    check.int_param(fetch_limit, 'fetch_limit')
    check.sequence_param(new_event_ids, 'new_event_ids', of_type=int)
    if overall_max_event_id is None:
        if new_event_ids:
            return max(new_event_ids)
        return 0
    if not new_event_ids:
        return overall_max_event_id
    max_new_event_id = max(new_event_ids)
    check.invariant(max_new_event_id > persisted_cursor, f'The new cursor {max_new_event_id} should be greater than the previous {persisted_cursor}')
    num_new_events = len(new_event_ids)
    check.invariant(num_new_events <= fetch_limit, 'Query returned more than the limit!')
    if num_new_events == fetch_limit:
        return max_new_event_id
    else:
        if overall_max_event_id >= max_new_event_id:
            return overall_max_event_id
        return max_new_event_id