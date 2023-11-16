import enum
from typing import NamedTuple, Optional
import dagster._check as check
from dagster import EventLogEntry
from dagster._core.events import DagsterEventType
from dagster._serdes.serdes import deserialize_value
from dagster._utils import datetime_as_float

class AssetCheckInstanceSupport(enum.Enum):
    """Reasons why a dagster instance might not support checks."""
    SUPPORTED = 'SUPPORTED'
    NEEDS_MIGRATION = 'NEEDS_MIGRATION'
    NEEDS_AGENT_UPGRADE = 'NEEDS_AGENT_UPGRADE'

class AssetCheckExecutionRecordStatus(enum.Enum):
    PLANNED = 'PLANNED'
    SUCCEEDED = 'SUCCEEDED'
    FAILED = 'FAILED'

class AssetCheckExecutionResolvedStatus(enum.Enum):
    IN_PROGRESS = 'IN_PROGRESS'
    SUCCEEDED = 'SUCCEEDED'
    FAILED = 'FAILED'
    EXECUTION_FAILED = 'EXECUTION_FAILED'
    SKIPPED = 'SKIPPED'

class AssetCheckExecutionRecord(NamedTuple('_AssetCheckExecutionRecord', [('id', int), ('run_id', str), ('status', AssetCheckExecutionRecordStatus), ('event', Optional[EventLogEntry]), ('create_timestamp', float)])):

    def __new__(cls, id: int, run_id: str, status: AssetCheckExecutionRecordStatus, event: Optional[EventLogEntry], create_timestamp: float):
        if False:
            while True:
                i = 10
        check.int_param(id, 'id')
        check.str_param(run_id, 'run_id')
        check.inst_param(status, 'status', AssetCheckExecutionRecordStatus)
        check.opt_inst_param(event, 'event', EventLogEntry)
        check.float_param(create_timestamp, 'create_timestamp')
        event_type = event.dagster_event_type if event else None
        if status == AssetCheckExecutionRecordStatus.PLANNED:
            check.invariant(event is None or event_type == DagsterEventType.ASSET_CHECK_EVALUATION_PLANNED, f'The asset check row status is PLANNED, but the event is type {event_type} instead of ASSET_CHECK_EVALUATION_PLANNED')
        elif status in [AssetCheckExecutionRecordStatus.FAILED, AssetCheckExecutionRecordStatus.SUCCEEDED]:
            check.invariant(event_type == DagsterEventType.ASSET_CHECK_EVALUATION, f'The asset check row status is {status}, but the event is type {event_type} instead of ASSET_CHECK_EVALUATION')
        return super(AssetCheckExecutionRecord, cls).__new__(cls, id=id, run_id=run_id, status=status, event=event, create_timestamp=create_timestamp)

    @classmethod
    def from_db_row(cls, row) -> 'AssetCheckExecutionRecord':
        if False:
            while True:
                i = 10
        return cls(id=row['id'], run_id=row['run_id'], status=AssetCheckExecutionRecordStatus(row['execution_status']), event=deserialize_value(row['evaluation_event'], EventLogEntry) if row['evaluation_event'] else None, create_timestamp=datetime_as_float(row['create_timestamp']))