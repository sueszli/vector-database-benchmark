import collections.abc
import csv
import dataclasses
import datetime as dt
import gzip
import json
import tempfile
import typing
import uuid
from string import Template
import brotli
from asgiref.sync import sync_to_async
from django.conf import settings
from temporalio import activity, exceptions, workflow
from temporalio.common import RetryPolicy
from posthog.batch_exports.service import create_batch_export_backfill, create_batch_export_run, update_batch_export_backfill_status, update_batch_export_run_status
from posthog.temporal.workflows.logger import bind_batch_exports_logger
from posthog.temporal.workflows.metrics import get_export_finished_metric, get_export_started_metric
SELECT_QUERY_TEMPLATE = Template("\n    SELECT $fields\n    FROM events\n    WHERE\n        COALESCE(inserted_at, _timestamp) >= toDateTime64({data_interval_start}, 6, 'UTC')\n        AND COALESCE(inserted_at, _timestamp) < toDateTime64({data_interval_end}, 6, 'UTC')\n        AND team_id = {team_id}\n        $timestamp\n        $exclude_events\n        $include_events\n    $order_by\n    $format\n    ")
TIMESTAMP_PREDICATES = "\n-- These 'timestamp' checks are a heuristic to exploit the sort key.\n-- Ideally, we need a schema that serves our needs, i.e. with a sort key on the _timestamp field used for batch exports.\n-- As a side-effect, this heuristic will discard historical loads older than a day.\nAND timestamp >= toDateTime64({data_interval_start}, 6, 'UTC') - INTERVAL 2 DAY\nAND timestamp < toDateTime64({data_interval_end}, 6, 'UTC') + INTERVAL 1 DAY\n"

async def get_rows_count(client, team_id: int, interval_start: str, interval_end: str, exclude_events: collections.abc.Iterable[str] | None=None, include_events: collections.abc.Iterable[str] | None=None) -> int:
    data_interval_start_ch = dt.datetime.fromisoformat(interval_start).strftime('%Y-%m-%d %H:%M:%S')
    data_interval_end_ch = dt.datetime.fromisoformat(interval_end).strftime('%Y-%m-%d %H:%M:%S')
    if exclude_events:
        exclude_events_statement = 'AND event NOT IN {exclude_events}'
        events_to_exclude_tuple = tuple(exclude_events)
    else:
        exclude_events_statement = ''
        events_to_exclude_tuple = ()
    if include_events:
        include_events_statement = 'AND event IN {include_events}'
        events_to_include_tuple = tuple(include_events)
    else:
        include_events_statement = ''
        events_to_include_tuple = ()
    timestamp_predicates = TIMESTAMP_PREDICATES
    if str(team_id) in settings.UNCONSTRAINED_TIMESTAMP_TEAM_IDS:
        timestamp_predicates = ''
    query = SELECT_QUERY_TEMPLATE.substitute(fields='count(DISTINCT event, cityHash64(distinct_id), cityHash64(uuid)) as count', order_by='', format='', timestamp=timestamp_predicates, exclude_events=exclude_events_statement, include_events=include_events_statement)
    count = await client.read_query(query, query_parameters={'team_id': team_id, 'data_interval_start': data_interval_start_ch, 'data_interval_end': data_interval_end_ch, 'exclude_events': events_to_exclude_tuple, 'include_events': events_to_include_tuple})
    if count is None or len(count) == 0:
        raise ValueError('Unexpected result from ClickHouse: `None` returned for count query')
    return int(count)
FIELDS = '\nDISTINCT ON (event, cityHash64(distinct_id), cityHash64(uuid))\ntoString(uuid) as uuid,\nteam_id,\ntimestamp,\ninserted_at,\ncreated_at,\nevent,\nproperties,\n-- Point in time identity fields\ntoString(distinct_id) as distinct_id,\ntoString(person_id) as person_id,\n-- Autocapture fields\nelements_chain\n'
S3_FIELDS = '\nDISTINCT ON (event, cityHash64(distinct_id), cityHash64(uuid))\ntoString(uuid) as uuid,\nteam_id,\ntimestamp,\ninserted_at,\ncreated_at,\nevent,\nproperties,\n-- Point in time identity fields\ntoString(distinct_id) as distinct_id,\ntoString(person_id) as person_id,\nperson_properties,\n-- Autocapture fields\nelements_chain\n'

def get_results_iterator(client, team_id: int, interval_start: str, interval_end: str, exclude_events: collections.abc.Iterable[str] | None=None, include_events: collections.abc.Iterable[str] | None=None, include_person_properties: bool=False) -> typing.Generator[dict[str, typing.Any], None, None]:
    if False:
        return 10
    data_interval_start_ch = dt.datetime.fromisoformat(interval_start).strftime('%Y-%m-%d %H:%M:%S')
    data_interval_end_ch = dt.datetime.fromisoformat(interval_end).strftime('%Y-%m-%d %H:%M:%S')
    if exclude_events:
        exclude_events_statement = 'AND event NOT IN {exclude_events}'
        events_to_exclude_tuple = tuple(exclude_events)
    else:
        exclude_events_statement = ''
        events_to_exclude_tuple = ()
    if include_events:
        include_events_statement = 'AND event IN {include_events}'
        events_to_include_tuple = tuple(include_events)
    else:
        include_events_statement = ''
        events_to_include_tuple = ()
    timestamp_predicates = TIMESTAMP_PREDICATES
    if str(team_id) in settings.UNCONSTRAINED_TIMESTAMP_TEAM_IDS:
        timestamp_predicates = ''
    query = SELECT_QUERY_TEMPLATE.substitute(fields=S3_FIELDS if include_person_properties else FIELDS, order_by='ORDER BY inserted_at', format='FORMAT ArrowStream', timestamp=timestamp_predicates, exclude_events=exclude_events_statement, include_events=include_events_statement)
    for batch in client.stream_query_as_arrow(query, query_parameters={'team_id': team_id, 'data_interval_start': data_interval_start_ch, 'data_interval_end': data_interval_end_ch, 'exclude_events': events_to_exclude_tuple, 'include_events': events_to_include_tuple}):
        yield from iter_batch_records(batch)

def iter_batch_records(batch) -> typing.Generator[dict[str, typing.Any], None, None]:
    if False:
        return 10
    'Iterate over records of a batch.\n\n    During iteration, we yield dictionaries with all fields used by PostHog BatchExports.\n\n    Args:\n        batch: A record batch of rows.\n    '
    for record in batch.to_pylist():
        properties = record.get('properties')
        person_properties = record.get('person_properties')
        properties = json.loads(properties) if properties else None
        elements = json.dumps(record.get('elements_chain').decode())
        record = {'created_at': record.get('created_at').isoformat(), 'distinct_id': record.get('distinct_id').decode(), 'elements': elements, 'elements_chain': record.get('elements_chain').decode(), 'event': record.get('event').decode(), 'inserted_at': record.get('inserted_at').isoformat() if record.get('inserted_at') else None, 'ip': properties.get('$ip', None) if properties else None, 'person_id': record.get('person_id').decode(), 'person_properties': json.loads(person_properties) if person_properties else None, 'set': properties.get('$set', None) if properties else None, 'set_once': properties.get('$set_once', None) if properties else None, 'properties': properties, 'site_url': '', 'team_id': record.get('team_id'), 'timestamp': record.get('timestamp').isoformat(), 'uuid': record.get('uuid').decode()}
        yield record

def get_data_interval(interval: str, data_interval_end: str | None) -> tuple[dt.datetime, dt.datetime]:
    if False:
        while True:
            i = 10
    "Return the start and end of an export's data interval.\n\n    Args:\n        interval: The interval of the BatchExport associated with this Workflow.\n        data_interval_end: The optional end of the BatchExport period. If not included, we will\n            attempt to extract it from Temporal SearchAttributes.\n\n    Raises:\n        TypeError: If when trying to obtain the data interval end we run into non-str types.\n        ValueError: If passing an unsupported interval value.\n\n    Returns:\n        A tuple of two dt.datetime indicating start and end of the data_interval.\n    "
    data_interval_end_str = data_interval_end
    if not data_interval_end_str:
        data_interval_end_search_attr = workflow.info().search_attributes.get('TemporalScheduledStartTime')
        if data_interval_end_search_attr is None:
            msg = "Expected 'TemporalScheduledStartTime' of type 'list[str]' or 'list[datetime], found 'NoneType'.This should be set by the Temporal Schedule unless triggering workflow manually.In the latter case, ensure '{Type}BatchExportInputs.data_interval_end' is set."
            raise TypeError(msg)
        if isinstance(data_interval_end_search_attr[0], str):
            data_interval_end_str = data_interval_end_search_attr[0]
            data_interval_end_dt = dt.datetime.fromisoformat(data_interval_end_str)
        elif isinstance(data_interval_end_search_attr[0], dt.datetime):
            data_interval_end_dt = data_interval_end_search_attr[0]
        else:
            msg = f"Expected search attribute to be of type 'str' or 'datetime' but found '{data_interval_end_search_attr[0]}' of type '{type(data_interval_end_search_attr[0])}'."
            raise TypeError(msg)
    else:
        data_interval_end_dt = dt.datetime.fromisoformat(data_interval_end_str)
    if interval == 'hour':
        data_interval_start_dt = data_interval_end_dt - dt.timedelta(hours=1)
    elif interval == 'day':
        data_interval_start_dt = data_interval_end_dt - dt.timedelta(days=1)
    elif interval.startswith('every'):
        (_, value, unit) = interval.split(' ')
        kwargs = {unit: int(value)}
        data_interval_start_dt = data_interval_end_dt - dt.timedelta(**kwargs)
    else:
        raise ValueError(f"Unsupported interval: '{interval}'")
    return (data_interval_start_dt, data_interval_end_dt)

def json_dumps_bytes(d, encoding='utf-8') -> bytes:
    if False:
        print('Hello World!')
    return json.dumps(d).encode(encoding)

class BatchExportTemporaryFile:
    """A TemporaryFile used to as an intermediate step while exporting data.

    This class does not implement the file-like interface but rather passes any calls
    to the underlying tempfile.NamedTemporaryFile. We do override 'write' methods
    to allow tracking bytes and records.
    """

    def __init__(self, mode: str='w+b', buffering=-1, compression: str | None=None, encoding: str | None=None, newline: str | None=None, suffix: str | None=None, prefix: str | None=None, dir: str | None=None, *, errors: str | None=None):
        if False:
            print('Hello World!')
        self._file = tempfile.NamedTemporaryFile(mode=mode, encoding=encoding, newline=newline, buffering=buffering, suffix=suffix, prefix=prefix, dir=dir, errors=errors)
        self.compression = compression
        self.bytes_total = 0
        self.records_total = 0
        self.bytes_since_last_reset = 0
        self.records_since_last_reset = 0
        self._brotli_compressor = None

    def __getattr__(self, name):
        if False:
            while True:
                i = 10
        'Pass get attr to underlying tempfile.NamedTemporaryFile.'
        return self._file.__getattr__(name)

    def __enter__(self):
        if False:
            print('Hello World!')
        'Context-manager protocol enter method.'
        self._file.__enter__()
        return self

    def __exit__(self, exc, value, tb):
        if False:
            while True:
                i = 10
        'Context-manager protocol exit method.'
        return self._file.__exit__(exc, value, tb)

    def __iter__(self):
        if False:
            print('Hello World!')
        yield from self._file

    @property
    def brotli_compressor(self):
        if False:
            print('Hello World!')
        if self._brotli_compressor is None:
            self._brotli_compressor = brotli.Compressor()
        return self._brotli_compressor

    def compress(self, content: bytes | str) -> bytes:
        if False:
            print('Hello World!')
        if isinstance(content, str):
            encoded = content.encode('utf-8')
        else:
            encoded = content
        match self.compression:
            case 'gzip':
                return gzip.compress(encoded)
            case 'brotli':
                self.brotli_compressor.process(encoded)
                return self.brotli_compressor.flush()
            case None:
                return encoded
            case _:
                raise ValueError(f"Unsupported compression: '{self.compression}'")

    def write(self, content: bytes | str):
        if False:
            return 10
        'Write bytes to underlying file keeping track of how many bytes were written.'
        compressed_content = self.compress(content)
        if 'b' in self.mode:
            result = self._file.write(compressed_content)
        else:
            result = self._file.write(compressed_content.decode('utf-8'))
        self.bytes_total += result
        self.bytes_since_last_reset += result
        return result

    def write_records_to_jsonl(self, records):
        if False:
            print('Hello World!')
        'Write records to a temporary file as JSONL.'
        jsonl_dump = b'\n'.join(map(json_dumps_bytes, records))
        if len(records) == 1:
            jsonl_dump += b'\n'
        result = self.write(jsonl_dump)
        self.records_total += len(records)
        self.records_since_last_reset += len(records)
        return result

    def write_records_to_csv(self, records, fieldnames: None | collections.abc.Sequence[str]=None, extrasaction: typing.Literal['raise', 'ignore']='ignore', delimiter: str=',', quotechar: str='"', escapechar: str='\\', quoting=csv.QUOTE_NONE):
        if False:
            while True:
                i = 10
        'Write records to a temporary file as CSV.'
        if len(records) == 0:
            return
        if fieldnames is None:
            fieldnames = list(records[0].keys())
        writer = csv.DictWriter(self, fieldnames=fieldnames, extrasaction=extrasaction, delimiter=delimiter, quotechar=quotechar, escapechar=escapechar, quoting=quoting)
        writer.writerows(records)
        self.records_total += len(records)
        self.records_since_last_reset += len(records)

    def write_records_to_tsv(self, records, fieldnames: None | list[str]=None, extrasaction: typing.Literal['raise', 'ignore']='ignore', quotechar: str='"', escapechar: str='\\', quoting=csv.QUOTE_NONE):
        if False:
            return 10
        'Write records to a temporary file as TSV.'
        return self.write_records_to_csv(records, fieldnames=fieldnames, extrasaction=extrasaction, delimiter='\t', quotechar=quotechar, escapechar=escapechar, quoting=quoting)

    def rewind(self):
        if False:
            while True:
                i = 10
        'Rewind the file before reading it.'
        if self.compression == 'brotli':
            result = self._file.write(self.brotli_compressor.finish())
            self.bytes_total += result
            self.bytes_since_last_reset += result
            self._brotli_compressor = None
        self._file.seek(0)

    def reset(self):
        if False:
            print('Hello World!')
        'Reset underlying file by truncating it.\n\n        Also resets the tracker attributes for bytes and records since last reset.\n        '
        self._file.seek(0)
        self._file.truncate()
        self.bytes_since_last_reset = 0
        self.records_since_last_reset = 0

@dataclasses.dataclass
class CreateBatchExportRunInputs:
    """Inputs to the create_export_run activity.

    Attributes:
        team_id: The id of the team the BatchExportRun belongs to.
        batch_export_id: The id of the BatchExport this BatchExportRun belongs to.
        data_interval_start: Start of this BatchExportRun's data interval.
        data_interval_end: End of this BatchExportRun's data interval.
    """
    team_id: int
    batch_export_id: str
    data_interval_start: str
    data_interval_end: str
    status: str = 'Starting'

@activity.defn
async def create_export_run(inputs: CreateBatchExportRunInputs) -> str:
    """Activity that creates an BatchExportRun.

    Intended to be used in all export workflows, usually at the start, to create a model
    instance to represent them in our database.
    """
    logger = await bind_batch_exports_logger(team_id=inputs.team_id)
    logger.info('Creating batch export for range %s - %s', inputs.data_interval_start, inputs.data_interval_end)
    run = await sync_to_async(create_batch_export_run)(batch_export_id=uuid.UUID(inputs.batch_export_id), data_interval_start=inputs.data_interval_start, data_interval_end=inputs.data_interval_end, status=inputs.status)
    return str(run.id)

@dataclasses.dataclass
class UpdateBatchExportRunStatusInputs:
    """Inputs to the update_export_run_status activity."""
    id: str
    status: str
    team_id: int
    latest_error: str | None = None

@activity.defn
async def update_export_run_status(inputs: UpdateBatchExportRunStatusInputs):
    """Activity that updates the status of an BatchExportRun."""
    logger = await bind_batch_exports_logger(team_id=inputs.team_id)
    batch_export_run = await sync_to_async(update_batch_export_run_status)(run_id=uuid.UUID(inputs.id), status=inputs.status, latest_error=inputs.latest_error)
    if batch_export_run.status == 'Failed':
        logger.error('BatchExport failed with error: %s', batch_export_run.latest_error)
    elif batch_export_run.status == 'Cancelled':
        logger.warning('BatchExport was cancelled.')
    else:
        logger.info('Successfully finished exporting batch %s - %s', batch_export_run.data_interval_start, batch_export_run.data_interval_end)

@dataclasses.dataclass
class CreateBatchExportBackfillInputs:
    team_id: int
    batch_export_id: str
    start_at: str
    end_at: str
    status: str

@activity.defn
async def create_batch_export_backfill_model(inputs: CreateBatchExportBackfillInputs) -> str:
    """Activity that creates an BatchExportBackfill.

    Intended to be used in all batch export backfill workflows, usually at the start, to create a
    model instance to represent them in our database.
    """
    logger = await bind_batch_exports_logger(team_id=inputs.team_id)
    logger.info('Creating historical export for batches in range %s - %s', inputs.start_at, inputs.end_at)
    run = await sync_to_async(create_batch_export_backfill)(batch_export_id=uuid.UUID(inputs.batch_export_id), start_at=inputs.start_at, end_at=inputs.end_at, status=inputs.status, team_id=inputs.team_id)
    return str(run.id)

@dataclasses.dataclass
class UpdateBatchExportBackfillStatusInputs:
    """Inputs to the update_batch_export_backfill_status activity."""
    id: str
    status: str

@activity.defn
async def update_batch_export_backfill_model_status(inputs: UpdateBatchExportBackfillStatusInputs):
    """Activity that updates the status of an BatchExportRun."""
    backfill = await sync_to_async(update_batch_export_backfill_status)(backfill_id=uuid.UUID(inputs.id), status=inputs.status)
    logger = await bind_batch_exports_logger(team_id=backfill.team_id)
    if backfill.status == 'Failed':
        logger.error('Historical export failed')
    elif backfill.status == 'Cancelled':
        logger.warning('Historical export was cancelled.')
    else:
        logger.info('Successfully finished exporting historical batches in %s - %s', backfill.start_at, backfill.end_at)

async def execute_batch_export_insert_activity(activity, inputs, non_retryable_error_types: list[str], update_inputs: UpdateBatchExportRunStatusInputs, start_to_close_timeout_seconds: int=3600, heartbeat_timeout_seconds: int | None=120, maximum_attempts: int=10, initial_retry_interval_seconds: int=10, maximum_retry_interval_seconds: int=120) -> None:
    """Execute the main insert activity of a batch export handling any errors.

    All batch exports boil down to inserting some data somewhere, and they all follow the same error
    handling patterns: logging and updating run status. For this reason, we have this function
    to abstract executing the main insert activity of each batch export.

    Args:
        activity: The 'insert_into_*' activity function to execute.
        inputs: The inputs to the activity.
        non_retryable_error_types: A list of errors to not retry on when executing the activity.
        update_inputs: Inputs to the update_export_run_status to run at the end.
        start_to_close_timeout: A timeout for the 'insert_into_*' activity function.
        maximum_attempts: Maximum number of retries for the 'insert_into_*' activity function.
            Assuming the error that triggered the retry is not in non_retryable_error_types.
        initial_retry_interval_seconds: When retrying, seconds until the first retry.
        maximum_retry_interval_seconds: Maximum interval in seconds between retries.
    """
    get_export_started_metric().add(1)
    retry_policy = RetryPolicy(initial_interval=dt.timedelta(seconds=initial_retry_interval_seconds), maximum_interval=dt.timedelta(seconds=maximum_retry_interval_seconds), maximum_attempts=maximum_attempts, non_retryable_error_types=non_retryable_error_types)
    try:
        await workflow.execute_activity(activity, inputs, start_to_close_timeout=dt.timedelta(seconds=start_to_close_timeout_seconds), heartbeat_timeout=dt.timedelta(seconds=heartbeat_timeout_seconds) if heartbeat_timeout_seconds else None, retry_policy=retry_policy)
    except exceptions.ActivityError as e:
        if isinstance(e.cause, exceptions.CancelledError):
            update_inputs.status = 'Cancelled'
        else:
            update_inputs.status = 'Failed'
        update_inputs.latest_error = str(e.cause)
        raise
    except Exception:
        update_inputs.status = 'Failed'
        update_inputs.latest_error = 'An unexpected error has ocurred'
        raise
    finally:
        get_export_finished_metric(status=update_inputs.status.lower()).add(1)
        await workflow.execute_activity(update_export_run_status, update_inputs, start_to_close_timeout=dt.timedelta(minutes=5), retry_policy=RetryPolicy(initial_interval=dt.timedelta(seconds=10), maximum_interval=dt.timedelta(seconds=60), maximum_attempts=0, non_retryable_error_types=['NotNullViolation', 'IntegrityError']))