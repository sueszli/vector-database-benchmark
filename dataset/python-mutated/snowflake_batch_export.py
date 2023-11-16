import datetime as dt
import json
import tempfile
from dataclasses import dataclass
import snowflake.connector
from django.conf import settings
from snowflake.connector.cursor import SnowflakeCursor
from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from posthog.batch_exports.service import SnowflakeBatchExportInputs
from posthog.temporal.workflows.base import PostHogWorkflow
from posthog.temporal.workflows.batch_exports import CreateBatchExportRunInputs, UpdateBatchExportRunStatusInputs, create_export_run, execute_batch_export_insert_activity, get_data_interval, get_results_iterator, get_rows_count
from posthog.temporal.workflows.clickhouse import get_client
from posthog.temporal.workflows.logger import bind_batch_exports_logger
from posthog.temporal.workflows.metrics import get_bytes_exported_metric, get_rows_exported_metric

class SnowflakeFileNotUploadedError(Exception):
    """Raised when a PUT Snowflake query fails to upload a file."""

    def __init__(self, table_name: str, status: str, message: str):
        if False:
            print('Hello World!')
        super().__init__(f"Snowflake upload for table '{table_name}' expected status 'UPLOADED' but got '{status}': {message}")

class SnowflakeFileNotLoadedError(Exception):
    """Raised when a COPY INTO Snowflake query fails to copy a file to a table."""

    def __init__(self, table_name: str, status: str, errors_seen: int, first_error: str):
        if False:
            while True:
                i = 10
        super().__init__(f"Snowflake load for table '{table_name}' expected status 'LOADED' but got '{status}' with {errors_seen} errors: {first_error}")

@dataclass
class SnowflakeInsertInputs:
    """Inputs for Snowflake."""
    team_id: int
    user: str
    password: str
    account: str
    database: str
    warehouse: str
    schema: str
    table_name: str
    data_interval_start: str
    data_interval_end: str
    role: str | None = None
    exclude_events: list[str] | None = None
    include_events: list[str] | None = None

def put_file_to_snowflake_table(cursor: SnowflakeCursor, file_name: str, table_name: str):
    if False:
        i = 10
        return i + 15
    "Executes a PUT query using the provided cursor to the provided table_name.\n\n    Args:\n        cursor: A Snowflake cursor to execute the PUT query.\n        file_name: The name of the file to PUT.\n        table_name: The name of the table where to PUT the file.\n\n    Raises:\n        TypeError: If we don't get a tuple back from Snowflake (should never happen).\n        SnowflakeFileNotUploadedError: If the upload status is not 'UPLOADED'.\n    "
    cursor.execute(f'\n        PUT file://{file_name} @%"{table_name}"\n        ')
    result = cursor.fetchone()
    if not isinstance(result, tuple):
        raise TypeError(f"Expected tuple from Snowflake PUT query but got: '{result.__class__.__name__}'")
    (status, message) = result[6:8]
    if status != 'UPLOADED':
        raise SnowflakeFileNotUploadedError(table_name, status, message)

@activity.defn
async def insert_into_snowflake_activity(inputs: SnowflakeInsertInputs):
    """Activity streams data from ClickHouse to Snowflake.

    TODO: We're using JSON here, it's not the most efficient way to do this.
    """
    logger = await bind_batch_exports_logger(team_id=inputs.team_id, destination='Snowflake')
    logger.info('Exporting batch %s - %s', inputs.data_interval_start, inputs.data_interval_end)
    async with get_client() as client:
        if not await client.is_alive():
            raise ConnectionError('Cannot establish connection to ClickHouse')
        count = await get_rows_count(client=client, team_id=inputs.team_id, interval_start=inputs.data_interval_start, interval_end=inputs.data_interval_end, exclude_events=inputs.exclude_events, include_events=inputs.include_events)
        if count == 0:
            logger.info('Nothing to export in batch %s - %s', inputs.data_interval_start, inputs.data_interval_end)
            return
        logger.info('BatchExporting %s rows', count)
        conn = snowflake.connector.connect(user=inputs.user, password=inputs.password, account=inputs.account, warehouse=inputs.warehouse, database=inputs.database, schema=inputs.schema, role=inputs.role)
        try:
            cursor = conn.cursor()
            cursor.execute(f'USE DATABASE "{inputs.database}"')
            cursor.execute(f'USE SCHEMA "{inputs.schema}"')
            cursor.execute(f'''\n                CREATE TABLE IF NOT EXISTS "{inputs.database}"."{inputs.schema}"."{inputs.table_name}" (\n                    "uuid" STRING,\n                    "event" STRING,\n                    "properties" VARIANT,\n                    "elements" VARIANT,\n                    "people_set" VARIANT,\n                    "people_set_once" VARIANT,\n                    "distinct_id" STRING,\n                    "team_id" INTEGER,\n                    "ip" STRING,\n                    "site_url" STRING,\n                    "timestamp" TIMESTAMP\n                )\n                COMMENT = 'PostHog generated events table'\n                ''')
            results_iterator = get_results_iterator(client=client, team_id=inputs.team_id, interval_start=inputs.data_interval_start, interval_end=inputs.data_interval_end, exclude_events=inputs.exclude_events, include_events=inputs.include_events)
            result = None
            local_results_file = tempfile.NamedTemporaryFile(suffix='.jsonl')
            rows_in_file = 0
            rows_exported = get_rows_exported_metric()
            bytes_exported = get_bytes_exported_metric()

            def flush_to_snowflake(lrf: tempfile._TemporaryFileWrapper, rows_in_file: int):
                if False:
                    i = 10
                    return i + 15
                lrf.flush()
                put_file_to_snowflake_table(cursor, lrf.name, inputs.table_name)
                rows_exported.add(rows_in_file)
                bytes_exported.add(lrf.tell())
            try:
                while True:
                    try:
                        result = results_iterator.__next__()
                    except StopIteration:
                        break
                    except json.JSONDecodeError:
                        logger.info('Failed to decode a JSON value while iterating, potentially due to a ClickHouse error')
                        if result is None:
                            new_interval_start = None
                        else:
                            new_interval_start = result.get('inserted_at', None)
                        if not isinstance(new_interval_start, str):
                            new_interval_start = inputs.data_interval_start
                        results_iterator = get_results_iterator(client=client, team_id=inputs.team_id, interval_start=new_interval_start, interval_end=inputs.data_interval_end)
                        continue
                    if not result:
                        break
                    local_results_file.write(json.dumps(result).encode('utf-8'))
                    local_results_file.write('\n'.encode('utf-8'))
                    rows_in_file += 1
                    if local_results_file.tell() and local_results_file.tell() > settings.BATCH_EXPORT_SNOWFLAKE_UPLOAD_CHUNK_SIZE_BYTES:
                        logger.info('Uploading to Snowflake')
                        flush_to_snowflake(local_results_file, rows_in_file)
                        local_results_file.close()
                        local_results_file = tempfile.NamedTemporaryFile(suffix='.jsonl')
                        rows_in_file = 0
                flush_to_snowflake(local_results_file, rows_in_file)
                local_results_file.close()
                cursor.execute(f'''\n                    COPY INTO "{inputs.table_name}"\n                    FILE_FORMAT = (TYPE = 'JSON')\n                    MATCH_BY_COLUMN_NAME = CASE_SENSITIVE\n                    PURGE = TRUE\n                    ''')
                results = cursor.fetchall()
                for query_result in results:
                    if not isinstance(query_result, tuple):
                        raise TypeError(f"Expected tuple from Snowflake COPY INTO query but got: '{type(result)}'")
                    if len(query_result) < 2:
                        raise SnowflakeFileNotLoadedError(inputs.table_name, 'NO STATUS', 0, query_result[1] if len(query_result) == 1 else 'NO ERROR MESSAGE')
                    (_, status) = query_result[0:2]
                    if status != 'LOADED':
                        (errors_seen, first_error) = query_result[5:7]
                        raise SnowflakeFileNotLoadedError(inputs.table_name, status or 'NO STATUS', errors_seen or 0, first_error or 'NO ERROR MESSAGE')
            finally:
                local_results_file.close()
        finally:
            conn.close()

@workflow.defn(name='snowflake-export')
class SnowflakeBatchExportWorkflow(PostHogWorkflow):
    """A Temporal Workflow to export ClickHouse data into Snowflake.

    This Workflow is intended to be executed both manually and by a Temporal
    Schedule. When ran by a schedule, `data_interval_end` should be set to
    `None` so that we will fetch the end of the interval from the Temporal
    search attribute `TemporalScheduledStartTime`.
    """

    @staticmethod
    def parse_inputs(inputs: list[str]) -> SnowflakeBatchExportInputs:
        if False:
            while True:
                i = 10
        'Parse inputs from the management command CLI.'
        loaded = json.loads(inputs[0])
        return SnowflakeBatchExportInputs(**loaded)

    @workflow.run
    async def run(self, inputs: SnowflakeBatchExportInputs):
        """Workflow implementation to export data to Snowflake table."""
        (data_interval_start, data_interval_end) = get_data_interval(inputs.interval, inputs.data_interval_end)
        create_export_run_inputs = CreateBatchExportRunInputs(team_id=inputs.team_id, batch_export_id=inputs.batch_export_id, data_interval_start=data_interval_start.isoformat(), data_interval_end=data_interval_end.isoformat())
        run_id = await workflow.execute_activity(create_export_run, create_export_run_inputs, start_to_close_timeout=dt.timedelta(minutes=5), retry_policy=RetryPolicy(initial_interval=dt.timedelta(seconds=10), maximum_interval=dt.timedelta(seconds=60), maximum_attempts=0, non_retryable_error_types=['NotNullViolation', 'IntegrityError']))
        update_inputs = UpdateBatchExportRunStatusInputs(id=run_id, status='Completed', team_id=inputs.team_id)
        insert_inputs = SnowflakeInsertInputs(team_id=inputs.team_id, user=inputs.user, password=inputs.password, account=inputs.account, warehouse=inputs.warehouse, database=inputs.database, schema=inputs.schema, table_name=inputs.table_name, data_interval_start=data_interval_start.isoformat(), data_interval_end=data_interval_end.isoformat(), role=inputs.role, exclude_events=inputs.exclude_events, include_events=inputs.include_events)
        await execute_batch_export_insert_activity(insert_into_snowflake_activity, insert_inputs, non_retryable_error_types=['DatabaseError', 'ProgrammingError', 'ForbiddenError'], update_inputs=update_inputs, heartbeat_timeout_seconds=None)