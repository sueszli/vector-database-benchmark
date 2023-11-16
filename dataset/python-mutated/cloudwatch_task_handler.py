from __future__ import annotations
from datetime import date, datetime, timedelta
from functools import cached_property
from typing import TYPE_CHECKING, Any
import watchtower
from airflow.configuration import conf
from airflow.providers.amazon.aws.hooks.logs import AwsLogsHook
from airflow.providers.amazon.aws.utils import datetime_to_epoch_utc_ms
from airflow.utils.log.file_task_handler import FileTaskHandler
from airflow.utils.log.logging_mixin import LoggingMixin
if TYPE_CHECKING:
    from airflow.models import TaskInstance

def json_serialize_legacy(value: Any) -> str | None:
    if False:
        i = 10
        return i + 15
    '\n    JSON serializer replicating legacy watchtower behavior.\n\n    The legacy `watchtower@2.0.1` json serializer function that serialized\n    datetime objects as ISO format and all other non-JSON-serializable to `null`.\n\n    :param value: the object to serialize\n    :return: string representation of `value` if it is an instance of datetime or `None` otherwise\n    '
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    else:
        return None

def json_serialize(value: Any) -> str | None:
    if False:
        for i in range(10):
            print('nop')
    '\n    JSON serializer replicating current watchtower behavior.\n\n    This provides customers with an accessible import,\n    `airflow.providers.amazon.aws.log.cloudwatch_task_handler.json_serialize`\n\n    :param value: the object to serialize\n    :return: string representation of `value`\n    '
    return watchtower._json_serialize_default(value)

class CloudwatchTaskHandler(FileTaskHandler, LoggingMixin):
    """
    CloudwatchTaskHandler is a python log handler that handles and reads task instance logs.

    It extends airflow FileTaskHandler and uploads to and reads from Cloudwatch.

    :param base_log_folder: base folder to store logs locally
    :param log_group_arn: ARN of the Cloudwatch log group for remote log storage
        with format ``arn:aws:logs:{region name}:{account id}:log-group:{group name}``
    :param filename_template: template for file name (local storage) or log stream name (remote)
    """
    trigger_should_wrap = True

    def __init__(self, base_log_folder: str, log_group_arn: str, filename_template: str | None=None):
        if False:
            i = 10
            return i + 15
        super().__init__(base_log_folder, filename_template)
        split_arn = log_group_arn.split(':')
        self.handler = None
        self.log_group = split_arn[6]
        self.region_name = split_arn[3]
        self.closed = False

    @cached_property
    def hook(self):
        if False:
            return 10
        'Returns AwsLogsHook.'
        return AwsLogsHook(aws_conn_id=conf.get('logging', 'REMOTE_LOG_CONN_ID'), region_name=self.region_name)

    def _render_filename(self, ti, try_number):
        if False:
            print('Hello World!')
        return super()._render_filename(ti, try_number).replace(':', '_')

    def set_context(self, ti):
        if False:
            for i in range(10):
                print('nop')
        super().set_context(ti)
        self.json_serialize = conf.getimport('aws', 'cloudwatch_task_handler_json_serializer')
        self.handler = watchtower.CloudWatchLogHandler(log_group_name=self.log_group, log_stream_name=self._render_filename(ti, ti.try_number), use_queues=not getattr(ti, 'is_trigger_log_context', False), boto3_client=self.hook.get_conn(), json_serialize_default=self.json_serialize)

    def close(self):
        if False:
            while True:
                i = 10
        'Close the handler responsible for the upload of the local log file to Cloudwatch.'
        if self.closed:
            return
        if self.handler is not None:
            self.handler.close()
        self.closed = True

    def _read(self, task_instance, try_number, metadata=None):
        if False:
            for i in range(10):
                print('nop')
        stream_name = self._render_filename(task_instance, try_number)
        try:
            return (f'*** Reading remote log from Cloudwatch log_group: {self.log_group} log_stream: {stream_name}.\n{self.get_cloudwatch_logs(stream_name=stream_name, task_instance=task_instance)}\n', {'end_of_log': True})
        except Exception as e:
            log = f'*** Unable to read remote logs from Cloudwatch (log_group: {self.log_group}, log_stream: {stream_name})\n*** {e}\n\n'
            self.log.error(log)
            (local_log, metadata) = super()._read(task_instance, try_number, metadata)
            log += local_log
            return (log, metadata)

    def get_cloudwatch_logs(self, stream_name: str, task_instance: TaskInstance) -> str:
        if False:
            print('Hello World!')
        '\n        Return all logs from the given log stream.\n\n        :param stream_name: name of the Cloudwatch log stream to get all logs from\n        :param task_instance: the task instance to get logs about\n        :return: string of all logs from the given log stream\n        '
        end_time = None if task_instance.end_date is None else datetime_to_epoch_utc_ms(task_instance.end_date + timedelta(seconds=30))
        events = self.hook.get_log_events(log_group=self.log_group, log_stream_name=stream_name, end_time=end_time)
        return '\n'.join((self._event_to_str(event) for event in events))

    def _event_to_str(self, event: dict) -> str:
        if False:
            for i in range(10):
                print('nop')
        event_dt = datetime.utcfromtimestamp(event['timestamp'] / 1000.0)
        formatted_event_dt = event_dt.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        message = event['message']
        return f'[{formatted_event_dt}] {message}'