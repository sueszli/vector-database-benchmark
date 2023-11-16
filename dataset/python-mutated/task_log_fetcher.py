from __future__ import annotations
import time
from datetime import datetime, timedelta
from threading import Event, Thread
from typing import TYPE_CHECKING, Generator
from botocore.exceptions import ClientError, ConnectionClosedError
from airflow.providers.amazon.aws.hooks.logs import AwsLogsHook
if TYPE_CHECKING:
    from logging import Logger

class AwsTaskLogFetcher(Thread):
    """Fetch Cloudwatch log events with specific interval and send the log events to the logger.info."""

    def __init__(self, *, log_group: str, log_stream_name: str, fetch_interval: timedelta, logger: Logger, aws_conn_id: str | None='aws_default', region_name: str | None=None):
        if False:
            print('Hello World!')
        super().__init__()
        self._event = Event()
        self.fetch_interval = fetch_interval
        self.logger = logger
        self.log_group = log_group
        self.log_stream_name = log_stream_name
        self.hook = AwsLogsHook(aws_conn_id=aws_conn_id, region_name=region_name)

    def run(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        continuation_token = AwsLogsHook.ContinuationToken()
        while not self.is_stopped():
            time.sleep(self.fetch_interval.total_seconds())
            log_events = self._get_log_events(continuation_token)
            for log_event in log_events:
                self.logger.info(self.event_to_str(log_event))

    def _get_log_events(self, skip_token: AwsLogsHook.ContinuationToken | None=None) -> Generator:
        if False:
            while True:
                i = 10
        if skip_token is None:
            skip_token = AwsLogsHook.ContinuationToken()
        try:
            yield from self.hook.get_log_events(self.log_group, self.log_stream_name, continuation_token=skip_token)
        except ClientError as error:
            if error.response['Error']['Code'] != 'ResourceNotFoundException':
                self.logger.warning('Error on retrieving Cloudwatch log events', error)
            else:
                self.logger.info('Cannot find log stream yet, it can take a couple of seconds to show up. If this error persists, check that the log group and stream are correct: group: %s\tstream: %s', self.log_group, self.log_stream_name)
            yield from ()
        except ConnectionClosedError as error:
            self.logger.warning('ConnectionClosedError on retrieving Cloudwatch log events', error)
            yield from ()

    @staticmethod
    def event_to_str(event: dict) -> str:
        if False:
            for i in range(10):
                print('nop')
        event_dt = datetime.utcfromtimestamp(event['timestamp'] / 1000.0)
        formatted_event_dt = event_dt.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        message = event['message']
        return f'[{formatted_event_dt}] {message}'

    def get_last_log_messages(self, number_messages) -> list:
        if False:
            print('Hello World!')
        '\n        Get the last logs messages in one single request.\n\n         NOTE: some restrictions apply:\n         - if logs are too old, the response will be empty\n         - the max number of messages we can retrieve is constrained by cloudwatch limits (10,000).\n        '
        response = self.hook.conn.get_log_events(logGroupName=self.log_group, logStreamName=self.log_stream_name, startFromHead=False, limit=number_messages)
        return [log['message'] for log in response['events']]

    def get_last_log_message(self) -> str | None:
        if False:
            while True:
                i = 10
        try:
            return self.get_last_log_messages(1)[0]
        except IndexError:
            return None

    def is_stopped(self) -> bool:
        if False:
            print('Hello World!')
        return self._event.is_set()

    def stop(self):
        if False:
            print('Hello World!')
        self._event.set()