"""
CloudWatch log event puller implementation
"""
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from botocore.exceptions import ClientError
from samcli.lib.observability.cw_logs.cw_log_event import CWLogEvent
from samcli.lib.observability.observability_info_puller import ObservabilityEventConsumer, ObservabilityPuller
from samcli.lib.utils.time import to_datetime, to_timestamp
LOG = logging.getLogger(__name__)

class CWLogPuller(ObservabilityPuller):
    """
    Puller implementation that can pull events from CloudWatch log group
    """

    def __init__(self, logs_client: Any, consumer: ObservabilityEventConsumer, cw_log_group: str, resource_name: Optional[str]=None, max_retries: int=1000, poll_interval: int=1):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        logs_client: CloudWatchLogsClient\n            boto3 logs client instance\n        consumer : ObservabilityEventConsumer\n            Consumer instance that will process pulled events\n        cw_log_group : str\n            CloudWatch log group name\n        resource_name : Optional[str]\n            Optional parameter to assign a resource name for each event.\n        max_retries: int\n            Optional parameter to set maximum retries when tailing. Default value is 1000\n        poll_interval: int\n            Optional parameter to define sleep interval between pulling new log events when tailing. Default value is 1\n        '
        self.logs_client = logs_client
        self.consumer = consumer
        self.cw_log_group = cw_log_group
        self.resource_name = resource_name
        self._max_retries = max_retries
        self._poll_interval = poll_interval
        self.latest_event_time = 0
        self.had_data = False
        self._invalid_log_group = False

    def tail(self, start_time: Optional[datetime]=None, filter_pattern: Optional[str]=None):
        if False:
            i = 10
            return i + 15
        if start_time:
            self.latest_event_time = to_timestamp(start_time)
        counter = self._max_retries
        while counter > 0 and (not self.cancelled):
            LOG.debug('Tailing logs from %s starting at %s', self.cw_log_group, str(self.latest_event_time))
            counter -= 1
            try:
                self.load_time_period(to_datetime(self.latest_event_time), filter_pattern=filter_pattern)
            except ClientError as err:
                error_code = err.response.get('Error', {}).get('Code')
                if error_code == 'ThrottlingException':
                    if self._poll_interval == 1:
                        self._poll_interval += 1
                    else:
                        self._poll_interval **= 2
                    LOG.warning('Throttled by CloudWatch Logs API, consider pulling logs for certain resources. Increasing the poll interval time for resource %s to %s seconds', self.cw_log_group, self._poll_interval)
                else:
                    LOG.error('Failed while fetching new log events', exc_info=err)
                    raise err
            if self.had_data:
                counter = self._max_retries
                self.latest_event_time += 1
                self.had_data = False
            time.sleep(self._poll_interval)

    def load_time_period(self, start_time: Optional[datetime]=None, end_time: Optional[datetime]=None, filter_pattern: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        kwargs = {'logGroupName': self.cw_log_group, 'interleaved': True}
        if start_time:
            kwargs['startTime'] = to_timestamp(start_time)
        if end_time:
            kwargs['endTime'] = to_timestamp(end_time)
        if filter_pattern:
            kwargs['filterPattern'] = filter_pattern
        while True:
            LOG.debug('Fetching logs from CloudWatch with parameters %s', kwargs)
            try:
                result = self.logs_client.filter_log_events(**kwargs)
                self._invalid_log_group = False
            except self.logs_client.exceptions.ResourceNotFoundException:
                if not self._invalid_log_group:
                    LOG.debug('The specified log group %s does not exist. This may be due to your resource have not been invoked yet.', self.cw_log_group)
                    self._invalid_log_group = True
                break
            for event in result.get('events', []):
                self.had_data = True
                cw_event = CWLogEvent(self.cw_log_group, dict(event), self.resource_name)
                if cw_event.timestamp > self.latest_event_time:
                    self.latest_event_time = cw_event.timestamp
                self.consumer.consume(cw_event)
            next_token = result.get('nextToken', None)
            kwargs['nextToken'] = next_token
            if not next_token:
                break

    def load_events(self, event_ids: Union[List[Any], Dict]):
        if False:
            while True:
                i = 10
        LOG.debug('Loading specific events are not supported via CloudWatch Log Group')