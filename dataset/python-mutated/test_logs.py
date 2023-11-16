from __future__ import annotations
from unittest import mock
from unittest.mock import ANY, patch
import pytest
from moto import mock_logs
from airflow.providers.amazon.aws.hooks.logs import AwsLogsHook

@mock_logs
class TestAwsLogsHook:

    @pytest.mark.parametrize('get_log_events_response, num_skip_events, expected_num_events, end_time', [([{'nextForwardToken': '1', 'events': []}, {'nextForwardToken': '2', 'events': []}, {'nextForwardToken': '3', 'events': []}], 0, 0, None), ([{'nextForwardToken': '', 'events': []}, {'nextForwardToken': '', 'events': [{}, {}]}], 0, 2, None), ([{'nextForwardToken': '1', 'events': []}, {'nextForwardToken': '2', 'events': [{}, {}]}, {'nextForwardToken': '3', 'events': []}, {'nextForwardToken': '4', 'events': []}, {'nextForwardToken': '5', 'events': []}, {'nextForwardToken': '6', 'events': [{}, {}]}], 0, 2, 10), ([{'nextForwardToken': '1', 'events': []}, {'nextForwardToken': '2', 'events': [{}, {}]}, {'nextForwardToken': '3', 'events': []}, {'nextForwardToken': '4', 'events': []}, {'nextForwardToken': '6', 'events': [{}, {}]}, {'nextForwardToken': '6', 'events': [{}, {}]}, {'nextForwardToken': '6', 'events': [{}, {}]}], 0, 6, 20)])
    @patch('airflow.providers.amazon.aws.hooks.logs.AwsLogsHook.conn', new_callable=mock.PropertyMock)
    def test_get_log_events(self, mock_conn, get_log_events_response, num_skip_events, expected_num_events, end_time):
        if False:
            print('Hello World!')
        mock_conn().get_log_events.side_effect = get_log_events_response
        log_group_name = 'example-group'
        log_stream_name = 'example-log-stream'
        hook = AwsLogsHook(aws_conn_id='aws_default', region_name='us-east-1')
        events = hook.get_log_events(log_group=log_group_name, log_stream_name=log_stream_name, skip=num_skip_events, end_time=end_time)
        events = list(events)
        assert len(events) == expected_num_events
        kwargs = {'logGroupName': log_group_name, 'logStreamName': log_stream_name, 'startFromHead': True, 'startTime': 0, 'nextToken': ANY}
        if end_time:
            kwargs['endTime'] = end_time
        mock_conn().get_log_events.assert_called_with(**kwargs)