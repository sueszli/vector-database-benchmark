from __future__ import annotations
from unittest import mock
import pytest
from airflow.exceptions import AirflowException
from airflow.providers.segment.hooks.segment import SegmentHook
from airflow.providers.segment.operators.segment_track_event import SegmentTrackEventOperator
TEST_CONN_ID = 'test_segment'
WRITE_KEY = 'foo'

class TestSegmentHook:

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.conn = conn = mock.MagicMock()
        conn.write_key = WRITE_KEY
        self.expected_write_key = WRITE_KEY
        self.conn.extra_dejson = {'write_key': self.expected_write_key}

        class UnitTestSegmentHook(SegmentHook):

            def get_conn(self):
                if False:
                    print('Hello World!')
                return conn

            def get_connection(self, unused_connection_id):
                if False:
                    print('Hello World!')
                return conn
        self.test_hook = UnitTestSegmentHook(segment_conn_id=TEST_CONN_ID)

    def test_get_conn(self):
        if False:
            return 10
        expected_connection = self.test_hook.get_conn()
        assert expected_connection == self.conn
        assert expected_connection.write_key is not None
        assert expected_connection.write_key == self.expected_write_key

    def test_on_error(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(AirflowException):
            self.test_hook.on_error('error', ['items'])

class TestSegmentTrackEventOperator:

    @mock.patch('airflow.providers.segment.operators.segment_track_event.SegmentHook')
    def test_execute(self, mock_hook):
        if False:
            print('Hello World!')
        user_id = 'user_id'
        event = 'event'
        properties = {}
        operator = SegmentTrackEventOperator(task_id='segment-track', user_id=user_id, event=event, properties=properties)
        operator.execute(None)
        mock_hook.return_value.track.assert_called_once_with(user_id=user_id, event=event, properties=properties)