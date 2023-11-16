from __future__ import annotations
from unittest import mock
import pytest
from airflow.exceptions import AirflowException
from airflow.providers.segment.hooks.segment import SegmentHook
TEST_CONN_ID = 'test_segment'
WRITE_KEY = 'foo'

class TestSegmentHook:

    def setup_method(self):
        if False:
            print('Hello World!')
        self.conn = conn = mock.MagicMock()
        conn.write_key = WRITE_KEY
        self.expected_write_key = WRITE_KEY
        self.conn.extra_dejson = {'write_key': self.expected_write_key}

        class UnitTestSegmentHook(SegmentHook):

            def get_conn(self):
                if False:
                    return 10
                return conn

            def get_connection(self, _):
                if False:
                    for i in range(10):
                        print('nop')
                return conn
        self.test_hook = UnitTestSegmentHook(segment_conn_id=TEST_CONN_ID)

    def test_get_conn(self):
        if False:
            i = 10
            return i + 15
        expected_connection = self.test_hook.get_conn()
        assert expected_connection == self.conn
        assert expected_connection.write_key is not None
        assert expected_connection.write_key == self.expected_write_key

    def test_on_error(self):
        if False:
            return 10
        with pytest.raises(AirflowException):
            self.test_hook.on_error('error', ['items'])