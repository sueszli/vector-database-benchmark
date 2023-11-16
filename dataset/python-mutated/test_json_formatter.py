"""
Module for all tests airflow.utils.log.json_formatter.JSONFormatter
"""
from __future__ import annotations
import json
import sys
from logging import makeLogRecord
from airflow.utils.log.json_formatter import JSONFormatter

class TestJSONFormatter:
    """
    TestJSONFormatter class combine all tests for JSONFormatter
    """

    def test_json_formatter_is_not_none(self):
        if False:
            i = 10
            return i + 15
        '\n        JSONFormatter instance  should return not none\n        '
        json_fmt = JSONFormatter()
        assert json_fmt is not None

    def test_uses_time(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test usesTime method from JSONFormatter\n        '
        json_fmt_asctime = JSONFormatter(json_fields=['asctime', 'label'])
        json_fmt_no_asctime = JSONFormatter(json_fields=['label'])
        assert json_fmt_asctime.usesTime()
        assert not json_fmt_no_asctime.usesTime()

    def test_format(self):
        if False:
            while True:
                i = 10
        '\n        Test format method from JSONFormatter\n        '
        log_record = makeLogRecord({'label': 'value'})
        json_fmt = JSONFormatter(json_fields=['label'])
        assert json_fmt.format(log_record) == '{"label": "value"}'

    def test_format_with_extras(self):
        if False:
            return 10
        '\n        Test format with extras method from JSONFormatter\n        '
        log_record = makeLogRecord({'label': 'value'})
        json_fmt = JSONFormatter(json_fields=['label'], extras={'pod_extra': 'useful_message'})
        assert json.loads(json_fmt.format(log_record)) == {'label': 'value', 'pod_extra': 'useful_message'}

    def test_format_with_exception(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test exception is included in the message when using JSONFormatter\n        '
        try:
            raise RuntimeError('message')
        except RuntimeError:
            exc_info = sys.exc_info()
        log_record = makeLogRecord({'exc_info': exc_info, 'message': 'Some msg'})
        json_fmt = JSONFormatter(json_fields=['message'])
        log_fmt = json.loads(json_fmt.format(log_record))
        assert 'message' in log_fmt
        assert 'Traceback (most recent call last)' in log_fmt['message']
        assert 'raise RuntimeError("message")' in log_fmt['message']