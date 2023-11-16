import logging
import time
import unittest
from datetime import timedelta
from typing import Any, Optional
from unittest.mock import patch
from flask import current_app
from freezegun import freeze_time
from superset import security_manager
from superset.utils.log import AbstractEventLogger, DBEventLogger, get_event_logger_from_cfg_value
from tests.integration_tests.test_app import app

class TestEventLogger(unittest.TestCase):

    def test_correct_config_object(self):
        if False:
            for i in range(10):
                print('nop')
        obj = DBEventLogger()
        res = get_event_logger_from_cfg_value(obj)
        self.assertIs(obj, res)

    def test_config_class_deprecation(self):
        if False:
            print('Hello World!')
        res = None
        with self.assertLogs(level='WARNING'):
            res = get_event_logger_from_cfg_value(DBEventLogger)
        self.assertIsInstance(res, DBEventLogger)

    def test_raises_typerror_if_not_abc(self):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            get_event_logger_from_cfg_value(logging.getLogger())

    @patch.object(DBEventLogger, 'log')
    def test_log_this(self, mock_log):
        if False:
            return 10
        logger = DBEventLogger()

        @logger.log_this
        def test_func():
            if False:
                for i in range(10):
                    print('nop')
            time.sleep(0.05)
            return 1
        with app.test_request_context('/superset/dashboard/1/?myparam=foo'):
            result = test_func()
            payload = mock_log.call_args[1]
            self.assertEqual(result, 1)
            self.assertEqual(payload['records'], [{'myparam': 'foo', 'path': '/superset/dashboard/1/', 'url_rule': '/superset/dashboard/<dashboard_id_or_slug>/', 'object_ref': test_func.__qualname__}])
            self.assertGreaterEqual(payload['duration_ms'], 50)

    @patch.object(DBEventLogger, 'log')
    def test_log_this_with_extra_payload(self, mock_log):
        if False:
            return 10
        logger = DBEventLogger()

        @logger.log_this_with_extra_payload
        def test_func(arg1, add_extra_log_payload, karg1=1):
            if False:
                while True:
                    i = 10
            time.sleep(0.1)
            add_extra_log_payload(foo='bar')
            return arg1 * karg1
        with app.test_request_context():
            result = test_func(1, karg1=2)
            payload = mock_log.call_args[1]
            self.assertEqual(result, 2)
            self.assertEqual(payload['records'], [{'foo': 'bar', 'path': '/', 'karg1': 2, 'object_ref': test_func.__qualname__}])
            self.assertGreaterEqual(payload['duration_ms'], 100)

    @patch('superset.utils.core.g', spec={})
    @freeze_time('Jan 14th, 2020', auto_tick_seconds=15)
    def test_context_manager_log(self, mock_g):
        if False:
            i = 10
            return i + 15

        class DummyEventLogger(AbstractEventLogger):

            def __init__(self):
                if False:
                    return 10
                self.records = []

            def log(self, user_id: Optional[int], action: str, dashboard_id: Optional[int], duration_ms: Optional[int], slice_id: Optional[int], referrer: Optional[str], *args: Any, **kwargs: Any):
                if False:
                    for i in range(10):
                        print('nop')
                self.records.append({**kwargs, 'user_id': user_id, 'duration': duration_ms})
        logger = DummyEventLogger()
        with app.test_request_context():
            mock_g.user = security_manager.find_user('gamma')
            with logger(action='foo', engine='bar'):
                pass
        assert logger.records == [{'records': [{'path': '/', 'engine': 'bar'}], 'user_id': 2, 'duration': 15000.0}]

    @patch('superset.utils.core.g', spec={})
    def test_context_manager_log_with_context(self, mock_g):
        if False:
            print('Hello World!')

        class DummyEventLogger(AbstractEventLogger):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.records = []

            def log(self, user_id: Optional[int], action: str, dashboard_id: Optional[int], duration_ms: Optional[int], slice_id: Optional[int], referrer: Optional[str], *args: Any, **kwargs: Any):
                if False:
                    i = 10
                    return i + 15
                self.records.append({**kwargs, 'user_id': user_id, 'duration': duration_ms})
        logger = DummyEventLogger()
        with app.test_request_context():
            mock_g.user = security_manager.find_user('gamma')
            logger.log_with_context(action='foo', duration=timedelta(days=64, seconds=29156, microseconds=10), object_ref={'baz': 'food'}, log_to_statsd=False, payload_override={'engine': 'sqlite'})
        assert logger.records == [{'records': [{'path': '/', 'object_ref': {'baz': 'food'}, 'payload_override': {'engine': 'sqlite'}}], 'user_id': 2, 'duration': 5558756000}]

    @patch('superset.utils.core.g', spec={})
    def test_log_with_context_user_null(self, mock_g):
        if False:
            i = 10
            return i + 15

        class DummyEventLogger(AbstractEventLogger):

            def __init__(self):
                if False:
                    return 10
                self.records = []

            def log(self, user_id: Optional[int], action: str, dashboard_id: Optional[int], duration_ms: Optional[int], slice_id: Optional[int], referrer: Optional[str], *args: Any, **kwargs: Any):
                if False:
                    return 10
                self.records.append({**kwargs, 'user_id': user_id, 'duration': duration_ms})
        logger = DummyEventLogger()
        with app.test_request_context():
            mock_g.side_effect = Exception('oops')
            logger.log_with_context(action='foo', duration=timedelta(days=64, seconds=29156, microseconds=10), object_ref={'baz': 'food'}, log_to_statsd=False, payload_override={'engine': 'sqlite'})
        assert logger.records[0]['user_id'] == None