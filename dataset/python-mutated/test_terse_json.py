import json
import logging
from io import BytesIO, StringIO
from typing import cast
from unittest.mock import Mock, patch
from twisted.web.http import HTTPChannel
from twisted.web.server import Request
from synapse.http.site import SynapseRequest
from synapse.logging._terse_json import JsonFormatter, TerseJsonFormatter
from synapse.logging.context import LoggingContext, LoggingContextFilter
from synapse.types import JsonDict
from tests.logging import LoggerCleanupMixin
from tests.server import FakeChannel, get_clock
from tests.unittest import TestCase

class TerseJsonTestCase(LoggerCleanupMixin, TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.output = StringIO()
        (self.reactor, _) = get_clock()

    def get_log_line(self) -> JsonDict:
        if False:
            print('Hello World!')
        data = self.output.getvalue()
        logs = data.splitlines()
        self.assertEqual(len(logs), 1)
        self.assertEqual(data.count('\n'), 1)
        return json.loads(logs[0])

    def test_terse_json_output(self) -> None:
        if False:
            return 10
        '\n        The Terse JSON formatter converts log messages to JSON.\n        '
        handler = logging.StreamHandler(self.output)
        handler.setFormatter(TerseJsonFormatter())
        logger = self.get_logger(handler)
        logger.info('Hello there, %s!', 'wally')
        log = self.get_log_line()
        expected_log_keys = ['log', 'time', 'level', 'namespace']
        self.assertCountEqual(log.keys(), expected_log_keys)
        self.assertEqual(log['log'], 'Hello there, wally!')

    def test_extra_data(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Additional information can be included in the structured logging.\n        '
        handler = logging.StreamHandler(self.output)
        handler.setFormatter(TerseJsonFormatter())
        logger = self.get_logger(handler)
        logger.info('Hello there, %s!', 'wally', extra={'foo': 'bar', 'int': 3, 'bool': True})
        log = self.get_log_line()
        expected_log_keys = ['log', 'time', 'level', 'namespace', 'foo', 'int', 'bool']
        self.assertCountEqual(log.keys(), expected_log_keys)
        self.assertEqual(log['foo'], 'bar')
        self.assertEqual(log['int'], 3)
        self.assertIs(log['bool'], True)

    def test_json_output(self) -> None:
        if False:
            return 10
        '\n        The Terse JSON formatter converts log messages to JSON.\n        '
        handler = logging.StreamHandler(self.output)
        handler.setFormatter(JsonFormatter())
        logger = self.get_logger(handler)
        logger.info('Hello there, %s!', 'wally')
        log = self.get_log_line()
        expected_log_keys = ['log', 'level', 'namespace']
        self.assertCountEqual(log.keys(), expected_log_keys)
        self.assertEqual(log['log'], 'Hello there, wally!')

    def test_with_context(self) -> None:
        if False:
            while True:
                i = 10
        '\n        The logging context should be added to the JSON response.\n        '
        handler = logging.StreamHandler(self.output)
        handler.setFormatter(JsonFormatter())
        handler.addFilter(LoggingContextFilter())
        logger = self.get_logger(handler)
        with LoggingContext('name'):
            logger.info('Hello there, %s!', 'wally')
        log = self.get_log_line()
        expected_log_keys = ['log', 'level', 'namespace', 'request']
        self.assertCountEqual(log.keys(), expected_log_keys)
        self.assertEqual(log['log'], 'Hello there, wally!')
        self.assertEqual(log['request'], 'name')

    def test_with_request_context(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Information from the logging context request should be added to the JSON response.\n        '
        handler = logging.StreamHandler(self.output)
        handler.setFormatter(JsonFormatter())
        handler.addFilter(LoggingContextFilter())
        logger = self.get_logger(handler)
        site = Mock(spec=['site_tag', 'server_version_string', 'getResourceFor'])
        site.site_tag = 'test-site'
        site.server_version_string = 'Server v1'
        site.reactor = Mock()
        site.experimental_cors_msc3886 = False
        request = SynapseRequest(cast(HTTPChannel, FakeChannel(site, self.reactor)), site)
        request.content = BytesIO()
        request._started_processing = Mock()
        request.request_metrics = Mock(spec=['name'])
        with patch.object(Request, 'render'):
            request.requestReceived(b'POST', b'/_matrix/client/versions', b'1.1')
        request.requester = '@foo:test'
        with LoggingContext(request.get_request_id(), parent_context=request.logcontext):
            logger.info('Hello there, %s!', 'wally')
        log = self.get_log_line()
        expected_log_keys = ['log', 'level', 'namespace', 'request', 'ip_address', 'site_tag', 'requester', 'authenticated_entity', 'method', 'url', 'protocol', 'user_agent']
        self.assertCountEqual(log.keys(), expected_log_keys)
        self.assertEqual(log['log'], 'Hello there, wally!')
        self.assertTrue(log['request'].startswith('POST-'))
        self.assertEqual(log['ip_address'], '127.0.0.1')
        self.assertEqual(log['site_tag'], 'test-site')
        self.assertEqual(log['requester'], '@foo:test')
        self.assertEqual(log['authenticated_entity'], '@foo:test')
        self.assertEqual(log['method'], 'POST')
        self.assertEqual(log['url'], '/_matrix/client/versions')
        self.assertEqual(log['protocol'], '1.1')
        self.assertEqual(log['user_agent'], '')

    def test_with_exception(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        The logging exception type & value should be added to the JSON response.\n        '
        handler = logging.StreamHandler(self.output)
        handler.setFormatter(JsonFormatter())
        logger = self.get_logger(handler)
        try:
            raise ValueError("That's wrong, you wally!")
        except ValueError:
            logger.exception('Hello there, %s!', 'wally')
        log = self.get_log_line()
        expected_log_keys = ['log', 'level', 'namespace', 'exc_type', 'exc_value']
        self.assertCountEqual(log.keys(), expected_log_keys)
        self.assertEqual(log['log'], 'Hello there, wally!')
        self.assertEqual(log['exc_type'], 'ValueError')
        self.assertEqual(log['exc_value'], "That's wrong, you wally!")