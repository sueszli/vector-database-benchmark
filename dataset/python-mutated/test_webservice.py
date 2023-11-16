import sys
from unittest.mock import MagicMock, patch
from PyQt6.QtCore import QUrl
from PyQt6.QtNetwork import QNetworkProxy, QNetworkRequest
from test.picardtestcase import PicardTestCase
from picard import config
from picard.webservice import TEMP_ERRORS_RETRIES, RequestPriorityQueue, RequestTask, UnknownResponseParserError, WebService, WSRequest, ratecontrol
from picard.webservice.utils import host_port_to_url, hostkey_from_url, port_from_qurl
PROXY_SETTINGS = {'use_proxy': True, 'proxy_type': 'http', 'proxy_server_host': '127.0.0.1', 'proxy_server_port': 3128, 'proxy_username': 'user', 'proxy_password': 'password', 'network_transfer_timeout_seconds': 30}

def dummy_handler(*args, **kwargs):
    if False:
        return 10
    'Dummy handler method for tests'

class WebServiceTest(PicardTestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.set_config_values({'use_proxy': False, 'server_host': '', 'network_transfer_timeout_seconds': 30})
        self.ws = WebService()

    @patch.object(WebService, 'add_task')
    def test_webservice_method_calls(self, mock_add_task):
        if False:
            print('Hello World!')
        host = 'abc.xyz'
        port = 80
        path = ''
        handler = dummy_handler
        data = None

        def get_wsreq(mock_add_task):
            if False:
                return 10
            return mock_add_task.call_args[0][1]
        self.ws.get(host, port, path, handler)
        self.assertEqual(1, mock_add_task.call_count)
        self.assertEqual(host, get_wsreq(mock_add_task).host)
        self.assertEqual(port, get_wsreq(mock_add_task).port)
        self.assertIn('GET', get_wsreq(mock_add_task).method)
        self.ws.post(host, port, path, data, handler)
        self.assertIn('POST', get_wsreq(mock_add_task).method)
        self.ws.put(host, port, path, data, handler)
        self.assertIn('PUT', get_wsreq(mock_add_task).method)
        self.ws.delete(host, port, path, handler)
        self.assertIn('DELETE', get_wsreq(mock_add_task).method)
        self.ws.download(host, port, path, handler)
        self.assertIn('GET', get_wsreq(mock_add_task).method)
        self.assertEqual(5, mock_add_task.call_count)

    @patch.object(WebService, 'add_task')
    def test_webservice_url_method_calls(self, mock_add_task):
        if False:
            print('Hello World!')
        url = 'http://abc.xyz'
        handler = dummy_handler
        data = None

        def get_wsreq(mock_add_task):
            if False:
                return 10
            return mock_add_task.call_args[0][1]
        self.ws.get_url(url=url, handler=handler)
        self.assertEqual(1, mock_add_task.call_count)
        self.assertEqual('abc.xyz', get_wsreq(mock_add_task).host)
        self.assertEqual(80, get_wsreq(mock_add_task).port)
        self.assertIn('GET', get_wsreq(mock_add_task).method)
        self.ws.post_url(url=url, data=data, handler=handler)
        self.assertIn('POST', get_wsreq(mock_add_task).method)
        self.ws.put_url(url=url, data=data, handler=handler)
        self.assertIn('PUT', get_wsreq(mock_add_task).method)
        self.ws.delete_url(url=url, handler=handler)
        self.assertIn('DELETE', get_wsreq(mock_add_task).method)
        self.ws.download_url(url=url, handler=handler)
        self.assertIn('GET', get_wsreq(mock_add_task).method)
        self.assertEqual(5, mock_add_task.call_count)

class WebServiceTaskTest(PicardTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.set_config_values({'use_proxy': False, 'network_transfer_timeout_seconds': 30})
        self.ws = WebService()
        self.queue = self.ws._queue = MagicMock()
        self.ws._timer_run_next_task = MagicMock()
        self.ws._timer_count_pending_requests = MagicMock()

    def test_add_task(self):
        if False:
            while True:
                i = 10
        request = WSRequest(method='GET', url='http://abc.xyz', handler=dummy_handler)
        func = 1
        task = self.ws.add_task(func, request)
        self.assertEqual((request.get_host_key(), func, 0), task)
        self.ws._queue.add_task.assert_called_with(task, False)
        request.important = True
        task = self.ws.add_task(func, request)
        self.ws._queue.add_task.assert_called_with(task, True)

    def test_add_task_calls_timers(self):
        if False:
            i = 10
            return i + 15
        mock_timer1 = self.ws._timer_run_next_task
        mock_timer2 = self.ws._timer_count_pending_requests
        request = WSRequest(method='GET', url='http://abc.xyz', handler=dummy_handler)
        self.ws.add_task(0, request)
        mock_timer1.start.assert_not_called()
        mock_timer2.start.assert_not_called()
        mock_timer1.isActive.return_value = False
        mock_timer2.isActive.return_value = False
        self.ws.add_task(0, request)
        mock_timer1.start.assert_called_with(0)
        mock_timer2.start.assert_called_with(0)

    def test_remove_task(self):
        if False:
            i = 10
            return i + 15
        task = RequestTask(('example.com', 80), dummy_handler, priority=0)
        self.ws.remove_task(task)
        self.ws._queue.remove_task.assert_called_with(task)

    def test_remove_task_calls_timers(self):
        if False:
            for i in range(10):
                print('nop')
        mock_timer = self.ws._timer_count_pending_requests
        task = RequestTask(('example.com', 80), dummy_handler, priority=0)
        self.ws.remove_task(task)
        mock_timer.start.assert_not_called()
        mock_timer.isActive.return_value = False
        self.ws.remove_task(task)
        mock_timer.start.assert_called_with(0)

    def test_run_next_task(self):
        if False:
            i = 10
            return i + 15
        mock_timer = self.ws._timer_run_next_task
        self.ws._queue.run_ready_tasks.return_value = sys.maxsize
        self.ws._run_next_task()
        self.ws._queue.run_ready_tasks.assert_called()
        mock_timer.start.assert_not_called()

    def test_run_next_task_starts_next(self):
        if False:
            print('Hello World!')
        mock_timer = self.ws._timer_run_next_task
        delay = 42
        self.ws._queue.run_ready_tasks.return_value = delay
        self.ws._run_next_task()
        self.ws._queue.run_ready_tasks.assert_called()
        mock_timer.start.assert_called_with(42)

class RequestTaskTest(PicardTestCase):

    def test_from_request(self):
        if False:
            print('Hello World!')
        request = WSRequest(method='GET', url='https://example.com', handler=dummy_handler, priority=True)
        func = 1
        task = RequestTask.from_request(request, func)
        self.assertEqual(request.get_host_key(), task.hostkey)
        self.assertEqual(func, task.func)
        self.assertEqual(1, task.priority)
        self.assertEqual((request.get_host_key(), func, 1), task)

class RequestPriorityQueueTest(PicardTestCase):

    def test_add_task(self):
        if False:
            print('Hello World!')
        queue = RequestPriorityQueue(ratecontrol)
        key = ('abc.xyz', 80)
        task1 = RequestTask(key, dummy_handler, priority=0)
        queue.add_task(task1)
        task2 = RequestTask(key, dummy_handler, priority=1)
        queue.add_task(task2)
        task3 = RequestTask(key, dummy_handler, priority=0)
        queue.add_task(task3, important=True)
        task4 = RequestTask(key, dummy_handler, priority=1)
        queue.add_task(task4, important=True)
        self.assertEqual(len(queue._queues[0][key]), 2)
        self.assertEqual(len(queue._queues[1][key]), 2)
        self.assertEqual(queue._queues[0][key][0], task3.func)
        self.assertEqual(queue._queues[0][key][1], task1.func)
        self.assertEqual(queue._queues[1][key][0], task4.func)
        self.assertEqual(queue._queues[1][key][1], task2.func)

    def test_remove_task(self):
        if False:
            i = 10
            return i + 15
        queue = RequestPriorityQueue(ratecontrol)
        key = ('abc.xyz', 80)
        task = RequestTask(key, dummy_handler, priority=0)
        task = queue.add_task(task)
        self.assertIn(key, queue._queues[0])
        self.assertEqual(len(queue._queues[0][key]), 1)
        queue.remove_task(task)
        self.assertIn(key, queue._queues[0])
        self.assertEqual(len(queue._queues[0][key]), 0)
        non_existing_task = (1, 'a', 'b')
        queue.remove_task(non_existing_task)

    def test_run_task(self):
        if False:
            return 10
        mock_ratecontrol = MagicMock()
        delay_func = mock_ratecontrol.get_delay_to_next_request = MagicMock()
        queue = RequestPriorityQueue(mock_ratecontrol)
        key = ('abc.xyz', 80)
        delay_func.side_effect = [(False, 0), (True, 0), (False, 0), (False, 0), (False, 0), (False, 0)]
        func1 = MagicMock()
        task1 = RequestTask(key, func1, priority=0)
        queue.add_task(task1)
        func2 = MagicMock()
        task2 = RequestTask(key, func2, priority=1)
        queue.add_task(task2)
        task3 = RequestTask(key, func1, priority=0)
        queue.add_task(task3)
        task4 = RequestTask(key, func1, priority=0)
        queue.add_task(task4)
        self.assertEqual(func1.call_count, 0)
        queue.run_ready_tasks()
        self.assertEqual(func2.call_count, 1)
        self.assertEqual(func1.call_count, 0)
        self.assertIn(key, queue._queues[1])
        queue.run_ready_tasks()
        self.assertEqual(func1.call_count, 1)
        self.assertNotIn(key, queue._queues[1])
        queue.run_ready_tasks()
        self.assertEqual(func1.call_count, 2)
        queue.run_ready_tasks()
        self.assertEqual(func1.call_count, 3)
        queue.run_ready_tasks()
        self.assertEqual(func1.call_count, 3)
        self.assertNotIn(key, queue._queues[0])

    def test_count(self):
        if False:
            return 10
        queue = RequestPriorityQueue(ratecontrol)
        key = ('abc.xyz', 80)
        self.assertEqual(0, queue.count())
        task1 = RequestTask(key, dummy_handler, priority=0)
        queue.add_task(task1)
        self.assertEqual(1, queue.count())
        task2 = RequestTask(key, dummy_handler, priority=1)
        queue.add_task(task2)
        self.assertEqual(2, queue.count())
        task3 = RequestTask(key, dummy_handler, priority=0)
        queue.add_task(task3, important=True)
        self.assertEqual(3, queue.count())
        task4 = RequestTask(key, dummy_handler, priority=1)
        queue.add_task(task4, important=True)
        self.assertEqual(4, queue.count())
        queue.remove_task(task1)
        self.assertEqual(3, queue.count())
        queue.remove_task(task2)
        self.assertEqual(2, queue.count())
        queue.remove_task(task3)
        self.assertEqual(1, queue.count())
        queue.remove_task(task4)
        self.assertEqual(0, queue.count())

class WebServiceProxyTest(PicardTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.set_config_values(PROXY_SETTINGS)

    def test_proxy_setup(self):
        if False:
            i = 10
            return i + 15
        proxy_types = [('http', QNetworkProxy.ProxyType.HttpProxy), ('socks', QNetworkProxy.ProxyType.Socks5Proxy)]
        for (proxy_type, expected_qt_type) in proxy_types:
            config.setting['proxy_type'] = proxy_type
            ws = WebService()
            proxy = ws.manager.proxy()
            self.assertEqual(proxy.type(), expected_qt_type)
            self.assertEqual(proxy.user(), PROXY_SETTINGS['proxy_username'])
            self.assertEqual(proxy.password(), PROXY_SETTINGS['proxy_password'])
            self.assertEqual(proxy.hostName(), PROXY_SETTINGS['proxy_server_host'])
            self.assertEqual(proxy.port(), PROXY_SETTINGS['proxy_server_port'])

class ParserHookTest(PicardTestCase):

    def test_parser_hook(self):
        if False:
            while True:
                i = 10
        WebService.add_parser('A', 'mime', 'parser')
        self.assertIn('A', WebService.PARSERS)
        self.assertEqual(WebService.PARSERS['A'].mimetype, 'mime')
        self.assertEqual(WebService.PARSERS['A'].mimetype, WebService.get_response_mimetype('A'))
        self.assertEqual(WebService.PARSERS['A'].parser, 'parser')
        self.assertEqual(WebService.PARSERS['A'].parser, WebService.get_response_parser('A'))
        with self.assertRaises(UnknownResponseParserError):
            WebService.get_response_parser('B')
        with self.assertRaises(UnknownResponseParserError):
            WebService.get_response_mimetype('B')

class WSRequestTest(PicardTestCase):

    def test_init_minimal(self):
        if False:
            for i in range(10):
                print('nop')
        request = WSRequest(url='https://example.org/path', method='GET', handler=dummy_handler)
        self.assertEqual(request.host, 'example.org')
        self.assertEqual(request.port, 443)
        self.assertEqual(request.path, '/path')
        self.assertEqual(request.handler, dummy_handler)
        self.assertEqual(request.method, 'GET')
        self.assertEqual(request.get_host_key(), ('example.org', 443))
        self.assertIsNone(request.parse_response_type)
        self.assertIsNone(request.data)
        self.assertIsNone(request.cacheloadcontrol)
        self.assertIsNone(request.request_mimetype)
        self.assertFalse(request.mblogin)
        self.assertFalse(request.refresh)
        self.assertFalse(request.priority)
        self.assertFalse(request.important)
        self.assertFalse(request.has_auth)

    def test_init_minimal_extra(self):
        if False:
            return 10
        request = WSRequest(url='https://example.org/path', method='GET', handler=dummy_handler, priority=True, important=True, refresh=True)
        self.assertTrue(request.priority)
        self.assertTrue(request.important)
        self.assertTrue(request.refresh)

    def test_init_minimal_qurl(self):
        if False:
            print('Hello World!')
        url = 'https://example.org/path?q=1'
        request = WSRequest(url=QUrl(url), method='GET', handler=dummy_handler)
        self.assertEqual(request.url().toString(), url)

    def test_init_port_80(self):
        if False:
            for i in range(10):
                print('nop')
        request = WSRequest(url='http://example.org/path', method='GET', handler=dummy_handler)
        self.assertEqual(request.port, 80)

    def test_init_port_other(self):
        if False:
            while True:
                i = 10
        request = WSRequest(url='http://example.org:666/path', method='GET', handler=dummy_handler)
        self.assertEqual(request.port, 666)

    def test_missing_url(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(AssertionError):
            WSRequest(method='GET', handler=dummy_handler)

    def test_missing_method(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(AssertionError):
            WSRequest(url='http://x', handler=dummy_handler)

    def test_missing_handler(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(AssertionError):
            WSRequest(url='http://x', method='GET')

    def test_invalid_method(self):
        if False:
            return 10
        with self.assertRaises(AssertionError):
            WSRequest(url='http://x', method='XXX', handler=dummy_handler)

    def test_set_cacheloadcontrol(self):
        if False:
            return 10
        request = WSRequest(url='http://example.org/path', method='GET', handler=dummy_handler, cacheloadcontrol=QNetworkRequest.CacheLoadControl.AlwaysNetwork)
        self.assertEqual(request.cacheloadcontrol, QNetworkRequest.CacheLoadControl.AlwaysNetwork)

    def test_set_parse_response_type(self):
        if False:
            i = 10
            return i + 15
        WebService.add_parser('A', 'mime', 'parser')
        request = WSRequest(url='http://example.org/path', method='GET', handler=dummy_handler, parse_response_type='A')
        self.assertEqual(request.response_mimetype, 'mime')
        self.assertEqual(request.response_parser, 'parser')

    def test_set_invalid_parse_response_type(self):
        if False:
            while True:
                i = 10
        WebService.add_parser('A', 'mime', 'parser')
        request = WSRequest(url='http://example.org/path', method='GET', handler=dummy_handler, parse_response_type='invalid')
        self.assertEqual(request.response_mimetype, None)
        self.assertEqual(request.response_parser, None)

    def test_set_mblogin_access_token(self):
        if False:
            i = 10
            return i + 15
        request = WSRequest(url='http://example.org/path', method='POST', handler=dummy_handler)
        request.mblogin = 'test'
        self.assertEqual(request.mblogin, 'test')
        self.assertFalse(request.has_auth)
        request.access_token = 'token'
        self.assertEqual(request.access_token, 'token')
        self.assertTrue(request.has_auth)

    def test_set_data(self):
        if False:
            while True:
                i = 10
        request = WSRequest(url='http://example.org/path', method='POST', handler=dummy_handler, data='data')
        self.assertEqual(request.data, 'data')

    def test_set_retries_reached(self):
        if False:
            for i in range(10):
                print('nop')
        request = WSRequest(url='http://example.org/path', method='GET', handler=dummy_handler)
        for i in range(0, TEMP_ERRORS_RETRIES):
            self.assertEqual(request.mark_for_retry(), i + 1)
        self.assertTrue(request.max_retries_reached())

    def test_set_retries_not_reached(self):
        if False:
            while True:
                i = 10
        request = WSRequest(url='http://example.org/path', method='GET', handler=dummy_handler)
        self.assertTrue(TEMP_ERRORS_RETRIES > 1)
        self.assertEqual(request.mark_for_retry(), 1)
        self.assertFalse(request.max_retries_reached())

    def test_queryargs(self):
        if False:
            i = 10
            return i + 15
        request = WSRequest(url='http://example.org/path?a=1', method='GET', handler=dummy_handler, queryargs={'a': 2, 'b': 'x%20x', 'c': '1+2', 'd': '&', 'e': '?'})
        expected = 'http://example.org/path?a=1&a=2&b=x x&c=1+2&d=%26&e=?'
        self.assertEqual(request.url().toString(), expected)

    def test_unencoded_queryargs(self):
        if False:
            i = 10
            return i + 15
        request = WSRequest(url='http://example.org/path?a=1', method='GET', handler=dummy_handler, unencoded_queryargs={'a': 2, 'b': 'x%20x', 'c': '1+2', 'd': '&', 'e': '?'})
        expected = 'http://example.org/path?a=1&a=2&b=x%2520x&c=1%2B2&d=%26&e=%3F'
        self.assertEqual(request.url().toString(), expected)

    def test_mixed_queryargs(self):
        if False:
            for i in range(10):
                print('nop')
        request = WSRequest(url='http://example.org/path?a=1', method='GET', handler=dummy_handler, queryargs={'a': '2&', 'b': '1&', 'c': '&'}, unencoded_queryargs={'a': '1&', 'b': '2&', 'd': '&'})
        expected = 'http://example.org/path?a=1&a=1%26&b=2%26&c=%26&d=%26'
        self.assertEqual(request.url().toString(), expected)

class WebServiceUtilsTest(PicardTestCase):

    def test_port_from_qurl_http(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(port_from_qurl(QUrl('http://example.org')), 80)

    def test_port_from_qurl_http_other(self):
        if False:
            print('Hello World!')
        self.assertEqual(port_from_qurl(QUrl('http://example.org:666')), 666)

    def test_port_from_qurl_https(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(port_from_qurl(QUrl('https://example.org')), 443)

    def test_port_from_qurl_https_other(self):
        if False:
            while True:
                i = 10
        self.assertEqual(port_from_qurl(QUrl('https://example.org:666')), 666)

    def test_port_from_qurl_exception(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(AttributeError):
            port_from_qurl('xxx')

    def test_hostkey_from_qurl_http(self):
        if False:
            while True:
                i = 10
        self.assertEqual(hostkey_from_url(QUrl('http://example.org')), ('example.org', 80))

    def test_hostkey_from_url_https_other(self):
        if False:
            print('Hello World!')
        self.assertEqual(hostkey_from_url('https://example.org:666'), ('example.org', 666))

    def test_host_port_to_url_http_80(self):
        if False:
            return 10
        self.assertEqual(host_port_to_url('example.org', 80, as_string=True), 'http://example.org')

    def test_host_port_to_url_http_80_qurl(self):
        if False:
            while True:
                i = 10
        self.assertEqual(host_port_to_url('example.org', 80).toString(), 'http://example.org')

    def test_host_port_to_url_https_443(self):
        if False:
            print('Hello World!')
        self.assertEqual(host_port_to_url('example.org', 443, as_string=True), 'https://example.org')

    def test_host_port_to_url_https_scheme_80(self):
        if False:
            return 10
        self.assertEqual(host_port_to_url('example.org', 80, scheme='https', as_string=True), 'https://example.org:80')

    def test_host_port_to_url_http_666_with_path(self):
        if False:
            while True:
                i = 10
        self.assertEqual(host_port_to_url('example.org', 666, path='/abc', as_string=True), 'http://example.org:666/abc')