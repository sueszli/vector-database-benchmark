import gzip
from unittest import skipUnless, TestCase
from prometheus_client import CollectorRegistry, Counter
from prometheus_client.exposition import CONTENT_TYPE_LATEST
try:
    import asyncio
    from asgiref.testing import ApplicationCommunicator
    from prometheus_client import make_asgi_app
    HAVE_ASYNCIO_AND_ASGI = True
except ImportError:
    HAVE_ASYNCIO_AND_ASGI = False

def setup_testing_defaults(scope):
    if False:
        while True:
            i = 10
    scope.update({'client': ('127.0.0.1', 32767), 'headers': [], 'http_version': '1.0', 'method': 'GET', 'path': '/', 'query_string': b'', 'scheme': 'http', 'server': ('127.0.0.1', 80), 'type': 'http'})

class ASGITest(TestCase):

    @skipUnless(HAVE_ASYNCIO_AND_ASGI, "Don't have asyncio/asgi installed.")
    def setUp(self):
        if False:
            while True:
                i = 10
        self.registry = CollectorRegistry()
        self.captured_status = None
        self.captured_headers = None
        self.scope = {}
        setup_testing_defaults(self.scope)
        self.communicator = None

    def tearDown(self):
        if False:
            print('Hello World!')
        if self.communicator:
            asyncio.get_event_loop().run_until_complete(self.communicator.wait())

    def seed_app(self, app):
        if False:
            i = 10
            return i + 15
        self.communicator = ApplicationCommunicator(app, self.scope)

    def send_input(self, payload):
        if False:
            i = 10
            return i + 15
        asyncio.get_event_loop().run_until_complete(self.communicator.send_input(payload))

    def send_default_request(self):
        if False:
            i = 10
            return i + 15
        self.send_input({'type': 'http.request', 'body': b''})

    def get_output(self):
        if False:
            print('Hello World!')
        output = asyncio.get_event_loop().run_until_complete(self.communicator.receive_output(0))
        return output

    def get_all_output(self):
        if False:
            return 10
        outputs = []
        while True:
            try:
                outputs.append(self.get_output())
            except asyncio.TimeoutError:
                break
        return outputs

    def get_all_response_headers(self):
        if False:
            for i in range(10):
                print('nop')
        outputs = self.get_all_output()
        response_start = next((o for o in outputs if o['type'] == 'http.response.start'))
        return response_start['headers']

    def get_response_header_value(self, header_name):
        if False:
            for i in range(10):
                print('nop')
        response_headers = self.get_all_response_headers()
        return next((value.decode('utf-8') for (name, value) in response_headers if name.decode('utf-8') == header_name))

    def increment_metrics(self, metric_name, help_text, increments):
        if False:
            i = 10
            return i + 15
        c = Counter(metric_name, help_text, registry=self.registry)
        for _ in range(increments):
            c.inc()

    def assert_outputs(self, outputs, metric_name, help_text, increments, compressed):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(len(outputs), 2)
        response_start = outputs[0]
        self.assertEqual(response_start['type'], 'http.response.start')
        response_body = outputs[1]
        self.assertEqual(response_body['type'], 'http.response.body')
        self.assertEqual(response_start['status'], 200)
        num_of_headers = 2 if compressed else 1
        self.assertEqual(len(response_start['headers']), num_of_headers)
        self.assertIn((b'Content-Type', CONTENT_TYPE_LATEST.encode('utf8')), response_start['headers'])
        if compressed:
            self.assertIn((b'Content-Encoding', b'gzip'), response_start['headers'])
        if compressed:
            output = gzip.decompress(response_body['body']).decode('utf8')
        else:
            output = response_body['body'].decode('utf8')
        self.assertIn('# HELP ' + metric_name + '_total ' + help_text + '\n', output)
        self.assertIn('# TYPE ' + metric_name + '_total counter\n', output)
        self.assertIn(metric_name + '_total ' + str(increments) + '.0\n', output)

    def validate_metrics(self, metric_name, help_text, increments):
        if False:
            return 10
        '\n        ASGI app serves the metrics from the provided registry.\n        '
        self.increment_metrics(metric_name, help_text, increments)
        app = make_asgi_app(self.registry)
        self.seed_app(app)
        self.send_default_request()
        outputs = self.get_all_output()
        self.assert_outputs(outputs, metric_name, help_text, increments, compressed=False)

    def test_report_metrics_1(self):
        if False:
            while True:
                i = 10
        self.validate_metrics('counter', 'A counter', 2)

    def test_report_metrics_2(self):
        if False:
            i = 10
            return i + 15
        self.validate_metrics('counter', 'Another counter', 3)

    def test_report_metrics_3(self):
        if False:
            i = 10
            return i + 15
        self.validate_metrics('requests', 'Number of requests', 5)

    def test_report_metrics_4(self):
        if False:
            i = 10
            return i + 15
        self.validate_metrics('failed_requests', 'Number of failed requests', 7)

    def test_gzip(self):
        if False:
            while True:
                i = 10
        metric_name = 'counter'
        help_text = 'A counter'
        increments = 2
        self.increment_metrics(metric_name, help_text, increments)
        app = make_asgi_app(self.registry)
        self.seed_app(app)
        self.scope['headers'] = [(b'accept-encoding', b'gzip')]
        self.send_input({'type': 'http.request', 'body': b''})
        outputs = self.get_all_output()
        self.assert_outputs(outputs, metric_name, help_text, increments, compressed=True)

    def test_gzip_disabled(self):
        if False:
            while True:
                i = 10
        metric_name = 'counter'
        help_text = 'A counter'
        increments = 2
        self.increment_metrics(metric_name, help_text, increments)
        app = make_asgi_app(self.registry, disable_compression=True)
        self.seed_app(app)
        self.scope['headers'] = [(b'accept-encoding', b'gzip')]
        self.send_input({'type': 'http.request', 'body': b''})
        outputs = self.get_all_output()
        self.assert_outputs(outputs, metric_name, help_text, increments, compressed=False)

    def test_openmetrics_encoding(self):
        if False:
            for i in range(10):
                print('nop')
        'Response content type is application/openmetrics-text when appropriate Accept header is in request'
        app = make_asgi_app(self.registry)
        self.seed_app(app)
        self.scope['headers'] = [(b'Accept', b'application/openmetrics-text')]
        self.send_input({'type': 'http.request', 'body': b''})
        content_type = self.get_response_header_value('Content-Type').split(';')[0]
        assert content_type == 'application/openmetrics-text'

    def test_plaintext_encoding(self):
        if False:
            print('Hello World!')
        'Response content type is text/plain when Accept header is missing in request'
        app = make_asgi_app(self.registry)
        self.seed_app(app)
        self.send_input({'type': 'http.request', 'body': b''})
        content_type = self.get_response_header_value('Content-Type').split(';')[0]
        assert content_type == 'text/plain'