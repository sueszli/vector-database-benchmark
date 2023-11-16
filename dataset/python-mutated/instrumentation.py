from webob import Request
from st2common import log as logging
from st2common.metrics.base import CounterWithTimer
from st2common.metrics.base import get_driver
from st2common.util.date import get_datetime_utc_now
from st2common.router import NotFoundException
from st2common.router import Response
from st2common.util.jsonify import json_encode
__all__ = ['RequestInstrumentationMiddleware', 'ResponseInstrumentationMiddleware']
LOG = logging.getLogger(__name__)

class RequestInstrumentationMiddleware(object):
    """
    Instrumentation middleware which records various request related metrics.
    """

    def __init__(self, app, router, service_name):
        if False:
            print('Hello World!')
        '\n        :param service_name: Service name (e.g. api, stream, auth).\n        :type service_name: ``str``\n        '
        self.app = app
        self.router = router
        self._service_name = service_name

    def __call__(self, environ, start_response):
        if False:
            while True:
                i = 10
        request = Request(environ)
        try:
            (endpoint, _) = self.router.match(request)
        except NotFoundException:
            endpoint = {}
        except Exception as e:
            status_code = getattr(e, 'status_code', 500)
            headers = {'Content-Type': 'application/json'}
            body = {'faultstring': getattr(e, 'detail', str(e))}
            response_body = json_encode(body)
            resp = Response(response_body, status=status_code, headers=headers)
            return resp(environ, start_response)
        submit_metrics = endpoint.get('x-submit-metrics', True)
        operation_id = endpoint.get('operationId', None)
        is_get_one_endpoint = bool(operation_id) and (operation_id.endswith('.get') or operation_id.endswith('.get_one'))
        if is_get_one_endpoint:
            submit_metrics = False
        if not submit_metrics:
            LOG.debug('Not submitting request metrics for path: %s' % request.path)
            return self.app(environ, start_response)
        metrics_driver = get_driver()
        key = '%s.request.total' % self._service_name
        metrics_driver.inc_counter(key)
        key = '%s.request.method.%s' % (self._service_name, request.method)
        metrics_driver.inc_counter(key)
        path = request.path.replace('/', '_')
        key = '%s.request.path.%s' % (self._service_name, path)
        metrics_driver.inc_counter(key)
        if self._service_name == 'stream':
            key = '%s.request' % self._service_name
            metrics_driver.inc_counter(key)
            metrics_driver.inc_gauge('stream.connections', 1)
            start_time = get_datetime_utc_now()

            def update_metrics_hook(env):
                if False:
                    print('Hello World!')
                time_delta = get_datetime_utc_now() - start_time
                duration = time_delta.total_seconds()
                metrics_driver.time(key, duration)
                metrics_driver.dec_gauge('stream.connections', 1)
            if 'eventlet.posthooks' in environ:
                environ['eventlet.posthooks'].append((update_metrics_hook, (), {}))
            return self.app(environ, start_response)
        else:
            key = '%s.request' % self._service_name
            with CounterWithTimer(key=key):
                return self.app(environ, start_response)

class ResponseInstrumentationMiddleware(object):
    """
    Instrumentation middleware which records various response related metrics.
    """

    def __init__(self, app, router, service_name):
        if False:
            i = 10
            return i + 15
        '\n        :param service_name: Service name (e.g. api, stream, auth).\n        :type service_name: ``str``\n        '
        self.app = app
        self.router = router
        self._service_name = service_name

    def __call__(self, environ, start_response):
        if False:
            for i in range(10):
                print('nop')

        def custom_start_response(status, headers, exc_info=None):
            if False:
                i = 10
                return i + 15
            status_code = int(status.split(' ')[0])
            metrics_driver = get_driver()
            metrics_driver.inc_counter('%s.response.status.%s' % (self._service_name, status_code))
            return start_response(status, headers, exc_info)
        return self.app(environ, custom_start_response)