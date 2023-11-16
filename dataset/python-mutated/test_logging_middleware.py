import mock
import unittest2
from oslo_config import cfg
from st2common.middleware.logging import LoggingMiddleware
from st2common.constants.secrets import MASKED_ATTRIBUTE_VALUE
__all__ = ['LoggingMiddlewareTestCase']

class LoggingMiddlewareTestCase(unittest2.TestCase):

    @mock.patch('st2common.middleware.logging.LOG')
    @mock.patch('st2common.middleware.logging.Request')
    def test_secret_parameters_are_masked_in_log_message(self, mock_request, mock_log):
        if False:
            print('Hello World!')

        def app(environ, custom_start_response):
            if False:
                while True:
                    i = 10
            custom_start_response(status='200 OK', headers=[('Content-Length', 100)])
            return [None]
        router = mock.Mock()
        endpoint = mock.Mock()
        router.match.return_value = (endpoint, None)
        middleware = LoggingMiddleware(app=app, router=router)
        cfg.CONF.set_override(group='log', name='mask_secrets_blacklist', override=['blacklisted_4', 'blacklisted_5'])
        environ = {}
        mock_request.return_value.GET.dict_of_lists.return_value = {'foo': 'bar', 'bar': 'baz', 'x-auth-token': 'secret', 'st2-api-key': 'secret', 'password': 'secret', 'st2_auth_token': 'secret', 'token': 'secret', 'blacklisted_4': 'super secret', 'blacklisted_5': 'super secret'}
        middleware(environ=environ, start_response=mock.Mock())
        expected_query = {'foo': 'bar', 'bar': 'baz', 'x-auth-token': MASKED_ATTRIBUTE_VALUE, 'st2-api-key': MASKED_ATTRIBUTE_VALUE, 'password': MASKED_ATTRIBUTE_VALUE, 'token': MASKED_ATTRIBUTE_VALUE, 'st2_auth_token': MASKED_ATTRIBUTE_VALUE, 'blacklisted_4': MASKED_ATTRIBUTE_VALUE, 'blacklisted_5': MASKED_ATTRIBUTE_VALUE}
        call_kwargs = mock_log.info.call_args_list[0][1]
        query = call_kwargs['extra']['query']
        self.assertEqual(query, expected_query)