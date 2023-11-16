import base64
import sys
import unittest
from dataclasses import dataclass
from typing import Tuple
from typing import Union
import urllib3
from apache_beam.io.requestresponseio import Caller
from apache_beam.io.requestresponseio import UserCodeExecutionException
from apache_beam.io.requestresponseio import UserCodeQuotaException
from apache_beam.options.pipeline_options import PipelineOptions
_HTTP_PATH = '/v1/echo'
_PAYLOAD = base64.b64encode(bytes('payload', 'utf-8'))
_HTTP_ENDPOINT_ADDRESS_FLAG = '--httpEndpointAddress'

class EchoITOptions(PipelineOptions):
    """Shared options for running integration tests on a deployed
      ``EchoServiceGrpc`` See https://github.com/apache/beam/tree/master/.test
      -infra/mock-apis#integration for details on how to acquire values
      required by ``EchoITOptions``.
      """

    @classmethod
    def _add_argparse_args(cls, parser) -> None:
        if False:
            return 10
        parser.add_argument(_HTTP_ENDPOINT_ADDRESS_FLAG, dest='http_endpoint_address', help='The HTTP address of the Echo API endpoint; must being with http(s)://')
        parser.add_argument('--neverExceedQuotaId', default='echo-should-never-exceed-quota', dest='never_exceed_quota_id', help='The ID for an allocated quota that should never exceed.')
        parser.add_argument('--shouldExceedQuotaId', default='echo-should-exceed-quota', dest='should_exceed_quota_id', help='The ID for an allocated quota that should exceed.')

@dataclass
class EchoRequest:
    id: str
    payload: bytes

@dataclass
class EchoResponse:
    id: str
    payload: bytes

class EchoHTTPCaller(Caller):
    """Implements ``Caller`` to call the ``EchoServiceGrpc``'s HTTP handler.
    The purpose of ``EchoHTTPCaller`` is to support integration tests.
    """

    def __init__(self, url: str):
        if False:
            while True:
                i = 10
        self.url = url + _HTTP_PATH

    def call(self, request: EchoRequest) -> EchoResponse:
        if False:
            i = 10
            return i + 15
        "Overrides ``Caller``'s call method invoking the\n        ``EchoServiceGrpc``'s HTTP handler with an ``EchoRequest``, returning\n        either a successful ``EchoResponse`` or throwing either a\n        ``UserCodeExecutionException``, ``UserCodeTimeoutException``,\n        or a ``UserCodeQuotaException``.\n        "
        try:
            resp = urllib3.request('POST', self.url, json={'id': request.id, 'payload': str(request.payload, 'utf-8')}, retries=False)
            if resp.status < 300:
                resp_body = resp.json()
                resp_id = resp_body['id']
                payload = resp_body['payload']
                return EchoResponse(id=resp_id, payload=bytes(payload, 'utf-8'))
            if resp.status == 429:
                raise UserCodeQuotaException(resp.reason)
            raise UserCodeExecutionException(resp.reason)
        except urllib3.exceptions.HTTPError as e:
            raise UserCodeExecutionException(e)

class EchoHTTPCallerTestIT(unittest.TestCase):
    options: Union[EchoITOptions, None] = None
    client: Union[EchoHTTPCaller, None] = None

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            return 10
        cls.options = EchoITOptions()
        http_endpoint_address = cls.options.http_endpoint_address
        if not http_endpoint_address or http_endpoint_address == '':
            raise unittest.SkipTest(f'{_HTTP_ENDPOINT_ADDRESS_FLAG} is required.')
        cls.client = EchoHTTPCaller(http_endpoint_address)

    def setUp(self) -> None:
        if False:
            return 10
        (client, options) = EchoHTTPCallerTestIT._get_client_and_options()
        req = EchoRequest(id=options.should_exceed_quota_id, payload=_PAYLOAD)
        try:
            client.call(req)
            client.call(req)
            client.call(req)
        except UserCodeExecutionException as e:
            if not isinstance(e, UserCodeQuotaException):
                raise e

    @classmethod
    def _get_client_and_options(cls) -> Tuple[EchoHTTPCaller, EchoITOptions]:
        if False:
            for i in range(10):
                print('nop')
        assert cls.options is not None
        assert cls.client is not None
        return (cls.client, cls.options)

    def test_given_valid_request_receives_response(self):
        if False:
            while True:
                i = 10
        (client, options) = EchoHTTPCallerTestIT._get_client_and_options()
        req = EchoRequest(id=options.never_exceed_quota_id, payload=_PAYLOAD)
        response: EchoResponse = client.call(req)
        self.assertEqual(req.id, response.id)
        self.assertEqual(req.payload, response.payload)

    def test_given_exceeded_quota_should_raise(self):
        if False:
            return 10
        (client, options) = EchoHTTPCallerTestIT._get_client_and_options()
        req = EchoRequest(id=options.should_exceed_quota_id, payload=_PAYLOAD)
        self.assertRaises(UserCodeQuotaException, lambda : client.call(req))

    def test_not_found_should_raise(self):
        if False:
            for i in range(10):
                print('nop')
        (client, _) = EchoHTTPCallerTestIT._get_client_and_options()
        req = EchoRequest(id='i-dont-exist-quota-id', payload=_PAYLOAD)
        self.assertRaisesRegex(UserCodeExecutionException, 'Not Found', lambda : client.call(req))
if __name__ == '__main__':
    unittest.main(argv=sys.argv[:1])