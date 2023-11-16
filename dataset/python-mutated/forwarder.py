"""
This module contains utilities to call a backend (e.g., an external service process like
DynamoDBLocal) from a service provider.
"""
from typing import Any, Callable, Mapping, Optional, Union
from urllib.parse import urlsplit
from botocore.awsrequest import AWSPreparedRequest, prepare_request_dict
from botocore.config import Config as BotoConfig
from werkzeug.datastructures import Headers
from localstack.aws.api.core import Request, RequestContext, ServiceRequest, ServiceRequestHandler, ServiceResponse
from localstack.aws.client import parse_response, raise_service_exception
from localstack.aws.connect import connect_to
from localstack.aws.skeleton import DispatchTable, create_dispatch_table
from localstack.aws.spec import load_service
from localstack.constants import AWS_REGION_US_EAST_1
from localstack.http import Response
from localstack.http.proxy import Proxy
from localstack.utils.strings import to_str

class AwsRequestProxy:
    """
    Implements the ``ServiceRequestHandler`` protocol to forward AWS requests to a backend. It is stateful and uses a
    ``Proxy`` instance for re-using client connections to the backend.
    """

    def __init__(self, endpoint_url: str, parse_response: bool=True, include_response_metadata: bool=False):
        if False:
            return 10
        '\n        Create a new AwsRequestProxy. ``parse_response`` control the return behavior of ``forward``. If\n        ``parse_response`` is set, then ``forward`` parses the HTTP response from the backend and returns a\n        ``ServiceResponse``, otherwise it returns the raw HTTP ``Response`` object.\n\n        :param endpoint_url: the backend to proxy the requests to, used as ``forward_base_url`` for the ``Proxy``.\n        :param parse_response: whether to parse the response before returning it\n        :param include_response_metadata: include AWS response metadata, only used with ``parse_response=True``\n        '
        self.endpoint_url = endpoint_url
        self.parse_response = parse_response
        self.include_response_metadata = include_response_metadata
        self.proxy = Proxy(forward_base_url=endpoint_url)

    def __call__(self, context: RequestContext, service_request: ServiceRequest=None) -> Optional[Union[ServiceResponse, Response]]:
        if False:
            print('Hello World!')
        'Method to satisfy the ``ServiceRequestHandler`` protocol.'
        return self.forward(context, service_request)

    def forward(self, context: RequestContext, service_request: ServiceRequest=None) -> Optional[Union[ServiceResponse, Response]]:
        if False:
            return 10
        '\n        Forwards the given request to the backend configured by ``endpoint_url``.\n\n        :param context: the original request context of the incoming request\n        :param service_request: optionally a new service\n        :return:\n        '
        if service_request is not None:
            context = self.new_request_context(context, service_request)
        http_response = self.proxy.forward(context.request, forward_path=context.request.path)
        if not self.parse_response:
            return http_response
        parsed_response = parse_response(context.operation, http_response, self.include_response_metadata)
        raise_service_exception(http_response, parsed_response)
        return parsed_response

    def new_request_context(self, original: RequestContext, service_request: ServiceRequest):
        if False:
            for i in range(10):
                print('nop')
        context = create_aws_request_context(service_name=original.service.service_name, action=original.operation.name, parameters=service_request, region=original.region)
        headers = Headers(original.request.headers)
        headers.pop('Content-Type', None)
        headers.pop('Content-Length', None)
        context.request.headers.update(headers)
        return context

def ForwardingFallbackDispatcher(provider: object, request_forwarder: ServiceRequestHandler) -> DispatchTable:
    if False:
        for i in range(10):
            print('nop')
    '\n    Wraps a provider with a request forwarder. It does by creating a new DispatchTable from the original\n    provider, and wrapping each method with a fallthrough method that calls ``request_forwarder`` if the\n    original provider raises a ``NotImplementedError``.\n\n    :param provider: the ASF provider\n    :param request_forwarder: callable that forwards the request (e.g., to a backend server)\n    :return: a modified DispatchTable\n    '
    table = create_dispatch_table(provider)
    for (op, fn) in table.items():
        table[op] = _wrap_with_fallthrough(fn, request_forwarder)
    return table

class NotImplementedAvoidFallbackError(NotImplementedError):
    pass

def _wrap_with_fallthrough(handler: ServiceRequestHandler, fallthrough_handler: ServiceRequestHandler) -> ServiceRequestHandler:
    if False:
        print('Hello World!')

    def _call(context, req) -> ServiceResponse:
        if False:
            return 10
        try:
            return handler(context, req)
        except NotImplementedAvoidFallbackError as e:
            raise e
        except NotImplementedError:
            pass
        return fallthrough_handler(context, req)
    return _call

def HttpFallbackDispatcher(provider: object, forward_url_getter: Callable[[str, str], str]):
    if False:
        return 10
    return ForwardingFallbackDispatcher(provider, get_request_forwarder_http(forward_url_getter))

def get_request_forwarder_http(forward_url_getter: Callable[[str, str], str]) -> ServiceRequestHandler:
    if False:
        i = 10
        return i + 15
    '\n    Returns a ServiceRequestHandler that creates for each invocation a new AwsRequestProxy with the result of\n    forward_url_getter. Note that this is an inefficient method of proxying, since for every call a new client\n    connection has to be established. Try to instead use static forward URL values and use ``AwsRequestProxy`` directly.\n\n    :param forward_url_getter: a factory method for returning forward base urls for the proxy\n    :return: a ServiceRequestHandler acting as a proxy\n    '

    def _forward_request(context: RequestContext, service_request: ServiceRequest=None) -> ServiceResponse:
        if False:
            i = 10
            return i + 15
        return AwsRequestProxy(forward_url_getter(context.account_id, context.region)).forward(context, service_request)
    return _forward_request

def dispatch_to_backend(context: RequestContext, http_request_dispatcher: Callable[[RequestContext], Response], include_response_metadata=False) -> ServiceResponse:
    if False:
        for i in range(10):
            print('nop')
    '\n    Dispatch the given request to a backend by using the `request_forwarder` function to\n    fetch an HTTP response, converting it to a ServiceResponse.\n    :param context: the request context\n    :param http_request_dispatcher: dispatcher that performs the request and returns an HTTP response\n    :param include_response_metadata: whether to include boto3 response metadata in the response\n    :return: parsed service response\n    :raises ServiceException: if the dispatcher returned an error response\n    '
    http_response = http_request_dispatcher(context)
    parsed_response = parse_response(context.operation, http_response, include_response_metadata)
    raise_service_exception(http_response, parsed_response)
    return parsed_response
_non_validating_boto_config = BotoConfig(parameter_validation=False)

def create_aws_request_context(service_name: str, action: str, parameters: Mapping[str, Any]=None, region: str=None, endpoint_url: Optional[str]=None) -> RequestContext:
    if False:
        for i in range(10):
            print('nop')
    '\n    This is a stripped-down version of what the botocore client does to perform an HTTP request from a client call. A\n    client call looks something like this: boto3.client("sqs").create_queue(QueueName="myqueue"), which will be\n    serialized into an HTTP request. This method does the same, without performing the actual request, and with a\n    more low-level interface. An equivalent call would be\n\n         create_aws_request_context("sqs", "CreateQueue", {"QueueName": "myqueue"})\n\n    :param service_name: the AWS service\n    :param action: the action to invoke\n    :param parameters: the invocation parameters\n    :param region: the region name (default is us-east-1)\n    :param endpoint_url: the endpoint to call (defaults to localstack)\n    :return: a RequestContext object that describes this request\n    '
    if parameters is None:
        parameters = {}
    if region is None:
        region = AWS_REGION_US_EAST_1
    service = load_service(service_name)
    operation = service.operation_model(action)
    client = connect_to.get_client(service_name, endpoint_url=endpoint_url, region_name=region, config=_non_validating_boto_config)
    request_context = {'client_region': region, 'has_streaming_input': operation.has_streaming_input, 'auth_type': operation.auth_type}
    if not endpoint_url:
        endpoint_url = 'http://localhost.localstack.cloud'
    parameters = client._emit_api_params(parameters, operation, request_context)
    request_dict = client._convert_to_request_dict(parameters, operation, endpoint_url, context=request_context)
    if (auth_path := request_dict.get('auth_path')):
        (path, sep, query) = request_dict['url_path'].partition('?')
        request_dict['url_path'] = f'{auth_path}{sep}{query}'
        prepare_request_dict(request_dict, endpoint_url=endpoint_url, user_agent=client._client_config.user_agent, context=request_context)
    aws_request: AWSPreparedRequest = client._endpoint.create_request(request_dict, operation)
    context = RequestContext()
    context.service = service
    context.operation = operation
    context.region = region
    context.request = create_http_request(aws_request)
    context.service_request = parameters
    return context

def create_http_request(aws_request: AWSPreparedRequest) -> Request:
    if False:
        for i in range(10):
            print('nop')
    split_url = urlsplit(aws_request.url)
    host = split_url.netloc.split(':')
    if len(host) == 1:
        server = (to_str(host[0]), None)
    elif len(host) == 2:
        server = (to_str(host[0]), int(host[1]))
    else:
        raise ValueError
    headers = Headers()
    for (k, v) in aws_request.headers.items():
        headers[k] = to_str(v, 'latin-1')
    return Request(method=aws_request.method, path=split_url.path, query_string=split_url.query, headers=headers, body=aws_request.body, server=server)