"""
This module provides tools to call moto using moto and botocore internals without going through the moto HTTP server.
"""
import copy
import sys
from functools import lru_cache
from typing import Callable, Optional, Union
import moto.backends as moto_backends
from moto.core import BackendDict
from moto.core.exceptions import RESTError
from moto.moto_server.utilities import RegexConverter
from werkzeug.exceptions import NotFound
from werkzeug.routing import Map, Rule
from localstack import __version__ as localstack_version
from localstack import constants
from localstack.aws.api import CommonServiceException, HttpRequest, RequestContext, ServiceRequest, ServiceResponse
from localstack.aws.forwarder import ForwardingFallbackDispatcher, create_aws_request_context, dispatch_to_backend
from localstack.aws.skeleton import DispatchTable
from localstack.constants import DEFAULT_AWS_ACCOUNT_ID
from localstack.http import Response
from localstack.http.request import get_full_raw_path, get_raw_current_url
MotoDispatcher = Callable[[HttpRequest, str, dict], Response]
user_agent = f"Localstack/{localstack_version} Python/{sys.version.split(' ')[0]}"

def call_moto(context: RequestContext, include_response_metadata=False) -> ServiceResponse:
    if False:
        i = 10
        return i + 15
    '\n    Call moto with the given request context and receive a parsed ServiceResponse.\n\n    :param context: the request context\n    :param include_response_metadata: whether to include botocore\'s "ResponseMetadata" attribute\n    :return: a serialized AWS ServiceResponse (same as boto3 would return)\n    '
    return dispatch_to_backend(context, dispatch_to_moto, include_response_metadata)

def call_moto_with_request(context: RequestContext, service_request: ServiceRequest) -> ServiceResponse:
    if False:
        i = 10
        return i + 15
    '\n    Like `call_moto`, but you can pass a modified version of the service request before calling moto. The caveat is\n    that a new HTTP request has to be created. The service_request is serialized into a new RequestContext object,\n    and headers from the old request are merged into the new one.\n\n    :param context: the original request context\n    :param service_request: the dictionary containing the service request parameters\n    :param override_headers: whether to override headers that are also request parameters\n    :return: an ASF ServiceResponse (same as a service provider would return)\n    '
    local_context = create_aws_request_context(service_name=context.service.service_name, action=context.operation.name, parameters=service_request, region=context.region)
    headers = copy.deepcopy(context.request.headers)
    headers.update(local_context.request.headers)
    local_context.request.headers = headers
    return call_moto(local_context)

def _proxy_moto(context: RequestContext, request: ServiceRequest) -> Optional[Union[ServiceResponse, Response]]:
    if False:
        while True:
            i = 10
    '\n    Wraps `call_moto` such that the interface is compliant with a ServiceRequestHandler.\n\n    :param context: the request context\n    :param service_request: currently not being used, added to satisfy ServiceRequestHandler contract\n    :return: the Response from moto\n    '
    return call_moto(context)

def MotoFallbackDispatcher(provider: object) -> DispatchTable:
    if False:
        i = 10
        return i + 15
    '\n    Wraps a provider with a moto fallthrough mechanism. It does by creating a new DispatchTable from the original\n    provider, and wrapping each method with a fallthrough method that calls ``request`` if the original provider\n    raises a ``NotImplementedError``.\n\n    :param provider: the ASF provider\n    :return: a modified DispatchTable\n    '
    return ForwardingFallbackDispatcher(provider, _proxy_moto)

def dispatch_to_moto(context: RequestContext) -> Response:
    if False:
        for i in range(10):
            print('nop')
    "\n    Internal method to dispatch the request to moto without changing moto's dispatcher output.\n    :param context: the request context\n    :return: the response from moto\n    "
    service = context.service
    request = context.request
    dispatch = get_dispatcher(service.service_name, request.path)
    try:
        raw_url = get_raw_current_url(request.scheme, request.host, request.root_path, get_full_raw_path(request))
        response = dispatch(request, raw_url, request.headers)
        if not response:
            raise NotImplementedError
        (status, headers, content) = response
        if isinstance(content, str) and len(content) == 0:
            content = None
        return Response(content, status, headers)
    except RESTError as e:
        raise CommonServiceException(e.error_type, e.message, status_code=e.code) from e

def get_dispatcher(service: str, path: str) -> MotoDispatcher:
    if False:
        for i in range(10):
            print('nop')
    url_map = get_moto_routing_table(service)
    if len(url_map._rules) == 1:
        rule = next(url_map.iter_rules())
        return rule.endpoint
    matcher = url_map.bind(constants.LOCALHOST)
    try:
        (endpoint, _) = matcher.match(path_info=path)
    except NotFound as e:
        raise NotImplementedError(f'No moto route for service {service} on path {path} found.') from e
    return endpoint

@lru_cache()
def get_moto_routing_table(service: str) -> Map:
    if False:
        print('Hello World!')
    'Cached version of load_moto_routing_table.'
    return load_moto_routing_table(service)

def load_moto_routing_table(service: str) -> Map:
    if False:
        while True:
            i = 10
    '\n    Creates from moto service url_paths a werkzeug URL rule map that can be used to locate moto methods to dispatch\n    requests to.\n\n    :param service: the service to get the map for.\n    :return: a new Map object\n    '
    backend_dict = moto_backends.get_backend(service)
    if isinstance(backend_dict, BackendDict):
        if 'us-east-1' in backend_dict[DEFAULT_AWS_ACCOUNT_ID]:
            backend = backend_dict[DEFAULT_AWS_ACCOUNT_ID]['us-east-1']
        else:
            backend = backend_dict[DEFAULT_AWS_ACCOUNT_ID]['global']
    else:
        backend = backend_dict['global']
    url_map = Map()
    url_map.converters['regex'] = RegexConverter
    for (url_path, handler) in backend.flask_paths.items():
        strict_slashes = False
        endpoint = handler
        url_map.add(Rule(url_path, endpoint=endpoint, strict_slashes=strict_slashes))
    return url_map