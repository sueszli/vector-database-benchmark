"""
Lambda event construction and generation
"""
import base64
import logging
from datetime import datetime
from time import time
from typing import Any, Dict
from samcli.local.apigw.path_converter import PathConverter
from samcli.local.events.api_event import ApiGatewayLambdaEvent, ApiGatewayV2LambdaEvent, ContextHTTP, ContextIdentity, RequestContext, RequestContextV2
LOG = logging.getLogger(__name__)

def construct_v1_event(flask_request, port, binary_types, stage_name=None, stage_variables=None, operation_name=None) -> Dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper method that constructs the Event to be passed to Lambda\n\n    :param request flask_request: Flask Request\n    :param port: the port number\n    :param binary_types: list of binary types\n    :param stage_name: Optional, the stage name string\n    :param stage_variables: Optional, API Gateway Stage Variables\n    :return: JSON object\n    '
    identity = ContextIdentity(source_ip=flask_request.remote_addr)
    endpoint = PathConverter.convert_path_to_api_gateway(flask_request.endpoint)
    method = flask_request.method
    protocol = flask_request.environ.get('SERVER_PROTOCOL', 'HTTP/1.1')
    host = flask_request.host
    request_data = flask_request.get_data()
    request_mimetype = flask_request.mimetype
    is_base_64 = _should_base64_encode(binary_types, request_mimetype)
    if is_base_64:
        LOG.debug('Incoming Request seems to be binary. Base64 encoding the request data before sending to Lambda.')
        request_data = base64.b64encode(request_data)
    if request_data:
        request_data = request_data.decode('utf-8')
    (query_string_dict, multi_value_query_string_dict) = _query_string_params(flask_request)
    context = RequestContext(resource_path=endpoint, http_method=method, stage=stage_name, identity=identity, path=endpoint, protocol=protocol, domain_name=host, operation_name=operation_name)
    (headers_dict, multi_value_headers_dict) = _event_headers(flask_request, port)
    event = ApiGatewayLambdaEvent(http_method=method, body=request_data, resource=endpoint, request_context=context, query_string_params=query_string_dict, multi_value_query_string_params=multi_value_query_string_dict, headers=headers_dict, multi_value_headers=multi_value_headers_dict, path_parameters=flask_request.view_args, path=flask_request.path, is_base_64_encoded=is_base_64, stage_variables=stage_variables)
    event_dict = event.to_dict()
    LOG.debug('Constructed Event 1.0 to invoke Lambda. Event: %s', event_dict)
    return event_dict

def construct_v2_event_http(flask_request, port, binary_types, stage_name=None, stage_variables=None, route_key=None, request_time_epoch=int(time()), request_time=datetime.utcnow().strftime('%d/%b/%Y:%H:%M:%S +0000')) -> Dict[str, Any]:
    if False:
        while True:
            i = 10
    '\n    Helper method that constructs the Event 2.0 to be passed to Lambda\n\n    https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-develop-integrations-lambda.html\n\n    :param request flask_request: Flask Request\n    :param port: the port number\n    :param binary_types: list of binary types\n    :param stage_name: Optional, the stage name string\n    :param stage_variables: Optional, API Gateway Stage Variables\n    :param route_key: Optional, the route key for the route\n    :return: JSON object\n    '
    method = flask_request.method
    request_data = flask_request.get_data()
    request_mimetype = flask_request.mimetype
    is_base_64 = _should_base64_encode(binary_types, request_mimetype)
    if is_base_64:
        LOG.debug('Incoming Request seems to be binary. Base64 encoding the request data before sending to Lambda.')
        request_data = base64.b64encode(request_data)
    if request_data is not None:
        request_data = request_data.decode('utf-8')
    query_string_dict = _query_string_params_v_2_0(flask_request)
    cookies = _event_http_cookies(flask_request)
    headers = _event_http_headers(flask_request, port)
    context_http = ContextHTTP(method=method, path=flask_request.path, source_ip=flask_request.remote_addr)
    context = RequestContextV2(http=context_http, route_key=route_key, stage=stage_name, request_time_epoch=request_time_epoch, request_time=request_time)
    event = ApiGatewayV2LambdaEvent(route_key=route_key, raw_path=flask_request.path, raw_query_string=flask_request.query_string.decode('utf-8'), cookies=cookies, headers=headers, query_string_params=query_string_dict, request_context=context, body=request_data, path_parameters=flask_request.view_args, is_base_64_encoded=is_base_64, stage_variables=stage_variables)
    event_dict = event.to_dict()
    LOG.debug('Constructed Event Version 2.0 to invoke Lambda. Event: %s', event_dict)
    return event_dict

def _query_string_params(flask_request):
    if False:
        i = 10
        return i + 15
    '\n    Constructs an APIGW equivalent query string dictionary\n\n    Parameters\n    ----------\n    flask_request request\n        Request from Flask\n\n    Returns dict (str: str), dict (str: list of str)\n    -------\n        Empty dict if no query params where in the request otherwise returns a dictionary of key to value\n\n    '
    query_string_dict = {}
    multi_value_query_string_dict = {}
    for (query_string_key, query_string_list) in flask_request.args.lists():
        query_string_value_length = len(query_string_list)
        if not query_string_value_length:
            query_string_dict[query_string_key] = ''
            multi_value_query_string_dict[query_string_key] = ['']
        else:
            query_string_dict[query_string_key] = query_string_list[-1]
            multi_value_query_string_dict[query_string_key] = query_string_list
    return (query_string_dict, multi_value_query_string_dict)

def _query_string_params_v_2_0(flask_request):
    if False:
        return 10
    '\n    Constructs an APIGW equivalent query string dictionary using the 2.0 format\n    https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-develop-integrations-lambda.html#2.0\n\n    Parameters\n    ----------\n    flask_request request\n        Request from Flask\n\n    Returns dict (str: str)\n    -------\n        Empty dict if no query params where in the request otherwise returns a dictionary of key to value\n\n    '
    query_string_dict = {}
    query_string_dict = {query_string_key: ','.join(query_string_list) for (query_string_key, query_string_list) in flask_request.args.lists()}
    return query_string_dict

def _event_headers(flask_request, port):
    if False:
        print('Hello World!')
    '\n    Constructs an APIGW equivalent headers dictionary\n\n    Parameters\n    ----------\n    flask_request request\n        Request from Flask\n    int port\n        Forwarded Port\n    cors_headers dict\n        Dict of the Cors properties\n\n    Returns dict (str: str), dict (str: list of str)\n    -------\n        Returns a dictionary of key to list of strings\n\n    '
    headers_dict = {}
    multi_value_headers_dict = {}
    for header_key in flask_request.headers.keys():
        headers_dict[header_key] = flask_request.headers.get(header_key)
        multi_value_headers_dict[header_key] = flask_request.headers.getlist(header_key)
    headers_dict['X-Forwarded-Proto'] = flask_request.scheme
    multi_value_headers_dict['X-Forwarded-Proto'] = [flask_request.scheme]
    headers_dict['X-Forwarded-Port'] = str(port)
    multi_value_headers_dict['X-Forwarded-Port'] = [str(port)]
    return (headers_dict, multi_value_headers_dict)

def _event_http_cookies(flask_request):
    if False:
        while True:
            i = 10
    '\n    All cookie headers in the request are combined with commas.\n\n    https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-develop-integrations-lambda.html\n\n    Parameters\n    ----------\n    flask_request request\n        Request from Flask\n\n    Returns list\n    -------\n        Returns a list of cookies\n\n    '
    cookies = []
    for cookie_key in flask_request.cookies.keys():
        cookies.append(f'{cookie_key}={flask_request.cookies.get(cookie_key)}')
    return cookies

def _event_http_headers(flask_request, port):
    if False:
        return 10
    '\n    Duplicate headers are combined with commas.\n\n    https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-develop-integrations-lambda.html\n\n    Parameters\n    ----------\n    flask_request request\n        Request from Flask\n\n    Returns list\n    -------\n        Returns a list of cookies\n\n    '
    headers = {}
    for header_key in flask_request.headers.keys():
        headers[header_key] = flask_request.headers.get(header_key)
    headers['X-Forwarded-Proto'] = flask_request.scheme
    headers['X-Forwarded-Port'] = str(port)
    return headers

def _should_base64_encode(binary_types, request_mimetype):
    if False:
        while True:
            i = 10
    '\n    Whether or not to encode the data from the request to Base64\n\n    Parameters\n    ----------\n    binary_types list(basestring)\n        Corresponds to self.binary_types (aka. what is parsed from SAM Template\n    request_mimetype str\n        Mimetype for the request\n\n    Returns\n    -------\n        True if the data should be encoded to Base64 otherwise False\n\n    '
    return request_mimetype in binary_types or '*/*' in binary_types