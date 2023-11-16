"""Synchronized request in the Azure Cosmos database service.
"""
import json
import time
from urllib.parse import urlparse
from azure.core.exceptions import DecodeError
from . import exceptions
from . import http_constants
from . import _retry_utility

def _is_readable_stream(obj):
    if False:
        return 10
    'Checks whether obj is a file-like readable stream.\n\n    :param Union[str, unicode, file-like stream object, dict, list, None] obj: the object to be checked.\n    :returns: whether the object is a file-like readable stream.\n    :rtype: boolean\n    '
    if hasattr(obj, 'read') and callable(getattr(obj, 'read')):
        return True
    return False

def _request_body_from_data(data):
    if False:
        i = 10
        return i + 15
    'Gets request body from data.\n\n    When `data` is dict and list into unicode string; otherwise return `data`\n    without making any change.\n\n    :param Union[str, unicode, file-like stream object, dict, list, None] data:\n    :returns: the json dump data.\n    :rtype: Union[str, unicode, file-like stream object, None]\n\n    '
    if data is None or isinstance(data, str) or _is_readable_stream(data):
        return data
    if isinstance(data, (dict, list, tuple)):
        json_dumped = json.dumps(data, separators=(',', ':'))
        return json_dumped
    return None

def _Request(global_endpoint_manager, request_params, connection_policy, pipeline_client, request, **kwargs):
    if False:
        i = 10
        return i + 15
    'Makes one http request using the requests module.\n\n    :param _GlobalEndpointManager global_endpoint_manager:\n    :param dict request_params:\n        contains the resourceType, operationType, endpointOverride,\n        useWriteEndpoint, useAlternateWriteEndpoint information\n    :param documents.ConnectionPolicy connection_policy:\n    :param azure.core.PipelineClient pipeline_client:\n        Pipeline client to process the request\n    :param azure.core.HttpRequest request:\n        The request object to send through the pipeline\n    :return: tuple of (result, headers)\n    :rtype: tuple of (dict, dict)\n\n    '
    connection_timeout = connection_policy.RequestTimeout
    connection_timeout = kwargs.pop('connection_timeout', connection_timeout)
    client_timeout = kwargs.get('timeout')
    start_time = time.time()
    global_endpoint_manager.refresh_endpoint_list(None, **kwargs)
    if client_timeout is not None:
        kwargs['timeout'] = client_timeout - (time.time() - start_time)
        if kwargs['timeout'] <= 0:
            raise exceptions.CosmosClientTimeoutError()
    if request_params.endpoint_override:
        base_url = request_params.endpoint_override
    else:
        base_url = global_endpoint_manager.resolve_service_endpoint(request_params)
    if base_url != pipeline_client._base_url:
        request.url = request.url.replace(pipeline_client._base_url, base_url)
    parse_result = urlparse(request.url)
    request.headers.update({header: str(value) for (header, value) in request.headers.items()})
    is_ssl_enabled = parse_result.hostname != 'localhost' and parse_result.hostname != '127.0.0.1' and (not connection_policy.DisableSSLVerification)
    if connection_policy.SSLConfiguration or 'connection_cert' in kwargs:
        ca_certs = connection_policy.SSLConfiguration.SSLCaCerts
        cert_files = (connection_policy.SSLConfiguration.SSLCertFile, connection_policy.SSLConfiguration.SSLKeyFile)
        response = _PipelineRunFunction(pipeline_client, request, connection_timeout=connection_timeout, connection_verify=kwargs.pop('connection_verify', ca_certs), connection_cert=kwargs.pop('connection_cert', cert_files), **kwargs)
    else:
        response = _PipelineRunFunction(pipeline_client, request, connection_timeout=connection_timeout, connection_verify=kwargs.pop('connection_verify', is_ssl_enabled), **kwargs)
    response = response.http_response
    headers = dict(response.headers)
    data = response.body()
    if data:
        data = data.decode('utf-8')
    if response.status_code == 404:
        raise exceptions.CosmosResourceNotFoundError(message=data, response=response)
    if response.status_code == 409:
        raise exceptions.CosmosResourceExistsError(message=data, response=response)
    if response.status_code == 412:
        raise exceptions.CosmosAccessConditionFailedError(message=data, response=response)
    if response.status_code >= 400:
        raise exceptions.CosmosHttpResponseError(message=data, response=response)
    result = None
    if data:
        try:
            result = json.loads(data)
        except Exception as e:
            raise DecodeError(message='Failed to decode JSON data: {}'.format(e), response=response, error=e) from e
    return (result, headers)

def _PipelineRunFunction(pipeline_client, request, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return pipeline_client._pipeline.run(request, **kwargs)

def SynchronizedRequest(client, request_params, global_endpoint_manager, connection_policy, pipeline_client, request, request_data, **kwargs):
    if False:
        i = 10
        return i + 15
    'Performs one synchronized http request according to the parameters.\n\n    :param object client: Document client instance\n    :param dict request_params:\n    :param _GlobalEndpointManager global_endpoint_manager:\n    :param documents.ConnectionPolicy connection_policy:\n    :param azure.core.PipelineClient pipeline_client: PipelineClient to process the request.\n    :param HttpRequest request: the HTTP request to be sent\n    :param (str, unicode, file-like stream object, dict, list or None) request_data: the data to be sent in the request\n    :return: tuple of (result, headers)\n    :rtype: tuple of (dict dict)\n    '
    request.data = _request_body_from_data(request_data)
    if request.data and isinstance(request.data, str):
        request.headers[http_constants.HttpHeaders.ContentLength] = len(request.data)
    elif request.data is None:
        request.headers[http_constants.HttpHeaders.ContentLength] = 0
    return _retry_utility.Execute(client, global_endpoint_manager, _Request, request_params, connection_policy, pipeline_client, request, **kwargs)