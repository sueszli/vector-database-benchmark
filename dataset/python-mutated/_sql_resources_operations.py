from io import IOBase
from typing import Any, Callable, Dict, IO, Iterable, Optional, TypeVar, Union, cast, overload
import urllib.parse
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceExistsError, ResourceNotFoundError, ResourceNotModifiedError, map_error
from azure.core.paging import ItemPaged
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import HttpResponse
from azure.core.polling import LROPoller, NoPolling, PollingMethod
from azure.core.rest import HttpRequest
from azure.core.tracing.decorator import distributed_trace
from azure.core.utils import case_insensitive_dict
from azure.mgmt.core.exceptions import ARMErrorFormat
from azure.mgmt.core.polling.arm_polling import ARMPolling
from .. import models as _models
from .._serialization import Serializer
from .._vendor import _convert_request
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_list_sql_databases_request(resource_group_name: str, account_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_sql_database_request(resource_group_name: str, account_name: str, database_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_create_update_sql_database_request(resource_group_name: str, account_name: str, database_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_sql_database_request(resource_group_name: str, account_name: str, database_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, **kwargs)

def build_get_sql_database_throughput_request(resource_group_name: str, account_name: str, database_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/throughputSettings/default')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_update_sql_database_throughput_request(resource_group_name: str, account_name: str, database_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/throughputSettings/default')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_migrate_sql_database_to_autoscale_request(resource_group_name: str, account_name: str, database_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/throughputSettings/default/migrateToAutoscale')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

def build_migrate_sql_database_to_manual_throughput_request(resource_group_name: str, account_name: str, database_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/throughputSettings/default/migrateToManualThroughput')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_sql_containers_request(resource_group_name: str, account_name: str, database_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_sql_container_request(resource_group_name: str, account_name: str, database_name: str, container_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str'), 'containerName': _SERIALIZER.url('container_name', container_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_create_update_sql_container_request(resource_group_name: str, account_name: str, database_name: str, container_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str'), 'containerName': _SERIALIZER.url('container_name', container_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_sql_container_request(resource_group_name: str, account_name: str, database_name: str, container_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str'), 'containerName': _SERIALIZER.url('container_name', container_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, **kwargs)

def build_get_sql_container_throughput_request(resource_group_name: str, account_name: str, database_name: str, container_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/throughputSettings/default')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str'), 'containerName': _SERIALIZER.url('container_name', container_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_update_sql_container_throughput_request(resource_group_name: str, account_name: str, database_name: str, container_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/throughputSettings/default')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str'), 'containerName': _SERIALIZER.url('container_name', container_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_migrate_sql_container_to_autoscale_request(resource_group_name: str, account_name: str, database_name: str, container_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/throughputSettings/default/migrateToAutoscale')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str'), 'containerName': _SERIALIZER.url('container_name', container_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

def build_migrate_sql_container_to_manual_throughput_request(resource_group_name: str, account_name: str, database_name: str, container_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/throughputSettings/default/migrateToManualThroughput')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str'), 'containerName': _SERIALIZER.url('container_name', container_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_client_encryption_keys_request(resource_group_name: str, account_name: str, database_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/clientEncryptionKeys')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_client_encryption_key_request(resource_group_name: str, account_name: str, database_name: str, client_encryption_key_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/clientEncryptionKeys/{clientEncryptionKeyName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str'), 'clientEncryptionKeyName': _SERIALIZER.url('client_encryption_key_name', client_encryption_key_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_create_update_client_encryption_key_request(resource_group_name: str, account_name: str, database_name: str, client_encryption_key_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/clientEncryptionKeys/{clientEncryptionKeyName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str'), 'clientEncryptionKeyName': _SERIALIZER.url('client_encryption_key_name', client_encryption_key_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_sql_stored_procedures_request(resource_group_name: str, account_name: str, database_name: str, container_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/storedProcedures')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str'), 'containerName': _SERIALIZER.url('container_name', container_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_sql_stored_procedure_request(resource_group_name: str, account_name: str, database_name: str, container_name: str, stored_procedure_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/storedProcedures/{storedProcedureName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str'), 'containerName': _SERIALIZER.url('container_name', container_name, 'str'), 'storedProcedureName': _SERIALIZER.url('stored_procedure_name', stored_procedure_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_create_update_sql_stored_procedure_request(resource_group_name: str, account_name: str, database_name: str, container_name: str, stored_procedure_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/storedProcedures/{storedProcedureName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str'), 'containerName': _SERIALIZER.url('container_name', container_name, 'str'), 'storedProcedureName': _SERIALIZER.url('stored_procedure_name', stored_procedure_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_sql_stored_procedure_request(resource_group_name: str, account_name: str, database_name: str, container_name: str, stored_procedure_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/storedProcedures/{storedProcedureName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str'), 'containerName': _SERIALIZER.url('container_name', container_name, 'str'), 'storedProcedureName': _SERIALIZER.url('stored_procedure_name', stored_procedure_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, **kwargs)

def build_list_sql_user_defined_functions_request(resource_group_name: str, account_name: str, database_name: str, container_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/userDefinedFunctions')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str'), 'containerName': _SERIALIZER.url('container_name', container_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_sql_user_defined_function_request(resource_group_name: str, account_name: str, database_name: str, container_name: str, user_defined_function_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/userDefinedFunctions/{userDefinedFunctionName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str'), 'containerName': _SERIALIZER.url('container_name', container_name, 'str'), 'userDefinedFunctionName': _SERIALIZER.url('user_defined_function_name', user_defined_function_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_create_update_sql_user_defined_function_request(resource_group_name: str, account_name: str, database_name: str, container_name: str, user_defined_function_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/userDefinedFunctions/{userDefinedFunctionName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str'), 'containerName': _SERIALIZER.url('container_name', container_name, 'str'), 'userDefinedFunctionName': _SERIALIZER.url('user_defined_function_name', user_defined_function_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_sql_user_defined_function_request(resource_group_name: str, account_name: str, database_name: str, container_name: str, user_defined_function_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/userDefinedFunctions/{userDefinedFunctionName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str'), 'containerName': _SERIALIZER.url('container_name', container_name, 'str'), 'userDefinedFunctionName': _SERIALIZER.url('user_defined_function_name', user_defined_function_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, **kwargs)

def build_list_sql_triggers_request(resource_group_name: str, account_name: str, database_name: str, container_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/triggers')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str'), 'containerName': _SERIALIZER.url('container_name', container_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_sql_trigger_request(resource_group_name: str, account_name: str, database_name: str, container_name: str, trigger_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/triggers/{triggerName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str'), 'containerName': _SERIALIZER.url('container_name', container_name, 'str'), 'triggerName': _SERIALIZER.url('trigger_name', trigger_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_create_update_sql_trigger_request(resource_group_name: str, account_name: str, database_name: str, container_name: str, trigger_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/triggers/{triggerName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str'), 'containerName': _SERIALIZER.url('container_name', container_name, 'str'), 'triggerName': _SERIALIZER.url('trigger_name', trigger_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_sql_trigger_request(resource_group_name: str, account_name: str, database_name: str, container_name: str, trigger_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/triggers/{triggerName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str'), 'containerName': _SERIALIZER.url('container_name', container_name, 'str'), 'triggerName': _SERIALIZER.url('trigger_name', trigger_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, **kwargs)

def build_get_sql_role_definition_request(role_definition_id: str, resource_group_name: str, account_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlRoleDefinitions/{roleDefinitionId}')
    path_format_arguments = {'roleDefinitionId': _SERIALIZER.url('role_definition_id', role_definition_id, 'str'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_create_update_sql_role_definition_request(role_definition_id: str, resource_group_name: str, account_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlRoleDefinitions/{roleDefinitionId}')
    path_format_arguments = {'roleDefinitionId': _SERIALIZER.url('role_definition_id', role_definition_id, 'str'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_sql_role_definition_request(role_definition_id: str, resource_group_name: str, account_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlRoleDefinitions/{roleDefinitionId}')
    path_format_arguments = {'roleDefinitionId': _SERIALIZER.url('role_definition_id', role_definition_id, 'str'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_sql_role_definitions_request(resource_group_name: str, account_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlRoleDefinitions')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_sql_role_assignment_request(role_assignment_id: str, resource_group_name: str, account_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlRoleAssignments/{roleAssignmentId}')
    path_format_arguments = {'roleAssignmentId': _SERIALIZER.url('role_assignment_id', role_assignment_id, 'str'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_create_update_sql_role_assignment_request(role_assignment_id: str, resource_group_name: str, account_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlRoleAssignments/{roleAssignmentId}')
    path_format_arguments = {'roleAssignmentId': _SERIALIZER.url('role_assignment_id', role_assignment_id, 'str'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_sql_role_assignment_request(role_assignment_id: str, resource_group_name: str, account_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlRoleAssignments/{roleAssignmentId}')
    path_format_arguments = {'roleAssignmentId': _SERIALIZER.url('role_assignment_id', role_assignment_id, 'str'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_sql_role_assignments_request(resource_group_name: str, account_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlRoleAssignments')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_retrieve_continuous_backup_information_request(resource_group_name: str, account_name: str, database_name: str, container_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-09-15'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/retrieveContinuousBackupInformation')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'accountName': _SERIALIZER.url('account_name', account_name, 'str', max_length=50, min_length=3, pattern='^[a-z0-9]+(-[a-z0-9]+)*'), 'databaseName': _SERIALIZER.url('database_name', database_name, 'str'), 'containerName': _SERIALIZER.url('container_name', container_name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

class SqlResourcesOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.cosmosdb.CosmosDBManagementClient`'s
        :attr:`sql_resources` attribute.
    """
    models = _models

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        input_args = list(args)
        self._client = input_args.pop(0) if input_args else kwargs.pop('client')
        self._config = input_args.pop(0) if input_args else kwargs.pop('config')
        self._serialize = input_args.pop(0) if input_args else kwargs.pop('serializer')
        self._deserialize = input_args.pop(0) if input_args else kwargs.pop('deserializer')

    @distributed_trace
    def list_sql_databases(self, resource_group_name: str, account_name: str, **kwargs: Any) -> Iterable['_models.SqlDatabaseGetResults']:
        if False:
            while True:
                i = 10
        'Lists the SQL databases under an existing Azure Cosmos DB database account.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either SqlDatabaseGetResults or the result of\n         cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.cosmosdb.models.SqlDatabaseGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.SqlDatabaseListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                return 10
            if not next_link:
                request = build_list_sql_databases_request(resource_group_name=resource_group_name, account_name=account_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.list_sql_databases.metadata['url'], headers=_headers, params=_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
            else:
                _parsed_next_link = urllib.parse.urlparse(next_link)
                _next_request_params = case_insensitive_dict({key: [urllib.parse.quote(v) for v in value] for (key, value) in urllib.parse.parse_qs(_parsed_next_link.query).items()})
                _next_request_params['api-version'] = self._config.api_version
                request = HttpRequest('GET', urllib.parse.urljoin(next_link, _parsed_next_link.path), params=_next_request_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
                request.method = 'GET'
            return request

        def extract_data(pipeline_response):
            if False:
                print('Hello World!')
            deserialized = self._deserialize('SqlDatabaseListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                return 10
            request = prepare_request(next_link)
            _stream = False
            pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                raise HttpResponseError(response=response, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_sql_databases.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases'}

    @distributed_trace
    def get_sql_database(self, resource_group_name: str, account_name: str, database_name: str, **kwargs: Any) -> _models.SqlDatabaseGetResults:
        if False:
            print('Hello World!')
        'Gets the SQL database under an existing Azure Cosmos DB database account with the provided\n        name.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: SqlDatabaseGetResults or the result of cls(response)\n        :rtype: ~azure.mgmt.cosmosdb.models.SqlDatabaseGetResults\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.SqlDatabaseGetResults] = kwargs.pop('cls', None)
        request = build_get_sql_database_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get_sql_database.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('SqlDatabaseGetResults', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get_sql_database.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}'}

    def _create_update_sql_database_initial(self, resource_group_name: str, account_name: str, database_name: str, create_update_sql_database_parameters: Union[_models.SqlDatabaseCreateUpdateParameters, IO], **kwargs: Any) -> Optional[_models.SqlDatabaseGetResults]:
        if False:
            for i in range(10):
                print('nop')
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[Optional[_models.SqlDatabaseGetResults]] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(create_update_sql_database_parameters, (IOBase, bytes)):
            _content = create_update_sql_database_parameters
        else:
            _json = self._serialize.body(create_update_sql_database_parameters, 'SqlDatabaseCreateUpdateParameters')
        request = build_create_update_sql_database_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._create_update_sql_database_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = None
        response_headers = {}
        if response.status_code == 200:
            deserialized = self._deserialize('SqlDatabaseGetResults', pipeline_response)
        if response.status_code == 202:
            response_headers['azure-AsyncOperation'] = self._deserialize('str', response.headers.get('azure-AsyncOperation'))
            response_headers['location'] = self._deserialize('str', response.headers.get('location'))
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    _create_update_sql_database_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}'}

    @overload
    def begin_create_update_sql_database(self, resource_group_name: str, account_name: str, database_name: str, create_update_sql_database_parameters: _models.SqlDatabaseCreateUpdateParameters, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.SqlDatabaseGetResults]:
        if False:
            return 10
        'Create or update an Azure Cosmos DB SQL database.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param create_update_sql_database_parameters: The parameters to provide for the current SQL\n         database. Required.\n        :type create_update_sql_database_parameters:\n         ~azure.mgmt.cosmosdb.models.SqlDatabaseCreateUpdateParameters\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either SqlDatabaseGetResults or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.SqlDatabaseGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_create_update_sql_database(self, resource_group_name: str, account_name: str, database_name: str, create_update_sql_database_parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.SqlDatabaseGetResults]:
        if False:
            print('Hello World!')
        'Create or update an Azure Cosmos DB SQL database.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param create_update_sql_database_parameters: The parameters to provide for the current SQL\n         database. Required.\n        :type create_update_sql_database_parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either SqlDatabaseGetResults or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.SqlDatabaseGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_create_update_sql_database(self, resource_group_name: str, account_name: str, database_name: str, create_update_sql_database_parameters: Union[_models.SqlDatabaseCreateUpdateParameters, IO], **kwargs: Any) -> LROPoller[_models.SqlDatabaseGetResults]:
        if False:
            return 10
        "Create or update an Azure Cosmos DB SQL database.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param create_update_sql_database_parameters: The parameters to provide for the current SQL\n         database. Is either a SqlDatabaseCreateUpdateParameters type or a IO type. Required.\n        :type create_update_sql_database_parameters:\n         ~azure.mgmt.cosmosdb.models.SqlDatabaseCreateUpdateParameters or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either SqlDatabaseGetResults or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.SqlDatabaseGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.SqlDatabaseGetResults] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._create_update_sql_database_initial(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, create_update_sql_database_parameters=create_update_sql_database_parameters, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                i = 10
                return i + 15
            deserialized = self._deserialize('SqlDatabaseGetResults', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_create_update_sql_database.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}'}

    def _delete_sql_database_initial(self, resource_group_name: str, account_name: str, database_name: str, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_delete_sql_database_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self._delete_sql_database_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [202, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        response_headers = {}
        if response.status_code == 202:
            response_headers['azure-AsyncOperation'] = self._deserialize('str', response.headers.get('azure-AsyncOperation'))
            response_headers['location'] = self._deserialize('str', response.headers.get('location'))
        if cls:
            return cls(pipeline_response, None, response_headers)
    _delete_sql_database_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}'}

    @distributed_trace
    def begin_delete_sql_database(self, resource_group_name: str, account_name: str, database_name: str, **kwargs: Any) -> LROPoller[None]:
        if False:
            for i in range(10):
                print('nop')
        'Deletes an existing Azure Cosmos DB SQL database.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either None or the result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[None]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._delete_sql_database_initial(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, api_version=api_version, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                while True:
                    i = 10
            if cls:
                return cls(pipeline_response, None, {})
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_delete_sql_database.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}'}

    @distributed_trace
    def get_sql_database_throughput(self, resource_group_name: str, account_name: str, database_name: str, **kwargs: Any) -> _models.ThroughputSettingsGetResults:
        if False:
            i = 10
            return i + 15
        'Gets the RUs per second of the SQL database under an existing Azure Cosmos DB database account\n        with the provided name.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ThroughputSettingsGetResults or the result of cls(response)\n        :rtype: ~azure.mgmt.cosmosdb.models.ThroughputSettingsGetResults\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.ThroughputSettingsGetResults] = kwargs.pop('cls', None)
        request = build_get_sql_database_throughput_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get_sql_database_throughput.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('ThroughputSettingsGetResults', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get_sql_database_throughput.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/throughputSettings/default'}

    def _update_sql_database_throughput_initial(self, resource_group_name: str, account_name: str, database_name: str, update_throughput_parameters: Union[_models.ThroughputSettingsUpdateParameters, IO], **kwargs: Any) -> Optional[_models.ThroughputSettingsGetResults]:
        if False:
            return 10
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[Optional[_models.ThroughputSettingsGetResults]] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(update_throughput_parameters, (IOBase, bytes)):
            _content = update_throughput_parameters
        else:
            _json = self._serialize.body(update_throughput_parameters, 'ThroughputSettingsUpdateParameters')
        request = build_update_sql_database_throughput_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._update_sql_database_throughput_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = None
        response_headers = {}
        if response.status_code == 200:
            deserialized = self._deserialize('ThroughputSettingsGetResults', pipeline_response)
        if response.status_code == 202:
            response_headers['azure-AsyncOperation'] = self._deserialize('str', response.headers.get('azure-AsyncOperation'))
            response_headers['location'] = self._deserialize('str', response.headers.get('location'))
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    _update_sql_database_throughput_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/throughputSettings/default'}

    @overload
    def begin_update_sql_database_throughput(self, resource_group_name: str, account_name: str, database_name: str, update_throughput_parameters: _models.ThroughputSettingsUpdateParameters, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.ThroughputSettingsGetResults]:
        if False:
            while True:
                i = 10
        'Update RUs per second of an Azure Cosmos DB SQL database.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param update_throughput_parameters: The parameters to provide for the RUs per second of the\n         current SQL database. Required.\n        :type update_throughput_parameters:\n         ~azure.mgmt.cosmosdb.models.ThroughputSettingsUpdateParameters\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either ThroughputSettingsGetResults or the\n         result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.ThroughputSettingsGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_update_sql_database_throughput(self, resource_group_name: str, account_name: str, database_name: str, update_throughput_parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.ThroughputSettingsGetResults]:
        if False:
            print('Hello World!')
        'Update RUs per second of an Azure Cosmos DB SQL database.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param update_throughput_parameters: The parameters to provide for the RUs per second of the\n         current SQL database. Required.\n        :type update_throughput_parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either ThroughputSettingsGetResults or the\n         result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.ThroughputSettingsGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_update_sql_database_throughput(self, resource_group_name: str, account_name: str, database_name: str, update_throughput_parameters: Union[_models.ThroughputSettingsUpdateParameters, IO], **kwargs: Any) -> LROPoller[_models.ThroughputSettingsGetResults]:
        if False:
            for i in range(10):
                print('nop')
        "Update RUs per second of an Azure Cosmos DB SQL database.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param update_throughput_parameters: The parameters to provide for the RUs per second of the\n         current SQL database. Is either a ThroughputSettingsUpdateParameters type or a IO type.\n         Required.\n        :type update_throughput_parameters:\n         ~azure.mgmt.cosmosdb.models.ThroughputSettingsUpdateParameters or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either ThroughputSettingsGetResults or the\n         result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.ThroughputSettingsGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.ThroughputSettingsGetResults] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._update_sql_database_throughput_initial(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, update_throughput_parameters=update_throughput_parameters, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                while True:
                    i = 10
            deserialized = self._deserialize('ThroughputSettingsGetResults', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_update_sql_database_throughput.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/throughputSettings/default'}

    def _migrate_sql_database_to_autoscale_initial(self, resource_group_name: str, account_name: str, database_name: str, **kwargs: Any) -> Optional[_models.ThroughputSettingsGetResults]:
        if False:
            i = 10
            return i + 15
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[Optional[_models.ThroughputSettingsGetResults]] = kwargs.pop('cls', None)
        request = build_migrate_sql_database_to_autoscale_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self._migrate_sql_database_to_autoscale_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = None
        response_headers = {}
        if response.status_code == 200:
            deserialized = self._deserialize('ThroughputSettingsGetResults', pipeline_response)
        if response.status_code == 202:
            response_headers['azure-AsyncOperation'] = self._deserialize('str', response.headers.get('azure-AsyncOperation'))
            response_headers['location'] = self._deserialize('str', response.headers.get('location'))
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    _migrate_sql_database_to_autoscale_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/throughputSettings/default/migrateToAutoscale'}

    @distributed_trace
    def begin_migrate_sql_database_to_autoscale(self, resource_group_name: str, account_name: str, database_name: str, **kwargs: Any) -> LROPoller[_models.ThroughputSettingsGetResults]:
        if False:
            for i in range(10):
                print('nop')
        'Migrate an Azure Cosmos DB SQL database from manual throughput to autoscale.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either ThroughputSettingsGetResults or the\n         result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.ThroughputSettingsGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.ThroughputSettingsGetResults] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._migrate_sql_database_to_autoscale_initial(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, api_version=api_version, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                return 10
            deserialized = self._deserialize('ThroughputSettingsGetResults', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_migrate_sql_database_to_autoscale.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/throughputSettings/default/migrateToAutoscale'}

    def _migrate_sql_database_to_manual_throughput_initial(self, resource_group_name: str, account_name: str, database_name: str, **kwargs: Any) -> Optional[_models.ThroughputSettingsGetResults]:
        if False:
            return 10
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[Optional[_models.ThroughputSettingsGetResults]] = kwargs.pop('cls', None)
        request = build_migrate_sql_database_to_manual_throughput_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self._migrate_sql_database_to_manual_throughput_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = None
        response_headers = {}
        if response.status_code == 200:
            deserialized = self._deserialize('ThroughputSettingsGetResults', pipeline_response)
        if response.status_code == 202:
            response_headers['azure-AsyncOperation'] = self._deserialize('str', response.headers.get('azure-AsyncOperation'))
            response_headers['location'] = self._deserialize('str', response.headers.get('location'))
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    _migrate_sql_database_to_manual_throughput_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/throughputSettings/default/migrateToManualThroughput'}

    @distributed_trace
    def begin_migrate_sql_database_to_manual_throughput(self, resource_group_name: str, account_name: str, database_name: str, **kwargs: Any) -> LROPoller[_models.ThroughputSettingsGetResults]:
        if False:
            for i in range(10):
                print('nop')
        'Migrate an Azure Cosmos DB SQL database from autoscale to manual throughput.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either ThroughputSettingsGetResults or the\n         result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.ThroughputSettingsGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.ThroughputSettingsGetResults] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._migrate_sql_database_to_manual_throughput_initial(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, api_version=api_version, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                i = 10
                return i + 15
            deserialized = self._deserialize('ThroughputSettingsGetResults', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_migrate_sql_database_to_manual_throughput.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/throughputSettings/default/migrateToManualThroughput'}

    @distributed_trace
    def list_sql_containers(self, resource_group_name: str, account_name: str, database_name: str, **kwargs: Any) -> Iterable['_models.SqlContainerGetResults']:
        if False:
            return 10
        'Lists the SQL container under an existing Azure Cosmos DB database account.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either SqlContainerGetResults or the result of\n         cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.cosmosdb.models.SqlContainerGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.SqlContainerListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                while True:
                    i = 10
            if not next_link:
                request = build_list_sql_containers_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.list_sql_containers.metadata['url'], headers=_headers, params=_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
            else:
                _parsed_next_link = urllib.parse.urlparse(next_link)
                _next_request_params = case_insensitive_dict({key: [urllib.parse.quote(v) for v in value] for (key, value) in urllib.parse.parse_qs(_parsed_next_link.query).items()})
                _next_request_params['api-version'] = self._config.api_version
                request = HttpRequest('GET', urllib.parse.urljoin(next_link, _parsed_next_link.path), params=_next_request_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
                request.method = 'GET'
            return request

        def extract_data(pipeline_response):
            if False:
                i = 10
                return i + 15
            deserialized = self._deserialize('SqlContainerListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                i = 10
                return i + 15
            request = prepare_request(next_link)
            _stream = False
            pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                raise HttpResponseError(response=response, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_sql_containers.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers'}

    @distributed_trace
    def get_sql_container(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, **kwargs: Any) -> _models.SqlContainerGetResults:
        if False:
            return 10
        'Gets the SQL container under an existing Azure Cosmos DB database account.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: SqlContainerGetResults or the result of cls(response)\n        :rtype: ~azure.mgmt.cosmosdb.models.SqlContainerGetResults\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.SqlContainerGetResults] = kwargs.pop('cls', None)
        request = build_get_sql_container_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get_sql_container.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('SqlContainerGetResults', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get_sql_container.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}'}

    def _create_update_sql_container_initial(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, create_update_sql_container_parameters: Union[_models.SqlContainerCreateUpdateParameters, IO], **kwargs: Any) -> Optional[_models.SqlContainerGetResults]:
        if False:
            i = 10
            return i + 15
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[Optional[_models.SqlContainerGetResults]] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(create_update_sql_container_parameters, (IOBase, bytes)):
            _content = create_update_sql_container_parameters
        else:
            _json = self._serialize.body(create_update_sql_container_parameters, 'SqlContainerCreateUpdateParameters')
        request = build_create_update_sql_container_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._create_update_sql_container_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = None
        response_headers = {}
        if response.status_code == 200:
            deserialized = self._deserialize('SqlContainerGetResults', pipeline_response)
        if response.status_code == 202:
            response_headers['azure-AsyncOperation'] = self._deserialize('str', response.headers.get('azure-AsyncOperation'))
            response_headers['location'] = self._deserialize('str', response.headers.get('location'))
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    _create_update_sql_container_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}'}

    @overload
    def begin_create_update_sql_container(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, create_update_sql_container_parameters: _models.SqlContainerCreateUpdateParameters, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.SqlContainerGetResults]:
        if False:
            i = 10
            return i + 15
        'Create or update an Azure Cosmos DB SQL container.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :param create_update_sql_container_parameters: The parameters to provide for the current SQL\n         container. Required.\n        :type create_update_sql_container_parameters:\n         ~azure.mgmt.cosmosdb.models.SqlContainerCreateUpdateParameters\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either SqlContainerGetResults or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.SqlContainerGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_create_update_sql_container(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, create_update_sql_container_parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.SqlContainerGetResults]:
        if False:
            print('Hello World!')
        'Create or update an Azure Cosmos DB SQL container.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :param create_update_sql_container_parameters: The parameters to provide for the current SQL\n         container. Required.\n        :type create_update_sql_container_parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either SqlContainerGetResults or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.SqlContainerGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_create_update_sql_container(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, create_update_sql_container_parameters: Union[_models.SqlContainerCreateUpdateParameters, IO], **kwargs: Any) -> LROPoller[_models.SqlContainerGetResults]:
        if False:
            while True:
                i = 10
        "Create or update an Azure Cosmos DB SQL container.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :param create_update_sql_container_parameters: The parameters to provide for the current SQL\n         container. Is either a SqlContainerCreateUpdateParameters type or a IO type. Required.\n        :type create_update_sql_container_parameters:\n         ~azure.mgmt.cosmosdb.models.SqlContainerCreateUpdateParameters or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either SqlContainerGetResults or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.SqlContainerGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.SqlContainerGetResults] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._create_update_sql_container_initial(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, create_update_sql_container_parameters=create_update_sql_container_parameters, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                print('Hello World!')
            deserialized = self._deserialize('SqlContainerGetResults', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_create_update_sql_container.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}'}

    def _delete_sql_container_initial(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_delete_sql_container_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self._delete_sql_container_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [202, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        response_headers = {}
        if response.status_code == 202:
            response_headers['azure-AsyncOperation'] = self._deserialize('str', response.headers.get('azure-AsyncOperation'))
            response_headers['location'] = self._deserialize('str', response.headers.get('location'))
        if cls:
            return cls(pipeline_response, None, response_headers)
    _delete_sql_container_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}'}

    @distributed_trace
    def begin_delete_sql_container(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, **kwargs: Any) -> LROPoller[None]:
        if False:
            i = 10
            return i + 15
        'Deletes an existing Azure Cosmos DB SQL container.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either None or the result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[None]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._delete_sql_container_initial(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, api_version=api_version, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                for i in range(10):
                    print('nop')
            if cls:
                return cls(pipeline_response, None, {})
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_delete_sql_container.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}'}

    @distributed_trace
    def get_sql_container_throughput(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, **kwargs: Any) -> _models.ThroughputSettingsGetResults:
        if False:
            for i in range(10):
                print('nop')
        'Gets the RUs per second of the SQL container under an existing Azure Cosmos DB database\n        account.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ThroughputSettingsGetResults or the result of cls(response)\n        :rtype: ~azure.mgmt.cosmosdb.models.ThroughputSettingsGetResults\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.ThroughputSettingsGetResults] = kwargs.pop('cls', None)
        request = build_get_sql_container_throughput_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get_sql_container_throughput.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('ThroughputSettingsGetResults', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get_sql_container_throughput.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/throughputSettings/default'}

    def _update_sql_container_throughput_initial(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, update_throughput_parameters: Union[_models.ThroughputSettingsUpdateParameters, IO], **kwargs: Any) -> Optional[_models.ThroughputSettingsGetResults]:
        if False:
            i = 10
            return i + 15
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[Optional[_models.ThroughputSettingsGetResults]] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(update_throughput_parameters, (IOBase, bytes)):
            _content = update_throughput_parameters
        else:
            _json = self._serialize.body(update_throughput_parameters, 'ThroughputSettingsUpdateParameters')
        request = build_update_sql_container_throughput_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._update_sql_container_throughput_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = None
        response_headers = {}
        if response.status_code == 200:
            deserialized = self._deserialize('ThroughputSettingsGetResults', pipeline_response)
        if response.status_code == 202:
            response_headers['azure-AsyncOperation'] = self._deserialize('str', response.headers.get('azure-AsyncOperation'))
            response_headers['location'] = self._deserialize('str', response.headers.get('location'))
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    _update_sql_container_throughput_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/throughputSettings/default'}

    @overload
    def begin_update_sql_container_throughput(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, update_throughput_parameters: _models.ThroughputSettingsUpdateParameters, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.ThroughputSettingsGetResults]:
        if False:
            print('Hello World!')
        'Update RUs per second of an Azure Cosmos DB SQL container.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :param update_throughput_parameters: The parameters to provide for the RUs per second of the\n         current SQL container. Required.\n        :type update_throughput_parameters:\n         ~azure.mgmt.cosmosdb.models.ThroughputSettingsUpdateParameters\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either ThroughputSettingsGetResults or the\n         result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.ThroughputSettingsGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_update_sql_container_throughput(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, update_throughput_parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.ThroughputSettingsGetResults]:
        if False:
            print('Hello World!')
        'Update RUs per second of an Azure Cosmos DB SQL container.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :param update_throughput_parameters: The parameters to provide for the RUs per second of the\n         current SQL container. Required.\n        :type update_throughput_parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either ThroughputSettingsGetResults or the\n         result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.ThroughputSettingsGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_update_sql_container_throughput(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, update_throughput_parameters: Union[_models.ThroughputSettingsUpdateParameters, IO], **kwargs: Any) -> LROPoller[_models.ThroughputSettingsGetResults]:
        if False:
            for i in range(10):
                print('nop')
        "Update RUs per second of an Azure Cosmos DB SQL container.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :param update_throughput_parameters: The parameters to provide for the RUs per second of the\n         current SQL container. Is either a ThroughputSettingsUpdateParameters type or a IO type.\n         Required.\n        :type update_throughput_parameters:\n         ~azure.mgmt.cosmosdb.models.ThroughputSettingsUpdateParameters or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either ThroughputSettingsGetResults or the\n         result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.ThroughputSettingsGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.ThroughputSettingsGetResults] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._update_sql_container_throughput_initial(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, update_throughput_parameters=update_throughput_parameters, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                print('Hello World!')
            deserialized = self._deserialize('ThroughputSettingsGetResults', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_update_sql_container_throughput.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/throughputSettings/default'}

    def _migrate_sql_container_to_autoscale_initial(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, **kwargs: Any) -> Optional[_models.ThroughputSettingsGetResults]:
        if False:
            while True:
                i = 10
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[Optional[_models.ThroughputSettingsGetResults]] = kwargs.pop('cls', None)
        request = build_migrate_sql_container_to_autoscale_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self._migrate_sql_container_to_autoscale_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = None
        response_headers = {}
        if response.status_code == 200:
            deserialized = self._deserialize('ThroughputSettingsGetResults', pipeline_response)
        if response.status_code == 202:
            response_headers['azure-AsyncOperation'] = self._deserialize('str', response.headers.get('azure-AsyncOperation'))
            response_headers['location'] = self._deserialize('str', response.headers.get('location'))
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    _migrate_sql_container_to_autoscale_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/throughputSettings/default/migrateToAutoscale'}

    @distributed_trace
    def begin_migrate_sql_container_to_autoscale(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, **kwargs: Any) -> LROPoller[_models.ThroughputSettingsGetResults]:
        if False:
            return 10
        'Migrate an Azure Cosmos DB SQL container from manual throughput to autoscale.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either ThroughputSettingsGetResults or the\n         result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.ThroughputSettingsGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.ThroughputSettingsGetResults] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._migrate_sql_container_to_autoscale_initial(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, api_version=api_version, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                return 10
            deserialized = self._deserialize('ThroughputSettingsGetResults', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_migrate_sql_container_to_autoscale.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/throughputSettings/default/migrateToAutoscale'}

    def _migrate_sql_container_to_manual_throughput_initial(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, **kwargs: Any) -> Optional[_models.ThroughputSettingsGetResults]:
        if False:
            for i in range(10):
                print('nop')
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[Optional[_models.ThroughputSettingsGetResults]] = kwargs.pop('cls', None)
        request = build_migrate_sql_container_to_manual_throughput_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self._migrate_sql_container_to_manual_throughput_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = None
        response_headers = {}
        if response.status_code == 200:
            deserialized = self._deserialize('ThroughputSettingsGetResults', pipeline_response)
        if response.status_code == 202:
            response_headers['azure-AsyncOperation'] = self._deserialize('str', response.headers.get('azure-AsyncOperation'))
            response_headers['location'] = self._deserialize('str', response.headers.get('location'))
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    _migrate_sql_container_to_manual_throughput_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/throughputSettings/default/migrateToManualThroughput'}

    @distributed_trace
    def begin_migrate_sql_container_to_manual_throughput(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, **kwargs: Any) -> LROPoller[_models.ThroughputSettingsGetResults]:
        if False:
            print('Hello World!')
        'Migrate an Azure Cosmos DB SQL container from autoscale to manual throughput.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either ThroughputSettingsGetResults or the\n         result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.ThroughputSettingsGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.ThroughputSettingsGetResults] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._migrate_sql_container_to_manual_throughput_initial(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, api_version=api_version, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                print('Hello World!')
            deserialized = self._deserialize('ThroughputSettingsGetResults', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_migrate_sql_container_to_manual_throughput.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/throughputSettings/default/migrateToManualThroughput'}

    @distributed_trace
    def list_client_encryption_keys(self, resource_group_name: str, account_name: str, database_name: str, **kwargs: Any) -> Iterable['_models.ClientEncryptionKeyGetResults']:
        if False:
            i = 10
            return i + 15
        'Lists the ClientEncryptionKeys under an existing Azure Cosmos DB SQL database.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either ClientEncryptionKeyGetResults or the result of\n         cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.cosmosdb.models.ClientEncryptionKeyGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.ClientEncryptionKeysListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                i = 10
                return i + 15
            if not next_link:
                request = build_list_client_encryption_keys_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.list_client_encryption_keys.metadata['url'], headers=_headers, params=_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
            else:
                _parsed_next_link = urllib.parse.urlparse(next_link)
                _next_request_params = case_insensitive_dict({key: [urllib.parse.quote(v) for v in value] for (key, value) in urllib.parse.parse_qs(_parsed_next_link.query).items()})
                _next_request_params['api-version'] = self._config.api_version
                request = HttpRequest('GET', urllib.parse.urljoin(next_link, _parsed_next_link.path), params=_next_request_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
                request.method = 'GET'
            return request

        def extract_data(pipeline_response):
            if False:
                for i in range(10):
                    print('nop')
            deserialized = self._deserialize('ClientEncryptionKeysListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                print('Hello World!')
            request = prepare_request(next_link)
            _stream = False
            pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                raise HttpResponseError(response=response, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_client_encryption_keys.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/clientEncryptionKeys'}

    @distributed_trace
    def get_client_encryption_key(self, resource_group_name: str, account_name: str, database_name: str, client_encryption_key_name: str, **kwargs: Any) -> _models.ClientEncryptionKeyGetResults:
        if False:
            for i in range(10):
                print('nop')
        'Gets the ClientEncryptionKey under an existing Azure Cosmos DB SQL database.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param client_encryption_key_name: Cosmos DB ClientEncryptionKey name. Required.\n        :type client_encryption_key_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ClientEncryptionKeyGetResults or the result of cls(response)\n        :rtype: ~azure.mgmt.cosmosdb.models.ClientEncryptionKeyGetResults\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.ClientEncryptionKeyGetResults] = kwargs.pop('cls', None)
        request = build_get_client_encryption_key_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, client_encryption_key_name=client_encryption_key_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get_client_encryption_key.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('ClientEncryptionKeyGetResults', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get_client_encryption_key.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/clientEncryptionKeys/{clientEncryptionKeyName}'}

    def _create_update_client_encryption_key_initial(self, resource_group_name: str, account_name: str, database_name: str, client_encryption_key_name: str, create_update_client_encryption_key_parameters: Union[_models.ClientEncryptionKeyCreateUpdateParameters, IO], **kwargs: Any) -> Optional[_models.ClientEncryptionKeyGetResults]:
        if False:
            for i in range(10):
                print('nop')
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[Optional[_models.ClientEncryptionKeyGetResults]] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(create_update_client_encryption_key_parameters, (IOBase, bytes)):
            _content = create_update_client_encryption_key_parameters
        else:
            _json = self._serialize.body(create_update_client_encryption_key_parameters, 'ClientEncryptionKeyCreateUpdateParameters')
        request = build_create_update_client_encryption_key_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, client_encryption_key_name=client_encryption_key_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._create_update_client_encryption_key_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = None
        response_headers = {}
        if response.status_code == 200:
            deserialized = self._deserialize('ClientEncryptionKeyGetResults', pipeline_response)
        if response.status_code == 202:
            response_headers['azure-AsyncOperation'] = self._deserialize('str', response.headers.get('azure-AsyncOperation'))
            response_headers['location'] = self._deserialize('str', response.headers.get('location'))
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    _create_update_client_encryption_key_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/clientEncryptionKeys/{clientEncryptionKeyName}'}

    @overload
    def begin_create_update_client_encryption_key(self, resource_group_name: str, account_name: str, database_name: str, client_encryption_key_name: str, create_update_client_encryption_key_parameters: _models.ClientEncryptionKeyCreateUpdateParameters, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.ClientEncryptionKeyGetResults]:
        if False:
            while True:
                i = 10
        'Create or update a ClientEncryptionKey. This API is meant to be invoked via tools such as the\n        Azure Powershell (instead of directly).\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param client_encryption_key_name: Cosmos DB ClientEncryptionKey name. Required.\n        :type client_encryption_key_name: str\n        :param create_update_client_encryption_key_parameters: The parameters to provide for the client\n         encryption key. Required.\n        :type create_update_client_encryption_key_parameters:\n         ~azure.mgmt.cosmosdb.models.ClientEncryptionKeyCreateUpdateParameters\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either ClientEncryptionKeyGetResults or the\n         result of cls(response)\n        :rtype:\n         ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.ClientEncryptionKeyGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_create_update_client_encryption_key(self, resource_group_name: str, account_name: str, database_name: str, client_encryption_key_name: str, create_update_client_encryption_key_parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.ClientEncryptionKeyGetResults]:
        if False:
            return 10
        'Create or update a ClientEncryptionKey. This API is meant to be invoked via tools such as the\n        Azure Powershell (instead of directly).\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param client_encryption_key_name: Cosmos DB ClientEncryptionKey name. Required.\n        :type client_encryption_key_name: str\n        :param create_update_client_encryption_key_parameters: The parameters to provide for the client\n         encryption key. Required.\n        :type create_update_client_encryption_key_parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either ClientEncryptionKeyGetResults or the\n         result of cls(response)\n        :rtype:\n         ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.ClientEncryptionKeyGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_create_update_client_encryption_key(self, resource_group_name: str, account_name: str, database_name: str, client_encryption_key_name: str, create_update_client_encryption_key_parameters: Union[_models.ClientEncryptionKeyCreateUpdateParameters, IO], **kwargs: Any) -> LROPoller[_models.ClientEncryptionKeyGetResults]:
        if False:
            print('Hello World!')
        "Create or update a ClientEncryptionKey. This API is meant to be invoked via tools such as the\n        Azure Powershell (instead of directly).\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param client_encryption_key_name: Cosmos DB ClientEncryptionKey name. Required.\n        :type client_encryption_key_name: str\n        :param create_update_client_encryption_key_parameters: The parameters to provide for the client\n         encryption key. Is either a ClientEncryptionKeyCreateUpdateParameters type or a IO type.\n         Required.\n        :type create_update_client_encryption_key_parameters:\n         ~azure.mgmt.cosmosdb.models.ClientEncryptionKeyCreateUpdateParameters or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either ClientEncryptionKeyGetResults or the\n         result of cls(response)\n        :rtype:\n         ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.ClientEncryptionKeyGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.ClientEncryptionKeyGetResults] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._create_update_client_encryption_key_initial(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, client_encryption_key_name=client_encryption_key_name, create_update_client_encryption_key_parameters=create_update_client_encryption_key_parameters, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                i = 10
                return i + 15
            deserialized = self._deserialize('ClientEncryptionKeyGetResults', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_create_update_client_encryption_key.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/clientEncryptionKeys/{clientEncryptionKeyName}'}

    @distributed_trace
    def list_sql_stored_procedures(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, **kwargs: Any) -> Iterable['_models.SqlStoredProcedureGetResults']:
        if False:
            for i in range(10):
                print('nop')
        'Lists the SQL storedProcedure under an existing Azure Cosmos DB database account.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either SqlStoredProcedureGetResults or the result of\n         cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.cosmosdb.models.SqlStoredProcedureGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.SqlStoredProcedureListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                while True:
                    i = 10
            if not next_link:
                request = build_list_sql_stored_procedures_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.list_sql_stored_procedures.metadata['url'], headers=_headers, params=_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
            else:
                _parsed_next_link = urllib.parse.urlparse(next_link)
                _next_request_params = case_insensitive_dict({key: [urllib.parse.quote(v) for v in value] for (key, value) in urllib.parse.parse_qs(_parsed_next_link.query).items()})
                _next_request_params['api-version'] = self._config.api_version
                request = HttpRequest('GET', urllib.parse.urljoin(next_link, _parsed_next_link.path), params=_next_request_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
                request.method = 'GET'
            return request

        def extract_data(pipeline_response):
            if False:
                for i in range(10):
                    print('nop')
            deserialized = self._deserialize('SqlStoredProcedureListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                return 10
            request = prepare_request(next_link)
            _stream = False
            pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                raise HttpResponseError(response=response, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_sql_stored_procedures.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/storedProcedures'}

    @distributed_trace
    def get_sql_stored_procedure(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, stored_procedure_name: str, **kwargs: Any) -> _models.SqlStoredProcedureGetResults:
        if False:
            return 10
        'Gets the SQL storedProcedure under an existing Azure Cosmos DB database account.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :param stored_procedure_name: Cosmos DB storedProcedure name. Required.\n        :type stored_procedure_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: SqlStoredProcedureGetResults or the result of cls(response)\n        :rtype: ~azure.mgmt.cosmosdb.models.SqlStoredProcedureGetResults\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.SqlStoredProcedureGetResults] = kwargs.pop('cls', None)
        request = build_get_sql_stored_procedure_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, stored_procedure_name=stored_procedure_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get_sql_stored_procedure.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('SqlStoredProcedureGetResults', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get_sql_stored_procedure.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/storedProcedures/{storedProcedureName}'}

    def _create_update_sql_stored_procedure_initial(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, stored_procedure_name: str, create_update_sql_stored_procedure_parameters: Union[_models.SqlStoredProcedureCreateUpdateParameters, IO], **kwargs: Any) -> Optional[_models.SqlStoredProcedureGetResults]:
        if False:
            for i in range(10):
                print('nop')
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[Optional[_models.SqlStoredProcedureGetResults]] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(create_update_sql_stored_procedure_parameters, (IOBase, bytes)):
            _content = create_update_sql_stored_procedure_parameters
        else:
            _json = self._serialize.body(create_update_sql_stored_procedure_parameters, 'SqlStoredProcedureCreateUpdateParameters')
        request = build_create_update_sql_stored_procedure_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, stored_procedure_name=stored_procedure_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._create_update_sql_stored_procedure_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = None
        response_headers = {}
        if response.status_code == 200:
            deserialized = self._deserialize('SqlStoredProcedureGetResults', pipeline_response)
        if response.status_code == 202:
            response_headers['azure-AsyncOperation'] = self._deserialize('str', response.headers.get('azure-AsyncOperation'))
            response_headers['location'] = self._deserialize('str', response.headers.get('location'))
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    _create_update_sql_stored_procedure_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/storedProcedures/{storedProcedureName}'}

    @overload
    def begin_create_update_sql_stored_procedure(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, stored_procedure_name: str, create_update_sql_stored_procedure_parameters: _models.SqlStoredProcedureCreateUpdateParameters, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.SqlStoredProcedureGetResults]:
        if False:
            print('Hello World!')
        'Create or update an Azure Cosmos DB SQL storedProcedure.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :param stored_procedure_name: Cosmos DB storedProcedure name. Required.\n        :type stored_procedure_name: str\n        :param create_update_sql_stored_procedure_parameters: The parameters to provide for the current\n         SQL storedProcedure. Required.\n        :type create_update_sql_stored_procedure_parameters:\n         ~azure.mgmt.cosmosdb.models.SqlStoredProcedureCreateUpdateParameters\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either SqlStoredProcedureGetResults or the\n         result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.SqlStoredProcedureGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_create_update_sql_stored_procedure(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, stored_procedure_name: str, create_update_sql_stored_procedure_parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.SqlStoredProcedureGetResults]:
        if False:
            for i in range(10):
                print('nop')
        'Create or update an Azure Cosmos DB SQL storedProcedure.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :param stored_procedure_name: Cosmos DB storedProcedure name. Required.\n        :type stored_procedure_name: str\n        :param create_update_sql_stored_procedure_parameters: The parameters to provide for the current\n         SQL storedProcedure. Required.\n        :type create_update_sql_stored_procedure_parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either SqlStoredProcedureGetResults or the\n         result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.SqlStoredProcedureGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_create_update_sql_stored_procedure(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, stored_procedure_name: str, create_update_sql_stored_procedure_parameters: Union[_models.SqlStoredProcedureCreateUpdateParameters, IO], **kwargs: Any) -> LROPoller[_models.SqlStoredProcedureGetResults]:
        if False:
            i = 10
            return i + 15
        "Create or update an Azure Cosmos DB SQL storedProcedure.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :param stored_procedure_name: Cosmos DB storedProcedure name. Required.\n        :type stored_procedure_name: str\n        :param create_update_sql_stored_procedure_parameters: The parameters to provide for the current\n         SQL storedProcedure. Is either a SqlStoredProcedureCreateUpdateParameters type or a IO type.\n         Required.\n        :type create_update_sql_stored_procedure_parameters:\n         ~azure.mgmt.cosmosdb.models.SqlStoredProcedureCreateUpdateParameters or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either SqlStoredProcedureGetResults or the\n         result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.SqlStoredProcedureGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.SqlStoredProcedureGetResults] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._create_update_sql_stored_procedure_initial(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, stored_procedure_name=stored_procedure_name, create_update_sql_stored_procedure_parameters=create_update_sql_stored_procedure_parameters, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                while True:
                    i = 10
            deserialized = self._deserialize('SqlStoredProcedureGetResults', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_create_update_sql_stored_procedure.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/storedProcedures/{storedProcedureName}'}

    def _delete_sql_stored_procedure_initial(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, stored_procedure_name: str, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_delete_sql_stored_procedure_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, stored_procedure_name=stored_procedure_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self._delete_sql_stored_procedure_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [202, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        response_headers = {}
        if response.status_code == 202:
            response_headers['azure-AsyncOperation'] = self._deserialize('str', response.headers.get('azure-AsyncOperation'))
            response_headers['location'] = self._deserialize('str', response.headers.get('location'))
        if cls:
            return cls(pipeline_response, None, response_headers)
    _delete_sql_stored_procedure_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/storedProcedures/{storedProcedureName}'}

    @distributed_trace
    def begin_delete_sql_stored_procedure(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, stored_procedure_name: str, **kwargs: Any) -> LROPoller[None]:
        if False:
            while True:
                i = 10
        'Deletes an existing Azure Cosmos DB SQL storedProcedure.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :param stored_procedure_name: Cosmos DB storedProcedure name. Required.\n        :type stored_procedure_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either None or the result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[None]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._delete_sql_stored_procedure_initial(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, stored_procedure_name=stored_procedure_name, api_version=api_version, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                for i in range(10):
                    print('nop')
            if cls:
                return cls(pipeline_response, None, {})
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_delete_sql_stored_procedure.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/storedProcedures/{storedProcedureName}'}

    @distributed_trace
    def list_sql_user_defined_functions(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, **kwargs: Any) -> Iterable['_models.SqlUserDefinedFunctionGetResults']:
        if False:
            i = 10
            return i + 15
        'Lists the SQL userDefinedFunction under an existing Azure Cosmos DB database account.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either SqlUserDefinedFunctionGetResults or the result of\n         cls(response)\n        :rtype:\n         ~azure.core.paging.ItemPaged[~azure.mgmt.cosmosdb.models.SqlUserDefinedFunctionGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.SqlUserDefinedFunctionListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                return 10
            if not next_link:
                request = build_list_sql_user_defined_functions_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.list_sql_user_defined_functions.metadata['url'], headers=_headers, params=_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
            else:
                _parsed_next_link = urllib.parse.urlparse(next_link)
                _next_request_params = case_insensitive_dict({key: [urllib.parse.quote(v) for v in value] for (key, value) in urllib.parse.parse_qs(_parsed_next_link.query).items()})
                _next_request_params['api-version'] = self._config.api_version
                request = HttpRequest('GET', urllib.parse.urljoin(next_link, _parsed_next_link.path), params=_next_request_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
                request.method = 'GET'
            return request

        def extract_data(pipeline_response):
            if False:
                for i in range(10):
                    print('nop')
            deserialized = self._deserialize('SqlUserDefinedFunctionListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                while True:
                    i = 10
            request = prepare_request(next_link)
            _stream = False
            pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                raise HttpResponseError(response=response, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_sql_user_defined_functions.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/userDefinedFunctions'}

    @distributed_trace
    def get_sql_user_defined_function(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, user_defined_function_name: str, **kwargs: Any) -> _models.SqlUserDefinedFunctionGetResults:
        if False:
            print('Hello World!')
        'Gets the SQL userDefinedFunction under an existing Azure Cosmos DB database account.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :param user_defined_function_name: Cosmos DB userDefinedFunction name. Required.\n        :type user_defined_function_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: SqlUserDefinedFunctionGetResults or the result of cls(response)\n        :rtype: ~azure.mgmt.cosmosdb.models.SqlUserDefinedFunctionGetResults\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.SqlUserDefinedFunctionGetResults] = kwargs.pop('cls', None)
        request = build_get_sql_user_defined_function_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, user_defined_function_name=user_defined_function_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get_sql_user_defined_function.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('SqlUserDefinedFunctionGetResults', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get_sql_user_defined_function.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/userDefinedFunctions/{userDefinedFunctionName}'}

    def _create_update_sql_user_defined_function_initial(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, user_defined_function_name: str, create_update_sql_user_defined_function_parameters: Union[_models.SqlUserDefinedFunctionCreateUpdateParameters, IO], **kwargs: Any) -> Optional[_models.SqlUserDefinedFunctionGetResults]:
        if False:
            i = 10
            return i + 15
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[Optional[_models.SqlUserDefinedFunctionGetResults]] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(create_update_sql_user_defined_function_parameters, (IOBase, bytes)):
            _content = create_update_sql_user_defined_function_parameters
        else:
            _json = self._serialize.body(create_update_sql_user_defined_function_parameters, 'SqlUserDefinedFunctionCreateUpdateParameters')
        request = build_create_update_sql_user_defined_function_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, user_defined_function_name=user_defined_function_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._create_update_sql_user_defined_function_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = None
        response_headers = {}
        if response.status_code == 200:
            deserialized = self._deserialize('SqlUserDefinedFunctionGetResults', pipeline_response)
        if response.status_code == 202:
            response_headers['azure-AsyncOperation'] = self._deserialize('str', response.headers.get('azure-AsyncOperation'))
            response_headers['location'] = self._deserialize('str', response.headers.get('location'))
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    _create_update_sql_user_defined_function_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/userDefinedFunctions/{userDefinedFunctionName}'}

    @overload
    def begin_create_update_sql_user_defined_function(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, user_defined_function_name: str, create_update_sql_user_defined_function_parameters: _models.SqlUserDefinedFunctionCreateUpdateParameters, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.SqlUserDefinedFunctionGetResults]:
        if False:
            print('Hello World!')
        'Create or update an Azure Cosmos DB SQL userDefinedFunction.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :param user_defined_function_name: Cosmos DB userDefinedFunction name. Required.\n        :type user_defined_function_name: str\n        :param create_update_sql_user_defined_function_parameters: The parameters to provide for the\n         current SQL userDefinedFunction. Required.\n        :type create_update_sql_user_defined_function_parameters:\n         ~azure.mgmt.cosmosdb.models.SqlUserDefinedFunctionCreateUpdateParameters\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either SqlUserDefinedFunctionGetResults or the\n         result of cls(response)\n        :rtype:\n         ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.SqlUserDefinedFunctionGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_create_update_sql_user_defined_function(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, user_defined_function_name: str, create_update_sql_user_defined_function_parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.SqlUserDefinedFunctionGetResults]:
        if False:
            i = 10
            return i + 15
        'Create or update an Azure Cosmos DB SQL userDefinedFunction.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :param user_defined_function_name: Cosmos DB userDefinedFunction name. Required.\n        :type user_defined_function_name: str\n        :param create_update_sql_user_defined_function_parameters: The parameters to provide for the\n         current SQL userDefinedFunction. Required.\n        :type create_update_sql_user_defined_function_parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either SqlUserDefinedFunctionGetResults or the\n         result of cls(response)\n        :rtype:\n         ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.SqlUserDefinedFunctionGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_create_update_sql_user_defined_function(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, user_defined_function_name: str, create_update_sql_user_defined_function_parameters: Union[_models.SqlUserDefinedFunctionCreateUpdateParameters, IO], **kwargs: Any) -> LROPoller[_models.SqlUserDefinedFunctionGetResults]:
        if False:
            while True:
                i = 10
        "Create or update an Azure Cosmos DB SQL userDefinedFunction.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :param user_defined_function_name: Cosmos DB userDefinedFunction name. Required.\n        :type user_defined_function_name: str\n        :param create_update_sql_user_defined_function_parameters: The parameters to provide for the\n         current SQL userDefinedFunction. Is either a SqlUserDefinedFunctionCreateUpdateParameters type\n         or a IO type. Required.\n        :type create_update_sql_user_defined_function_parameters:\n         ~azure.mgmt.cosmosdb.models.SqlUserDefinedFunctionCreateUpdateParameters or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either SqlUserDefinedFunctionGetResults or the\n         result of cls(response)\n        :rtype:\n         ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.SqlUserDefinedFunctionGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.SqlUserDefinedFunctionGetResults] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._create_update_sql_user_defined_function_initial(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, user_defined_function_name=user_defined_function_name, create_update_sql_user_defined_function_parameters=create_update_sql_user_defined_function_parameters, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                return 10
            deserialized = self._deserialize('SqlUserDefinedFunctionGetResults', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_create_update_sql_user_defined_function.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/userDefinedFunctions/{userDefinedFunctionName}'}

    def _delete_sql_user_defined_function_initial(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, user_defined_function_name: str, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_delete_sql_user_defined_function_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, user_defined_function_name=user_defined_function_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self._delete_sql_user_defined_function_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [202, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        response_headers = {}
        if response.status_code == 202:
            response_headers['azure-AsyncOperation'] = self._deserialize('str', response.headers.get('azure-AsyncOperation'))
            response_headers['location'] = self._deserialize('str', response.headers.get('location'))
        if cls:
            return cls(pipeline_response, None, response_headers)
    _delete_sql_user_defined_function_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/userDefinedFunctions/{userDefinedFunctionName}'}

    @distributed_trace
    def begin_delete_sql_user_defined_function(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, user_defined_function_name: str, **kwargs: Any) -> LROPoller[None]:
        if False:
            i = 10
            return i + 15
        'Deletes an existing Azure Cosmos DB SQL userDefinedFunction.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :param user_defined_function_name: Cosmos DB userDefinedFunction name. Required.\n        :type user_defined_function_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either None or the result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[None]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._delete_sql_user_defined_function_initial(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, user_defined_function_name=user_defined_function_name, api_version=api_version, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                while True:
                    i = 10
            if cls:
                return cls(pipeline_response, None, {})
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_delete_sql_user_defined_function.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/userDefinedFunctions/{userDefinedFunctionName}'}

    @distributed_trace
    def list_sql_triggers(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, **kwargs: Any) -> Iterable['_models.SqlTriggerGetResults']:
        if False:
            return 10
        'Lists the SQL trigger under an existing Azure Cosmos DB database account.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either SqlTriggerGetResults or the result of\n         cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.cosmosdb.models.SqlTriggerGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.SqlTriggerListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                for i in range(10):
                    print('nop')
            if not next_link:
                request = build_list_sql_triggers_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.list_sql_triggers.metadata['url'], headers=_headers, params=_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
            else:
                _parsed_next_link = urllib.parse.urlparse(next_link)
                _next_request_params = case_insensitive_dict({key: [urllib.parse.quote(v) for v in value] for (key, value) in urllib.parse.parse_qs(_parsed_next_link.query).items()})
                _next_request_params['api-version'] = self._config.api_version
                request = HttpRequest('GET', urllib.parse.urljoin(next_link, _parsed_next_link.path), params=_next_request_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
                request.method = 'GET'
            return request

        def extract_data(pipeline_response):
            if False:
                return 10
            deserialized = self._deserialize('SqlTriggerListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                i = 10
                return i + 15
            request = prepare_request(next_link)
            _stream = False
            pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                raise HttpResponseError(response=response, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_sql_triggers.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/triggers'}

    @distributed_trace
    def get_sql_trigger(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, trigger_name: str, **kwargs: Any) -> _models.SqlTriggerGetResults:
        if False:
            i = 10
            return i + 15
        'Gets the SQL trigger under an existing Azure Cosmos DB database account.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :param trigger_name: Cosmos DB trigger name. Required.\n        :type trigger_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: SqlTriggerGetResults or the result of cls(response)\n        :rtype: ~azure.mgmt.cosmosdb.models.SqlTriggerGetResults\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.SqlTriggerGetResults] = kwargs.pop('cls', None)
        request = build_get_sql_trigger_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, trigger_name=trigger_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get_sql_trigger.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('SqlTriggerGetResults', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get_sql_trigger.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/triggers/{triggerName}'}

    def _create_update_sql_trigger_initial(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, trigger_name: str, create_update_sql_trigger_parameters: Union[_models.SqlTriggerCreateUpdateParameters, IO], **kwargs: Any) -> Optional[_models.SqlTriggerGetResults]:
        if False:
            return 10
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[Optional[_models.SqlTriggerGetResults]] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(create_update_sql_trigger_parameters, (IOBase, bytes)):
            _content = create_update_sql_trigger_parameters
        else:
            _json = self._serialize.body(create_update_sql_trigger_parameters, 'SqlTriggerCreateUpdateParameters')
        request = build_create_update_sql_trigger_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, trigger_name=trigger_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._create_update_sql_trigger_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = None
        response_headers = {}
        if response.status_code == 200:
            deserialized = self._deserialize('SqlTriggerGetResults', pipeline_response)
        if response.status_code == 202:
            response_headers['azure-AsyncOperation'] = self._deserialize('str', response.headers.get('azure-AsyncOperation'))
            response_headers['location'] = self._deserialize('str', response.headers.get('location'))
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    _create_update_sql_trigger_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/triggers/{triggerName}'}

    @overload
    def begin_create_update_sql_trigger(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, trigger_name: str, create_update_sql_trigger_parameters: _models.SqlTriggerCreateUpdateParameters, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.SqlTriggerGetResults]:
        if False:
            print('Hello World!')
        'Create or update an Azure Cosmos DB SQL trigger.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :param trigger_name: Cosmos DB trigger name. Required.\n        :type trigger_name: str\n        :param create_update_sql_trigger_parameters: The parameters to provide for the current SQL\n         trigger. Required.\n        :type create_update_sql_trigger_parameters:\n         ~azure.mgmt.cosmosdb.models.SqlTriggerCreateUpdateParameters\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either SqlTriggerGetResults or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.SqlTriggerGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_create_update_sql_trigger(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, trigger_name: str, create_update_sql_trigger_parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.SqlTriggerGetResults]:
        if False:
            return 10
        'Create or update an Azure Cosmos DB SQL trigger.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :param trigger_name: Cosmos DB trigger name. Required.\n        :type trigger_name: str\n        :param create_update_sql_trigger_parameters: The parameters to provide for the current SQL\n         trigger. Required.\n        :type create_update_sql_trigger_parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either SqlTriggerGetResults or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.SqlTriggerGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_create_update_sql_trigger(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, trigger_name: str, create_update_sql_trigger_parameters: Union[_models.SqlTriggerCreateUpdateParameters, IO], **kwargs: Any) -> LROPoller[_models.SqlTriggerGetResults]:
        if False:
            return 10
        "Create or update an Azure Cosmos DB SQL trigger.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :param trigger_name: Cosmos DB trigger name. Required.\n        :type trigger_name: str\n        :param create_update_sql_trigger_parameters: The parameters to provide for the current SQL\n         trigger. Is either a SqlTriggerCreateUpdateParameters type or a IO type. Required.\n        :type create_update_sql_trigger_parameters:\n         ~azure.mgmt.cosmosdb.models.SqlTriggerCreateUpdateParameters or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either SqlTriggerGetResults or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.SqlTriggerGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.SqlTriggerGetResults] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._create_update_sql_trigger_initial(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, trigger_name=trigger_name, create_update_sql_trigger_parameters=create_update_sql_trigger_parameters, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                print('Hello World!')
            deserialized = self._deserialize('SqlTriggerGetResults', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_create_update_sql_trigger.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/triggers/{triggerName}'}

    def _delete_sql_trigger_initial(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, trigger_name: str, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_delete_sql_trigger_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, trigger_name=trigger_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self._delete_sql_trigger_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [202, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        response_headers = {}
        if response.status_code == 202:
            response_headers['azure-AsyncOperation'] = self._deserialize('str', response.headers.get('azure-AsyncOperation'))
            response_headers['location'] = self._deserialize('str', response.headers.get('location'))
        if cls:
            return cls(pipeline_response, None, response_headers)
    _delete_sql_trigger_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/triggers/{triggerName}'}

    @distributed_trace
    def begin_delete_sql_trigger(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, trigger_name: str, **kwargs: Any) -> LROPoller[None]:
        if False:
            while True:
                i = 10
        'Deletes an existing Azure Cosmos DB SQL trigger.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :param trigger_name: Cosmos DB trigger name. Required.\n        :type trigger_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either None or the result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[None]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._delete_sql_trigger_initial(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, trigger_name=trigger_name, api_version=api_version, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                for i in range(10):
                    print('nop')
            if cls:
                return cls(pipeline_response, None, {})
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_delete_sql_trigger.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/triggers/{triggerName}'}

    @distributed_trace
    def get_sql_role_definition(self, role_definition_id: str, resource_group_name: str, account_name: str, **kwargs: Any) -> _models.SqlRoleDefinitionGetResults:
        if False:
            return 10
        'Retrieves the properties of an existing Azure Cosmos DB SQL Role Definition with the given Id.\n\n        :param role_definition_id: The GUID for the Role Definition. Required.\n        :type role_definition_id: str\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: SqlRoleDefinitionGetResults or the result of cls(response)\n        :rtype: ~azure.mgmt.cosmosdb.models.SqlRoleDefinitionGetResults\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.SqlRoleDefinitionGetResults] = kwargs.pop('cls', None)
        request = build_get_sql_role_definition_request(role_definition_id=role_definition_id, resource_group_name=resource_group_name, account_name=account_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get_sql_role_definition.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('SqlRoleDefinitionGetResults', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get_sql_role_definition.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlRoleDefinitions/{roleDefinitionId}'}

    def _create_update_sql_role_definition_initial(self, role_definition_id: str, resource_group_name: str, account_name: str, create_update_sql_role_definition_parameters: Union[_models.SqlRoleDefinitionCreateUpdateParameters, IO], **kwargs: Any) -> Optional[_models.SqlRoleDefinitionGetResults]:
        if False:
            i = 10
            return i + 15
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[Optional[_models.SqlRoleDefinitionGetResults]] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(create_update_sql_role_definition_parameters, (IOBase, bytes)):
            _content = create_update_sql_role_definition_parameters
        else:
            _json = self._serialize.body(create_update_sql_role_definition_parameters, 'SqlRoleDefinitionCreateUpdateParameters')
        request = build_create_update_sql_role_definition_request(role_definition_id=role_definition_id, resource_group_name=resource_group_name, account_name=account_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._create_update_sql_role_definition_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('SqlRoleDefinitionGetResults', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    _create_update_sql_role_definition_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlRoleDefinitions/{roleDefinitionId}'}

    @overload
    def begin_create_update_sql_role_definition(self, role_definition_id: str, resource_group_name: str, account_name: str, create_update_sql_role_definition_parameters: _models.SqlRoleDefinitionCreateUpdateParameters, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.SqlRoleDefinitionGetResults]:
        if False:
            while True:
                i = 10
        'Creates or updates an Azure Cosmos DB SQL Role Definition.\n\n        :param role_definition_id: The GUID for the Role Definition. Required.\n        :type role_definition_id: str\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param create_update_sql_role_definition_parameters: The properties required to create or\n         update a Role Definition. Required.\n        :type create_update_sql_role_definition_parameters:\n         ~azure.mgmt.cosmosdb.models.SqlRoleDefinitionCreateUpdateParameters\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either SqlRoleDefinitionGetResults or the result\n         of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.SqlRoleDefinitionGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_create_update_sql_role_definition(self, role_definition_id: str, resource_group_name: str, account_name: str, create_update_sql_role_definition_parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.SqlRoleDefinitionGetResults]:
        if False:
            i = 10
            return i + 15
        'Creates or updates an Azure Cosmos DB SQL Role Definition.\n\n        :param role_definition_id: The GUID for the Role Definition. Required.\n        :type role_definition_id: str\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param create_update_sql_role_definition_parameters: The properties required to create or\n         update a Role Definition. Required.\n        :type create_update_sql_role_definition_parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either SqlRoleDefinitionGetResults or the result\n         of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.SqlRoleDefinitionGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_create_update_sql_role_definition(self, role_definition_id: str, resource_group_name: str, account_name: str, create_update_sql_role_definition_parameters: Union[_models.SqlRoleDefinitionCreateUpdateParameters, IO], **kwargs: Any) -> LROPoller[_models.SqlRoleDefinitionGetResults]:
        if False:
            for i in range(10):
                print('nop')
        "Creates or updates an Azure Cosmos DB SQL Role Definition.\n\n        :param role_definition_id: The GUID for the Role Definition. Required.\n        :type role_definition_id: str\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param create_update_sql_role_definition_parameters: The properties required to create or\n         update a Role Definition. Is either a SqlRoleDefinitionCreateUpdateParameters type or a IO\n         type. Required.\n        :type create_update_sql_role_definition_parameters:\n         ~azure.mgmt.cosmosdb.models.SqlRoleDefinitionCreateUpdateParameters or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either SqlRoleDefinitionGetResults or the result\n         of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.SqlRoleDefinitionGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.SqlRoleDefinitionGetResults] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._create_update_sql_role_definition_initial(role_definition_id=role_definition_id, resource_group_name=resource_group_name, account_name=account_name, create_update_sql_role_definition_parameters=create_update_sql_role_definition_parameters, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                print('Hello World!')
            deserialized = self._deserialize('SqlRoleDefinitionGetResults', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_create_update_sql_role_definition.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlRoleDefinitions/{roleDefinitionId}'}

    def _delete_sql_role_definition_initial(self, role_definition_id: str, resource_group_name: str, account_name: str, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_delete_sql_role_definition_request(role_definition_id=role_definition_id, resource_group_name=resource_group_name, account_name=account_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self._delete_sql_role_definition_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    _delete_sql_role_definition_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlRoleDefinitions/{roleDefinitionId}'}

    @distributed_trace
    def begin_delete_sql_role_definition(self, role_definition_id: str, resource_group_name: str, account_name: str, **kwargs: Any) -> LROPoller[None]:
        if False:
            i = 10
            return i + 15
        'Deletes an existing Azure Cosmos DB SQL Role Definition.\n\n        :param role_definition_id: The GUID for the Role Definition. Required.\n        :type role_definition_id: str\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either None or the result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[None]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._delete_sql_role_definition_initial(role_definition_id=role_definition_id, resource_group_name=resource_group_name, account_name=account_name, api_version=api_version, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                print('Hello World!')
            if cls:
                return cls(pipeline_response, None, {})
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_delete_sql_role_definition.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlRoleDefinitions/{roleDefinitionId}'}

    @distributed_trace
    def list_sql_role_definitions(self, resource_group_name: str, account_name: str, **kwargs: Any) -> Iterable['_models.SqlRoleDefinitionGetResults']:
        if False:
            print('Hello World!')
        'Retrieves the list of all Azure Cosmos DB SQL Role Definitions.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either SqlRoleDefinitionGetResults or the result of\n         cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.cosmosdb.models.SqlRoleDefinitionGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.SqlRoleDefinitionListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                i = 10
                return i + 15
            if not next_link:
                request = build_list_sql_role_definitions_request(resource_group_name=resource_group_name, account_name=account_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.list_sql_role_definitions.metadata['url'], headers=_headers, params=_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
            else:
                _parsed_next_link = urllib.parse.urlparse(next_link)
                _next_request_params = case_insensitive_dict({key: [urllib.parse.quote(v) for v in value] for (key, value) in urllib.parse.parse_qs(_parsed_next_link.query).items()})
                _next_request_params['api-version'] = self._config.api_version
                request = HttpRequest('GET', urllib.parse.urljoin(next_link, _parsed_next_link.path), params=_next_request_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
                request.method = 'GET'
            return request

        def extract_data(pipeline_response):
            if False:
                print('Hello World!')
            deserialized = self._deserialize('SqlRoleDefinitionListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                while True:
                    i = 10
            request = prepare_request(next_link)
            _stream = False
            pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                raise HttpResponseError(response=response, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_sql_role_definitions.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlRoleDefinitions'}

    @distributed_trace
    def get_sql_role_assignment(self, role_assignment_id: str, resource_group_name: str, account_name: str, **kwargs: Any) -> _models.SqlRoleAssignmentGetResults:
        if False:
            return 10
        'Retrieves the properties of an existing Azure Cosmos DB SQL Role Assignment with the given Id.\n\n        :param role_assignment_id: The GUID for the Role Assignment. Required.\n        :type role_assignment_id: str\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: SqlRoleAssignmentGetResults or the result of cls(response)\n        :rtype: ~azure.mgmt.cosmosdb.models.SqlRoleAssignmentGetResults\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.SqlRoleAssignmentGetResults] = kwargs.pop('cls', None)
        request = build_get_sql_role_assignment_request(role_assignment_id=role_assignment_id, resource_group_name=resource_group_name, account_name=account_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get_sql_role_assignment.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('SqlRoleAssignmentGetResults', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get_sql_role_assignment.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlRoleAssignments/{roleAssignmentId}'}

    def _create_update_sql_role_assignment_initial(self, role_assignment_id: str, resource_group_name: str, account_name: str, create_update_sql_role_assignment_parameters: Union[_models.SqlRoleAssignmentCreateUpdateParameters, IO], **kwargs: Any) -> Optional[_models.SqlRoleAssignmentGetResults]:
        if False:
            for i in range(10):
                print('nop')
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[Optional[_models.SqlRoleAssignmentGetResults]] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(create_update_sql_role_assignment_parameters, (IOBase, bytes)):
            _content = create_update_sql_role_assignment_parameters
        else:
            _json = self._serialize.body(create_update_sql_role_assignment_parameters, 'SqlRoleAssignmentCreateUpdateParameters')
        request = build_create_update_sql_role_assignment_request(role_assignment_id=role_assignment_id, resource_group_name=resource_group_name, account_name=account_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._create_update_sql_role_assignment_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('SqlRoleAssignmentGetResults', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    _create_update_sql_role_assignment_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlRoleAssignments/{roleAssignmentId}'}

    @overload
    def begin_create_update_sql_role_assignment(self, role_assignment_id: str, resource_group_name: str, account_name: str, create_update_sql_role_assignment_parameters: _models.SqlRoleAssignmentCreateUpdateParameters, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.SqlRoleAssignmentGetResults]:
        if False:
            for i in range(10):
                print('nop')
        'Creates or updates an Azure Cosmos DB SQL Role Assignment.\n\n        :param role_assignment_id: The GUID for the Role Assignment. Required.\n        :type role_assignment_id: str\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param create_update_sql_role_assignment_parameters: The properties required to create or\n         update a Role Assignment. Required.\n        :type create_update_sql_role_assignment_parameters:\n         ~azure.mgmt.cosmosdb.models.SqlRoleAssignmentCreateUpdateParameters\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either SqlRoleAssignmentGetResults or the result\n         of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.SqlRoleAssignmentGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_create_update_sql_role_assignment(self, role_assignment_id: str, resource_group_name: str, account_name: str, create_update_sql_role_assignment_parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.SqlRoleAssignmentGetResults]:
        if False:
            for i in range(10):
                print('nop')
        'Creates or updates an Azure Cosmos DB SQL Role Assignment.\n\n        :param role_assignment_id: The GUID for the Role Assignment. Required.\n        :type role_assignment_id: str\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param create_update_sql_role_assignment_parameters: The properties required to create or\n         update a Role Assignment. Required.\n        :type create_update_sql_role_assignment_parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either SqlRoleAssignmentGetResults or the result\n         of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.SqlRoleAssignmentGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_create_update_sql_role_assignment(self, role_assignment_id: str, resource_group_name: str, account_name: str, create_update_sql_role_assignment_parameters: Union[_models.SqlRoleAssignmentCreateUpdateParameters, IO], **kwargs: Any) -> LROPoller[_models.SqlRoleAssignmentGetResults]:
        if False:
            for i in range(10):
                print('nop')
        "Creates or updates an Azure Cosmos DB SQL Role Assignment.\n\n        :param role_assignment_id: The GUID for the Role Assignment. Required.\n        :type role_assignment_id: str\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param create_update_sql_role_assignment_parameters: The properties required to create or\n         update a Role Assignment. Is either a SqlRoleAssignmentCreateUpdateParameters type or a IO\n         type. Required.\n        :type create_update_sql_role_assignment_parameters:\n         ~azure.mgmt.cosmosdb.models.SqlRoleAssignmentCreateUpdateParameters or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either SqlRoleAssignmentGetResults or the result\n         of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.SqlRoleAssignmentGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.SqlRoleAssignmentGetResults] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._create_update_sql_role_assignment_initial(role_assignment_id=role_assignment_id, resource_group_name=resource_group_name, account_name=account_name, create_update_sql_role_assignment_parameters=create_update_sql_role_assignment_parameters, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                return 10
            deserialized = self._deserialize('SqlRoleAssignmentGetResults', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_create_update_sql_role_assignment.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlRoleAssignments/{roleAssignmentId}'}

    def _delete_sql_role_assignment_initial(self, role_assignment_id: str, resource_group_name: str, account_name: str, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_delete_sql_role_assignment_request(role_assignment_id=role_assignment_id, resource_group_name=resource_group_name, account_name=account_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self._delete_sql_role_assignment_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    _delete_sql_role_assignment_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlRoleAssignments/{roleAssignmentId}'}

    @distributed_trace
    def begin_delete_sql_role_assignment(self, role_assignment_id: str, resource_group_name: str, account_name: str, **kwargs: Any) -> LROPoller[None]:
        if False:
            for i in range(10):
                print('nop')
        'Deletes an existing Azure Cosmos DB SQL Role Assignment.\n\n        :param role_assignment_id: The GUID for the Role Assignment. Required.\n        :type role_assignment_id: str\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either None or the result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[None]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._delete_sql_role_assignment_initial(role_assignment_id=role_assignment_id, resource_group_name=resource_group_name, account_name=account_name, api_version=api_version, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                print('Hello World!')
            if cls:
                return cls(pipeline_response, None, {})
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_delete_sql_role_assignment.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlRoleAssignments/{roleAssignmentId}'}

    @distributed_trace
    def list_sql_role_assignments(self, resource_group_name: str, account_name: str, **kwargs: Any) -> Iterable['_models.SqlRoleAssignmentGetResults']:
        if False:
            i = 10
            return i + 15
        'Retrieves the list of all Azure Cosmos DB SQL Role Assignments.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either SqlRoleAssignmentGetResults or the result of\n         cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.cosmosdb.models.SqlRoleAssignmentGetResults]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.SqlRoleAssignmentListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                for i in range(10):
                    print('nop')
            if not next_link:
                request = build_list_sql_role_assignments_request(resource_group_name=resource_group_name, account_name=account_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.list_sql_role_assignments.metadata['url'], headers=_headers, params=_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
            else:
                _parsed_next_link = urllib.parse.urlparse(next_link)
                _next_request_params = case_insensitive_dict({key: [urllib.parse.quote(v) for v in value] for (key, value) in urllib.parse.parse_qs(_parsed_next_link.query).items()})
                _next_request_params['api-version'] = self._config.api_version
                request = HttpRequest('GET', urllib.parse.urljoin(next_link, _parsed_next_link.path), params=_next_request_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
                request.method = 'GET'
            return request

        def extract_data(pipeline_response):
            if False:
                while True:
                    i = 10
            deserialized = self._deserialize('SqlRoleAssignmentListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                return 10
            request = prepare_request(next_link)
            _stream = False
            pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                raise HttpResponseError(response=response, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_sql_role_assignments.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlRoleAssignments'}

    def _retrieve_continuous_backup_information_initial(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, location: Union[_models.ContinuousBackupRestoreLocation, IO], **kwargs: Any) -> Optional[_models.BackupInformation]:
        if False:
            print('Hello World!')
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[Optional[_models.BackupInformation]] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(location, (IOBase, bytes)):
            _content = location
        else:
            _json = self._serialize.body(location, 'ContinuousBackupRestoreLocation')
        request = build_retrieve_continuous_backup_information_request(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._retrieve_continuous_backup_information_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('BackupInformation', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    _retrieve_continuous_backup_information_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/retrieveContinuousBackupInformation'}

    @overload
    def begin_retrieve_continuous_backup_information(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, location: _models.ContinuousBackupRestoreLocation, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.BackupInformation]:
        if False:
            return 10
        'Retrieves continuous backup information for a container resource.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :param location: The name of the continuous backup restore location. Required.\n        :type location: ~azure.mgmt.cosmosdb.models.ContinuousBackupRestoreLocation\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either BackupInformation or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.BackupInformation]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_retrieve_continuous_backup_information(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, location: IO, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.BackupInformation]:
        if False:
            return 10
        'Retrieves continuous backup information for a container resource.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :param location: The name of the continuous backup restore location. Required.\n        :type location: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either BackupInformation or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.BackupInformation]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_retrieve_continuous_backup_information(self, resource_group_name: str, account_name: str, database_name: str, container_name: str, location: Union[_models.ContinuousBackupRestoreLocation, IO], **kwargs: Any) -> LROPoller[_models.BackupInformation]:
        if False:
            for i in range(10):
                print('nop')
        "Retrieves continuous backup information for a container resource.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param account_name: Cosmos DB database account name. Required.\n        :type account_name: str\n        :param database_name: Cosmos DB database name. Required.\n        :type database_name: str\n        :param container_name: Cosmos DB container name. Required.\n        :type container_name: str\n        :param location: The name of the continuous backup restore location. Is either a\n         ContinuousBackupRestoreLocation type or a IO type. Required.\n        :type location: ~azure.mgmt.cosmosdb.models.ContinuousBackupRestoreLocation or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either BackupInformation or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.cosmosdb.models.BackupInformation]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.BackupInformation] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._retrieve_continuous_backup_information_initial(resource_group_name=resource_group_name, account_name=account_name, database_name=database_name, container_name=container_name, location=location, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                return 10
            deserialized = self._deserialize('BackupInformation', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, ARMPolling(lro_delay, lro_options={'final-state-via': 'location'}, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_retrieve_continuous_backup_information.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/sqlDatabases/{databaseName}/containers/{containerName}/retrieveContinuousBackupInformation'}