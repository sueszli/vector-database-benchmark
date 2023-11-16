import sys
from typing import Any, Callable, Dict, IO, Iterable, Optional, TypeVar, Union, overload
import urllib.parse
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceExistsError, ResourceNotFoundError, ResourceNotModifiedError, map_error
from azure.core.paging import ItemPaged
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import HttpResponse
from azure.core.rest import HttpRequest
from azure.core.tracing.decorator import distributed_trace
from azure.core.utils import case_insensitive_dict
from azure.mgmt.core.exceptions import ARMErrorFormat
from .. import models as _models
from .._serialization import Serializer
from .._vendor import _convert_request, _format_url_section
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_list_by_account_request(resource_group_name: str, account_name: str, subscription_id: str, *, filter: Optional[str]=None, top: Optional[int]=None, skip: Optional[int]=None, select: Optional[str]=None, orderby: Optional[str]=None, count: Optional[bool]=None, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2019-11-01-preview'] = kwargs.pop('api_version', _params.pop('api-version', '2019-11-01-preview'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataLakeAnalytics/accounts/{accountName}/storageAccounts')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'accountName': _SERIALIZER.url('account_name', account_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    if filter is not None:
        _params['$filter'] = _SERIALIZER.query('filter', filter, 'str')
    if top is not None:
        _params['$top'] = _SERIALIZER.query('top', top, 'int', minimum=1)
    if skip is not None:
        _params['$skip'] = _SERIALIZER.query('skip', skip, 'int', minimum=1)
    if select is not None:
        _params['$select'] = _SERIALIZER.query('select', select, 'str')
    if orderby is not None:
        _params['$orderby'] = _SERIALIZER.query('orderby', orderby, 'str')
    if count is not None:
        _params['$count'] = _SERIALIZER.query('count', count, 'bool')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_add_request(resource_group_name: str, account_name: str, storage_account_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2019-11-01-preview'] = kwargs.pop('api_version', _params.pop('api-version', '2019-11-01-preview'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataLakeAnalytics/accounts/{accountName}/storageAccounts/{storageAccountName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'accountName': _SERIALIZER.url('account_name', account_name, 'str'), 'storageAccountName': _SERIALIZER.url('storage_account_name', storage_account_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_request(resource_group_name: str, account_name: str, storage_account_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2019-11-01-preview'] = kwargs.pop('api_version', _params.pop('api-version', '2019-11-01-preview'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataLakeAnalytics/accounts/{accountName}/storageAccounts/{storageAccountName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'accountName': _SERIALIZER.url('account_name', account_name, 'str'), 'storageAccountName': _SERIALIZER.url('storage_account_name', storage_account_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_update_request(resource_group_name: str, account_name: str, storage_account_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2019-11-01-preview'] = kwargs.pop('api_version', _params.pop('api-version', '2019-11-01-preview'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataLakeAnalytics/accounts/{accountName}/storageAccounts/{storageAccountName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'accountName': _SERIALIZER.url('account_name', account_name, 'str'), 'storageAccountName': _SERIALIZER.url('storage_account_name', storage_account_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PATCH', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_request(resource_group_name: str, account_name: str, storage_account_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2019-11-01-preview'] = kwargs.pop('api_version', _params.pop('api-version', '2019-11-01-preview'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataLakeAnalytics/accounts/{accountName}/storageAccounts/{storageAccountName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'accountName': _SERIALIZER.url('account_name', account_name, 'str'), 'storageAccountName': _SERIALIZER.url('storage_account_name', storage_account_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_storage_containers_request(resource_group_name: str, account_name: str, storage_account_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2019-11-01-preview'] = kwargs.pop('api_version', _params.pop('api-version', '2019-11-01-preview'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataLakeAnalytics/accounts/{accountName}/storageAccounts/{storageAccountName}/containers')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'accountName': _SERIALIZER.url('account_name', account_name, 'str'), 'storageAccountName': _SERIALIZER.url('storage_account_name', storage_account_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_storage_container_request(resource_group_name: str, account_name: str, storage_account_name: str, container_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2019-11-01-preview'] = kwargs.pop('api_version', _params.pop('api-version', '2019-11-01-preview'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataLakeAnalytics/accounts/{accountName}/storageAccounts/{storageAccountName}/containers/{containerName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'accountName': _SERIALIZER.url('account_name', account_name, 'str'), 'storageAccountName': _SERIALIZER.url('storage_account_name', storage_account_name, 'str'), 'containerName': _SERIALIZER.url('container_name', container_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_sas_tokens_request(resource_group_name: str, account_name: str, storage_account_name: str, container_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2019-11-01-preview'] = kwargs.pop('api_version', _params.pop('api-version', '2019-11-01-preview'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataLakeAnalytics/accounts/{accountName}/storageAccounts/{storageAccountName}/containers/{containerName}/listSasTokens')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'accountName': _SERIALIZER.url('account_name', account_name, 'str'), 'storageAccountName': _SERIALIZER.url('storage_account_name', storage_account_name, 'str'), 'containerName': _SERIALIZER.url('container_name', container_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

class StorageAccountsOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.datalake.analytics.account.DataLakeAnalyticsAccountManagementClient`'s
        :attr:`storage_accounts` attribute.
    """
    models = _models

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        input_args = list(args)
        self._client = input_args.pop(0) if input_args else kwargs.pop('client')
        self._config = input_args.pop(0) if input_args else kwargs.pop('config')
        self._serialize = input_args.pop(0) if input_args else kwargs.pop('serializer')
        self._deserialize = input_args.pop(0) if input_args else kwargs.pop('deserializer')

    @distributed_trace
    def list_by_account(self, resource_group_name: str, account_name: str, filter: Optional[str]=None, top: Optional[int]=None, skip: Optional[int]=None, select: Optional[str]=None, orderby: Optional[str]=None, count: Optional[bool]=None, **kwargs: Any) -> Iterable['_models.StorageAccountInformation']:
        if False:
            return 10
        'Gets the first page of Azure Storage accounts, if any, linked to the specified Data Lake\n        Analytics account. The response includes a link to the next page, if any.\n\n        :param resource_group_name: The name of the Azure resource group. Required.\n        :type resource_group_name: str\n        :param account_name: The name of the Data Lake Analytics account. Required.\n        :type account_name: str\n        :param filter: The OData filter. Optional. Default value is None.\n        :type filter: str\n        :param top: The number of items to return. Optional. Default value is None.\n        :type top: int\n        :param skip: The number of items to skip over before returning elements. Optional. Default\n         value is None.\n        :type skip: int\n        :param select: OData Select statement. Limits the properties on each entry to just those\n         requested, e.g. Categories?$select=CategoryName,Description. Optional. Default value is None.\n        :type select: str\n        :param orderby: OrderBy clause. One or more comma-separated expressions with an optional "asc"\n         (the default) or "desc" depending on the order you\'d like the values sorted, e.g.\n         Categories?$orderby=CategoryName desc. Optional. Default value is None.\n        :type orderby: str\n        :param count: The Boolean value of true or false to request a count of the matching resources\n         included with the resources in the response, e.g. Categories?$count=true. Optional. Default\n         value is None.\n        :type count: bool\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either StorageAccountInformation or the result of\n         cls(response)\n        :rtype:\n         ~azure.core.paging.ItemPaged[~azure.mgmt.datalake.analytics.account.models.StorageAccountInformation]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2019-11-01-preview'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.StorageAccountInformationListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                print('Hello World!')
            if not next_link:
                request = build_list_by_account_request(resource_group_name=resource_group_name, account_name=account_name, subscription_id=self._config.subscription_id, filter=filter, top=top, skip=skip, select=select, orderby=orderby, count=count, api_version=api_version, template_url=self.list_by_account.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('StorageAccountInformationListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                return 10
            request = prepare_request(next_link)
            pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_by_account.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataLakeAnalytics/accounts/{accountName}/storageAccounts'}

    @overload
    def add(self, resource_group_name: str, account_name: str, storage_account_name: str, parameters: _models.AddStorageAccountParameters, *, content_type: str='application/json', **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Updates the specified Data Lake Analytics account to add an Azure Storage account.\n\n        :param resource_group_name: The name of the Azure resource group. Required.\n        :type resource_group_name: str\n        :param account_name: The name of the Data Lake Analytics account. Required.\n        :type account_name: str\n        :param storage_account_name: The name of the Azure Storage account to add. Required.\n        :type storage_account_name: str\n        :param parameters: The parameters containing the access key and optional suffix for the Azure\n         Storage Account. Required.\n        :type parameters: ~azure.mgmt.datalake.analytics.account.models.AddStorageAccountParameters\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def add(self, resource_group_name: str, account_name: str, storage_account_name: str, parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Updates the specified Data Lake Analytics account to add an Azure Storage account.\n\n        :param resource_group_name: The name of the Azure resource group. Required.\n        :type resource_group_name: str\n        :param account_name: The name of the Data Lake Analytics account. Required.\n        :type account_name: str\n        :param storage_account_name: The name of the Azure Storage account to add. Required.\n        :type storage_account_name: str\n        :param parameters: The parameters containing the access key and optional suffix for the Azure\n         Storage Account. Required.\n        :type parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def add(self, resource_group_name: str, account_name: str, storage_account_name: str, parameters: Union[_models.AddStorageAccountParameters, IO], **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        "Updates the specified Data Lake Analytics account to add an Azure Storage account.\n\n        :param resource_group_name: The name of the Azure resource group. Required.\n        :type resource_group_name: str\n        :param account_name: The name of the Data Lake Analytics account. Required.\n        :type account_name: str\n        :param storage_account_name: The name of the Azure Storage account to add. Required.\n        :type storage_account_name: str\n        :param parameters: The parameters containing the access key and optional suffix for the Azure\n         Storage Account. Is either a model type or a IO type. Required.\n        :type parameters: ~azure.mgmt.datalake.analytics.account.models.AddStorageAccountParameters or\n         IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2019-11-01-preview'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[None] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(parameters, (IO, bytes)):
            _content = parameters
        else:
            _json = self._serialize.body(parameters, 'AddStorageAccountParameters')
        request = build_add_request(resource_group_name=resource_group_name, account_name=account_name, storage_account_name=storage_account_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.add.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    add.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataLakeAnalytics/accounts/{accountName}/storageAccounts/{storageAccountName}'}

    @distributed_trace
    def get(self, resource_group_name: str, account_name: str, storage_account_name: str, **kwargs: Any) -> _models.StorageAccountInformation:
        if False:
            for i in range(10):
                print('nop')
        'Gets the specified Azure Storage account linked to the given Data Lake Analytics account.\n\n        :param resource_group_name: The name of the Azure resource group. Required.\n        :type resource_group_name: str\n        :param account_name: The name of the Data Lake Analytics account. Required.\n        :type account_name: str\n        :param storage_account_name: The name of the Azure Storage account for which to retrieve the\n         details. Required.\n        :type storage_account_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: StorageAccountInformation or the result of cls(response)\n        :rtype: ~azure.mgmt.datalake.analytics.account.models.StorageAccountInformation\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2019-11-01-preview'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.StorageAccountInformation] = kwargs.pop('cls', None)
        request = build_get_request(resource_group_name=resource_group_name, account_name=account_name, storage_account_name=storage_account_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('StorageAccountInformation', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataLakeAnalytics/accounts/{accountName}/storageAccounts/{storageAccountName}'}

    @overload
    def update(self, resource_group_name: str, account_name: str, storage_account_name: str, parameters: Optional[_models.UpdateStorageAccountParameters]=None, *, content_type: str='application/json', **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Updates the Data Lake Analytics account to replace Azure Storage blob account details, such as\n        the access key and/or suffix.\n\n        :param resource_group_name: The name of the Azure resource group. Required.\n        :type resource_group_name: str\n        :param account_name: The name of the Data Lake Analytics account. Required.\n        :type account_name: str\n        :param storage_account_name: The Azure Storage account to modify. Required.\n        :type storage_account_name: str\n        :param parameters: The parameters containing the access key and suffix to update the storage\n         account with, if any. Passing nothing results in no change. Default value is None.\n        :type parameters: ~azure.mgmt.datalake.analytics.account.models.UpdateStorageAccountParameters\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def update(self, resource_group_name: str, account_name: str, storage_account_name: str, parameters: Optional[IO]=None, *, content_type: str='application/json', **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Updates the Data Lake Analytics account to replace Azure Storage blob account details, such as\n        the access key and/or suffix.\n\n        :param resource_group_name: The name of the Azure resource group. Required.\n        :type resource_group_name: str\n        :param account_name: The name of the Data Lake Analytics account. Required.\n        :type account_name: str\n        :param storage_account_name: The Azure Storage account to modify. Required.\n        :type storage_account_name: str\n        :param parameters: The parameters containing the access key and suffix to update the storage\n         account with, if any. Passing nothing results in no change. Default value is None.\n        :type parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def update(self, resource_group_name: str, account_name: str, storage_account_name: str, parameters: Optional[Union[_models.UpdateStorageAccountParameters, IO]]=None, **kwargs: Any) -> None:
        if False:
            return 10
        "Updates the Data Lake Analytics account to replace Azure Storage blob account details, such as\n        the access key and/or suffix.\n\n        :param resource_group_name: The name of the Azure resource group. Required.\n        :type resource_group_name: str\n        :param account_name: The name of the Data Lake Analytics account. Required.\n        :type account_name: str\n        :param storage_account_name: The Azure Storage account to modify. Required.\n        :type storage_account_name: str\n        :param parameters: The parameters containing the access key and suffix to update the storage\n         account with, if any. Passing nothing results in no change. Is either a model type or a IO\n         type. Default value is None.\n        :type parameters: ~azure.mgmt.datalake.analytics.account.models.UpdateStorageAccountParameters\n         or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2019-11-01-preview'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[None] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(parameters, (IO, bytes)):
            _content = parameters
        elif parameters is not None:
            _json = self._serialize.body(parameters, 'UpdateStorageAccountParameters')
        else:
            _json = None
        request = build_update_request(resource_group_name=resource_group_name, account_name=account_name, storage_account_name=storage_account_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.update.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    update.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataLakeAnalytics/accounts/{accountName}/storageAccounts/{storageAccountName}'}

    @distributed_trace
    def delete(self, resource_group_name: str, account_name: str, storage_account_name: str, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Updates the specified Data Lake Analytics account to remove an Azure Storage account.\n\n        :param resource_group_name: The name of the Azure resource group. Required.\n        :type resource_group_name: str\n        :param account_name: The name of the Data Lake Analytics account. Required.\n        :type account_name: str\n        :param storage_account_name: The name of the Azure Storage account to remove. Required.\n        :type storage_account_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2019-11-01-preview'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_delete_request(resource_group_name=resource_group_name, account_name=account_name, storage_account_name=storage_account_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.delete.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    delete.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataLakeAnalytics/accounts/{accountName}/storageAccounts/{storageAccountName}'}

    @distributed_trace
    def list_storage_containers(self, resource_group_name: str, account_name: str, storage_account_name: str, **kwargs: Any) -> Iterable['_models.StorageContainer']:
        if False:
            for i in range(10):
                print('nop')
        'Lists the Azure Storage containers, if any, associated with the specified Data Lake Analytics\n        and Azure Storage account combination. The response includes a link to the next page of\n        results, if any.\n\n        :param resource_group_name: The name of the Azure resource group. Required.\n        :type resource_group_name: str\n        :param account_name: The name of the Data Lake Analytics account. Required.\n        :type account_name: str\n        :param storage_account_name: The name of the Azure storage account from which to list blob\n         containers. Required.\n        :type storage_account_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either StorageContainer or the result of cls(response)\n        :rtype:\n         ~azure.core.paging.ItemPaged[~azure.mgmt.datalake.analytics.account.models.StorageContainer]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2019-11-01-preview'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.StorageContainerListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                return 10
            if not next_link:
                request = build_list_storage_containers_request(resource_group_name=resource_group_name, account_name=account_name, storage_account_name=storage_account_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.list_storage_containers.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('StorageContainerListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                while True:
                    i = 10
            request = prepare_request(next_link)
            pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_storage_containers.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataLakeAnalytics/accounts/{accountName}/storageAccounts/{storageAccountName}/containers'}

    @distributed_trace
    def get_storage_container(self, resource_group_name: str, account_name: str, storage_account_name: str, container_name: str, **kwargs: Any) -> _models.StorageContainer:
        if False:
            return 10
        'Gets the specified Azure Storage container associated with the given Data Lake Analytics and\n        Azure Storage accounts.\n\n        :param resource_group_name: The name of the Azure resource group. Required.\n        :type resource_group_name: str\n        :param account_name: The name of the Data Lake Analytics account. Required.\n        :type account_name: str\n        :param storage_account_name: The name of the Azure storage account from which to retrieve the\n         blob container. Required.\n        :type storage_account_name: str\n        :param container_name: The name of the Azure storage container to retrieve. Required.\n        :type container_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: StorageContainer or the result of cls(response)\n        :rtype: ~azure.mgmt.datalake.analytics.account.models.StorageContainer\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2019-11-01-preview'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.StorageContainer] = kwargs.pop('cls', None)
        request = build_get_storage_container_request(resource_group_name=resource_group_name, account_name=account_name, storage_account_name=storage_account_name, container_name=container_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get_storage_container.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('StorageContainer', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get_storage_container.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataLakeAnalytics/accounts/{accountName}/storageAccounts/{storageAccountName}/containers/{containerName}'}

    @distributed_trace
    def list_sas_tokens(self, resource_group_name: str, account_name: str, storage_account_name: str, container_name: str, **kwargs: Any) -> Iterable['_models.SasTokenInformation']:
        if False:
            print('Hello World!')
        'Gets the SAS token associated with the specified Data Lake Analytics and Azure Storage account\n        and container combination.\n\n        :param resource_group_name: The name of the Azure resource group. Required.\n        :type resource_group_name: str\n        :param account_name: The name of the Data Lake Analytics account. Required.\n        :type account_name: str\n        :param storage_account_name: The name of the Azure storage account for which the SAS token is\n         being requested. Required.\n        :type storage_account_name: str\n        :param container_name: The name of the Azure storage container for which the SAS token is being\n         requested. Required.\n        :type container_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either SasTokenInformation or the result of cls(response)\n        :rtype:\n         ~azure.core.paging.ItemPaged[~azure.mgmt.datalake.analytics.account.models.SasTokenInformation]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2019-11-01-preview'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.SasTokenInformationListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                while True:
                    i = 10
            if not next_link:
                request = build_list_sas_tokens_request(resource_group_name=resource_group_name, account_name=account_name, storage_account_name=storage_account_name, container_name=container_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.list_sas_tokens.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('SasTokenInformationListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                print('Hello World!')
            request = prepare_request(next_link)
            pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_sas_tokens.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataLakeAnalytics/accounts/{accountName}/storageAccounts/{storageAccountName}/containers/{containerName}/listSasTokens'}