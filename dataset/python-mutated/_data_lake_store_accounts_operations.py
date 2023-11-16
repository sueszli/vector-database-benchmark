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
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2019-11-01-preview'] = kwargs.pop('api_version', _params.pop('api-version', '2019-11-01-preview'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataLakeAnalytics/accounts/{accountName}/dataLakeStoreAccounts')
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

def build_add_request(resource_group_name: str, account_name: str, data_lake_store_account_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2019-11-01-preview'] = kwargs.pop('api_version', _params.pop('api-version', '2019-11-01-preview'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataLakeAnalytics/accounts/{accountName}/dataLakeStoreAccounts/{dataLakeStoreAccountName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'accountName': _SERIALIZER.url('account_name', account_name, 'str'), 'dataLakeStoreAccountName': _SERIALIZER.url('data_lake_store_account_name', data_lake_store_account_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_request(resource_group_name: str, account_name: str, data_lake_store_account_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2019-11-01-preview'] = kwargs.pop('api_version', _params.pop('api-version', '2019-11-01-preview'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataLakeAnalytics/accounts/{accountName}/dataLakeStoreAccounts/{dataLakeStoreAccountName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'accountName': _SERIALIZER.url('account_name', account_name, 'str'), 'dataLakeStoreAccountName': _SERIALIZER.url('data_lake_store_account_name', data_lake_store_account_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_request(resource_group_name: str, account_name: str, data_lake_store_account_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2019-11-01-preview'] = kwargs.pop('api_version', _params.pop('api-version', '2019-11-01-preview'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataLakeAnalytics/accounts/{accountName}/dataLakeStoreAccounts/{dataLakeStoreAccountName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'accountName': _SERIALIZER.url('account_name', account_name, 'str'), 'dataLakeStoreAccountName': _SERIALIZER.url('data_lake_store_account_name', data_lake_store_account_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, headers=_headers, **kwargs)

class DataLakeStoreAccountsOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.datalake.analytics.account.DataLakeAnalyticsAccountManagementClient`'s
        :attr:`data_lake_store_accounts` attribute.
    """
    models = _models

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        input_args = list(args)
        self._client = input_args.pop(0) if input_args else kwargs.pop('client')
        self._config = input_args.pop(0) if input_args else kwargs.pop('config')
        self._serialize = input_args.pop(0) if input_args else kwargs.pop('serializer')
        self._deserialize = input_args.pop(0) if input_args else kwargs.pop('deserializer')

    @distributed_trace
    def list_by_account(self, resource_group_name: str, account_name: str, filter: Optional[str]=None, top: Optional[int]=None, skip: Optional[int]=None, select: Optional[str]=None, orderby: Optional[str]=None, count: Optional[bool]=None, **kwargs: Any) -> Iterable['_models.DataLakeStoreAccountInformation']:
        if False:
            for i in range(10):
                print('nop')
        'Gets the first page of Data Lake Store accounts linked to the specified Data Lake Analytics\n        account. The response includes a link to the next page, if any.\n\n        :param resource_group_name: The name of the Azure resource group. Required.\n        :type resource_group_name: str\n        :param account_name: The name of the Data Lake Analytics account. Required.\n        :type account_name: str\n        :param filter: OData filter. Optional. Default value is None.\n        :type filter: str\n        :param top: The number of items to return. Optional. Default value is None.\n        :type top: int\n        :param skip: The number of items to skip over before returning elements. Optional. Default\n         value is None.\n        :type skip: int\n        :param select: OData Select statement. Limits the properties on each entry to just those\n         requested, e.g. Categories?$select=CategoryName,Description. Optional. Default value is None.\n        :type select: str\n        :param orderby: OrderBy clause. One or more comma-separated expressions with an optional "asc"\n         (the default) or "desc" depending on the order you\'d like the values sorted, e.g.\n         Categories?$orderby=CategoryName desc. Optional. Default value is None.\n        :type orderby: str\n        :param count: The Boolean value of true or false to request a count of the matching resources\n         included with the resources in the response, e.g. Categories?$count=true. Optional. Default\n         value is None.\n        :type count: bool\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either DataLakeStoreAccountInformation or the result of\n         cls(response)\n        :rtype:\n         ~azure.core.paging.ItemPaged[~azure.mgmt.datalake.analytics.account.models.DataLakeStoreAccountInformation]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2019-11-01-preview'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.DataLakeStoreAccountInformationListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                i = 10
                return i + 15
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
                return 10
            deserialized = self._deserialize('DataLakeStoreAccountInformationListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                for i in range(10):
                    print('nop')
            request = prepare_request(next_link)
            pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_by_account.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataLakeAnalytics/accounts/{accountName}/dataLakeStoreAccounts'}

    @overload
    def add(self, resource_group_name: str, account_name: str, data_lake_store_account_name: str, parameters: Optional[_models.AddDataLakeStoreParameters]=None, *, content_type: str='application/json', **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Updates the specified Data Lake Analytics account to include the additional Data Lake Store\n        account.\n\n        :param resource_group_name: The name of the Azure resource group. Required.\n        :type resource_group_name: str\n        :param account_name: The name of the Data Lake Analytics account. Required.\n        :type account_name: str\n        :param data_lake_store_account_name: The name of the Data Lake Store account to add. Required.\n        :type data_lake_store_account_name: str\n        :param parameters: The details of the Data Lake Store account. Default value is None.\n        :type parameters: ~azure.mgmt.datalake.analytics.account.models.AddDataLakeStoreParameters\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def add(self, resource_group_name: str, account_name: str, data_lake_store_account_name: str, parameters: Optional[IO]=None, *, content_type: str='application/json', **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        'Updates the specified Data Lake Analytics account to include the additional Data Lake Store\n        account.\n\n        :param resource_group_name: The name of the Azure resource group. Required.\n        :type resource_group_name: str\n        :param account_name: The name of the Data Lake Analytics account. Required.\n        :type account_name: str\n        :param data_lake_store_account_name: The name of the Data Lake Store account to add. Required.\n        :type data_lake_store_account_name: str\n        :param parameters: The details of the Data Lake Store account. Default value is None.\n        :type parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def add(self, resource_group_name: str, account_name: str, data_lake_store_account_name: str, parameters: Optional[Union[_models.AddDataLakeStoreParameters, IO]]=None, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Updates the specified Data Lake Analytics account to include the additional Data Lake Store\n        account.\n\n        :param resource_group_name: The name of the Azure resource group. Required.\n        :type resource_group_name: str\n        :param account_name: The name of the Data Lake Analytics account. Required.\n        :type account_name: str\n        :param data_lake_store_account_name: The name of the Data Lake Store account to add. Required.\n        :type data_lake_store_account_name: str\n        :param parameters: The details of the Data Lake Store account. Is either a model type or a IO\n         type. Default value is None.\n        :type parameters: ~azure.mgmt.datalake.analytics.account.models.AddDataLakeStoreParameters or\n         IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
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
            _json = self._serialize.body(parameters, 'AddDataLakeStoreParameters')
        else:
            _json = None
        request = build_add_request(resource_group_name=resource_group_name, account_name=account_name, data_lake_store_account_name=data_lake_store_account_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.add.metadata['url'], headers=_headers, params=_params)
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
    add.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataLakeAnalytics/accounts/{accountName}/dataLakeStoreAccounts/{dataLakeStoreAccountName}'}

    @distributed_trace
    def get(self, resource_group_name: str, account_name: str, data_lake_store_account_name: str, **kwargs: Any) -> _models.DataLakeStoreAccountInformation:
        if False:
            for i in range(10):
                print('nop')
        'Gets the specified Data Lake Store account details in the specified Data Lake Analytics\n        account.\n\n        :param resource_group_name: The name of the Azure resource group. Required.\n        :type resource_group_name: str\n        :param account_name: The name of the Data Lake Analytics account. Required.\n        :type account_name: str\n        :param data_lake_store_account_name: The name of the Data Lake Store account to retrieve.\n         Required.\n        :type data_lake_store_account_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: DataLakeStoreAccountInformation or the result of cls(response)\n        :rtype: ~azure.mgmt.datalake.analytics.account.models.DataLakeStoreAccountInformation\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2019-11-01-preview'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.DataLakeStoreAccountInformation] = kwargs.pop('cls', None)
        request = build_get_request(resource_group_name=resource_group_name, account_name=account_name, data_lake_store_account_name=data_lake_store_account_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('DataLakeStoreAccountInformation', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataLakeAnalytics/accounts/{accountName}/dataLakeStoreAccounts/{dataLakeStoreAccountName}'}

    @distributed_trace
    def delete(self, resource_group_name: str, account_name: str, data_lake_store_account_name: str, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        'Updates the Data Lake Analytics account specified to remove the specified Data Lake Store\n        account.\n\n        :param resource_group_name: The name of the Azure resource group. Required.\n        :type resource_group_name: str\n        :param account_name: The name of the Data Lake Analytics account. Required.\n        :type account_name: str\n        :param data_lake_store_account_name: The name of the Data Lake Store account to remove.\n         Required.\n        :type data_lake_store_account_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2019-11-01-preview'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        request = build_delete_request(resource_group_name=resource_group_name, account_name=account_name, data_lake_store_account_name=data_lake_store_account_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.delete.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    delete.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataLakeAnalytics/accounts/{accountName}/dataLakeStoreAccounts/{dataLakeStoreAccountName}'}