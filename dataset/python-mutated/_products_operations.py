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

def build_list_request(resource_group: str, registration_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2022-06-01'] = kwargs.pop('api_version', _params.pop('api-version', '2022-06-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroup}/providers/Microsoft.AzureStack/registrations/{registrationName}/products')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroup': _SERIALIZER.url('resource_group', resource_group, 'str'), 'registrationName': _SERIALIZER.url('registration_name', registration_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_request(resource_group: str, registration_name: str, product_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2022-06-01'] = kwargs.pop('api_version', _params.pop('api-version', '2022-06-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroup}/providers/Microsoft.AzureStack/registrations/{registrationName}/products/{productName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroup': _SERIALIZER.url('resource_group', resource_group, 'str'), 'registrationName': _SERIALIZER.url('registration_name', registration_name, 'str'), 'productName': _SERIALIZER.url('product_name', product_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_details_request(resource_group: str, registration_name: str, product_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2022-06-01'] = kwargs.pop('api_version', _params.pop('api-version', '2022-06-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroup}/providers/Microsoft.AzureStack/registrations/{registrationName}/products/{productName}/listDetails')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroup': _SERIALIZER.url('resource_group', resource_group, 'str'), 'registrationName': _SERIALIZER.url('registration_name', registration_name, 'str'), 'productName': _SERIALIZER.url('product_name', product_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_products_request(resource_group: str, registration_name: str, product_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2022-06-01'] = kwargs.pop('api_version', _params.pop('api-version', '2022-06-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroup}/providers/Microsoft.AzureStack/registrations/{registrationName}/products/{productName}/listProducts')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroup': _SERIALIZER.url('resource_group', resource_group, 'str'), 'registrationName': _SERIALIZER.url('registration_name', registration_name, 'str'), 'productName': _SERIALIZER.url('product_name', product_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_products_request(resource_group: str, registration_name: str, product_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2022-06-01'] = kwargs.pop('api_version', _params.pop('api-version', '2022-06-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroup}/providers/Microsoft.AzureStack/registrations/{registrationName}/products/{productName}/getProducts')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroup': _SERIALIZER.url('resource_group', resource_group, 'str'), 'registrationName': _SERIALIZER.url('registration_name', registration_name, 'str'), 'productName': _SERIALIZER.url('product_name', product_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_product_request(resource_group: str, registration_name: str, product_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2022-06-01'] = kwargs.pop('api_version', _params.pop('api-version', '2022-06-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroup}/providers/Microsoft.AzureStack/registrations/{registrationName}/products/{productName}/getProduct')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroup': _SERIALIZER.url('resource_group', resource_group, 'str'), 'registrationName': _SERIALIZER.url('registration_name', registration_name, 'str'), 'productName': _SERIALIZER.url('product_name', product_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

def build_upload_log_request(resource_group: str, registration_name: str, product_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2022-06-01'] = kwargs.pop('api_version', _params.pop('api-version', '2022-06-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroup}/providers/Microsoft.AzureStack/registrations/{registrationName}/products/{productName}/uploadProductLog')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroup': _SERIALIZER.url('resource_group', resource_group, 'str'), 'registrationName': _SERIALIZER.url('registration_name', registration_name, 'str'), 'productName': _SERIALIZER.url('product_name', product_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

class ProductsOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.azurestack.AzureStackManagementClient`'s
        :attr:`products` attribute.
    """
    models = _models

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        input_args = list(args)
        self._client = input_args.pop(0) if input_args else kwargs.pop('client')
        self._config = input_args.pop(0) if input_args else kwargs.pop('config')
        self._serialize = input_args.pop(0) if input_args else kwargs.pop('serializer')
        self._deserialize = input_args.pop(0) if input_args else kwargs.pop('deserializer')

    @distributed_trace
    def list(self, resource_group: str, registration_name: str, **kwargs: Any) -> Iterable['_models.Product']:
        if False:
            return 10
        'Returns a list of products.\n\n        :param resource_group: Name of the resource group. Required.\n        :type resource_group: str\n        :param registration_name: Name of the Azure Stack registration. Required.\n        :type registration_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either Product or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.azurestack.models.Product]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2022-06-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.ProductList] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                i = 10
                return i + 15
            if not next_link:
                request = build_list_request(resource_group=resource_group, registration_name=registration_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.list.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('ProductList', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                i = 10
                return i + 15
            request = prepare_request(next_link)
            pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroup}/providers/Microsoft.AzureStack/registrations/{registrationName}/products'}

    @distributed_trace
    def get(self, resource_group: str, registration_name: str, product_name: str, **kwargs: Any) -> _models.Product:
        if False:
            print('Hello World!')
        'Returns the specified product.\n\n        :param resource_group: Name of the resource group. Required.\n        :type resource_group: str\n        :param registration_name: Name of the Azure Stack registration. Required.\n        :type registration_name: str\n        :param product_name: Name of the product. Required.\n        :type product_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Product or the result of cls(response)\n        :rtype: ~azure.mgmt.azurestack.models.Product\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2022-06-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.Product] = kwargs.pop('cls', None)
        request = build_get_request(resource_group=resource_group, registration_name=registration_name, product_name=product_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('Product', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroup}/providers/Microsoft.AzureStack/registrations/{registrationName}/products/{productName}'}

    @distributed_trace
    def list_details(self, resource_group: str, registration_name: str, product_name: str, **kwargs: Any) -> _models.ExtendedProduct:
        if False:
            return 10
        'Returns the extended properties of a product.\n\n        :param resource_group: Name of the resource group. Required.\n        :type resource_group: str\n        :param registration_name: Name of the Azure Stack registration. Required.\n        :type registration_name: str\n        :param product_name: Name of the product. Required.\n        :type product_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ExtendedProduct or the result of cls(response)\n        :rtype: ~azure.mgmt.azurestack.models.ExtendedProduct\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2022-06-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.ExtendedProduct] = kwargs.pop('cls', None)
        request = build_list_details_request(resource_group=resource_group, registration_name=registration_name, product_name=product_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.list_details.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('ExtendedProduct', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    list_details.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroup}/providers/Microsoft.AzureStack/registrations/{registrationName}/products/{productName}/listDetails'}

    @overload
    def list_products(self, resource_group: str, registration_name: str, product_name: str, device_configuration: Optional[_models.DeviceConfiguration]=None, *, content_type: str='application/json', **kwargs: Any) -> _models.ProductList:
        if False:
            for i in range(10):
                print('nop')
        'Returns a list of products.\n\n        :param resource_group: Name of the resource group. Required.\n        :type resource_group: str\n        :param registration_name: Name of the Azure Stack registration. Required.\n        :type registration_name: str\n        :param product_name: Name of the product. Required.\n        :type product_name: str\n        :param device_configuration: Device configuration. Default value is None.\n        :type device_configuration: ~azure.mgmt.azurestack.models.DeviceConfiguration\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ProductList or the result of cls(response)\n        :rtype: ~azure.mgmt.azurestack.models.ProductList\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def list_products(self, resource_group: str, registration_name: str, product_name: str, device_configuration: Optional[IO]=None, *, content_type: str='application/json', **kwargs: Any) -> _models.ProductList:
        if False:
            print('Hello World!')
        'Returns a list of products.\n\n        :param resource_group: Name of the resource group. Required.\n        :type resource_group: str\n        :param registration_name: Name of the Azure Stack registration. Required.\n        :type registration_name: str\n        :param product_name: Name of the product. Required.\n        :type product_name: str\n        :param device_configuration: Device configuration. Default value is None.\n        :type device_configuration: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ProductList or the result of cls(response)\n        :rtype: ~azure.mgmt.azurestack.models.ProductList\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def list_products(self, resource_group: str, registration_name: str, product_name: str, device_configuration: Optional[Union[_models.DeviceConfiguration, IO]]=None, **kwargs: Any) -> _models.ProductList:
        if False:
            while True:
                i = 10
        "Returns a list of products.\n\n        :param resource_group: Name of the resource group. Required.\n        :type resource_group: str\n        :param registration_name: Name of the Azure Stack registration. Required.\n        :type registration_name: str\n        :param product_name: Name of the product. Required.\n        :type product_name: str\n        :param device_configuration: Device configuration. Is either a model type or a IO type. Default\n         value is None.\n        :type device_configuration: ~azure.mgmt.azurestack.models.DeviceConfiguration or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ProductList or the result of cls(response)\n        :rtype: ~azure.mgmt.azurestack.models.ProductList\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2022-06-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.ProductList] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(device_configuration, (IO, bytes)):
            _content = device_configuration
        elif device_configuration is not None:
            _json = self._serialize.body(device_configuration, 'DeviceConfiguration')
        else:
            _json = None
        request = build_list_products_request(resource_group=resource_group, registration_name=registration_name, product_name=product_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.list_products.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('ProductList', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    list_products.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroup}/providers/Microsoft.AzureStack/registrations/{registrationName}/products/{productName}/listProducts'}

    @overload
    def get_products(self, resource_group: str, registration_name: str, product_name: str, device_configuration: Optional[_models.DeviceConfiguration]=None, *, content_type: str='application/json', **kwargs: Any) -> _models.ProductList:
        if False:
            print('Hello World!')
        'Returns a list of products.\n\n        :param resource_group: Name of the resource group. Required.\n        :type resource_group: str\n        :param registration_name: Name of the Azure Stack registration. Required.\n        :type registration_name: str\n        :param product_name: Name of the product. Required.\n        :type product_name: str\n        :param device_configuration: Device configuration. Default value is None.\n        :type device_configuration: ~azure.mgmt.azurestack.models.DeviceConfiguration\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ProductList or the result of cls(response)\n        :rtype: ~azure.mgmt.azurestack.models.ProductList\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def get_products(self, resource_group: str, registration_name: str, product_name: str, device_configuration: Optional[IO]=None, *, content_type: str='application/json', **kwargs: Any) -> _models.ProductList:
        if False:
            i = 10
            return i + 15
        'Returns a list of products.\n\n        :param resource_group: Name of the resource group. Required.\n        :type resource_group: str\n        :param registration_name: Name of the Azure Stack registration. Required.\n        :type registration_name: str\n        :param product_name: Name of the product. Required.\n        :type product_name: str\n        :param device_configuration: Device configuration. Default value is None.\n        :type device_configuration: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ProductList or the result of cls(response)\n        :rtype: ~azure.mgmt.azurestack.models.ProductList\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def get_products(self, resource_group: str, registration_name: str, product_name: str, device_configuration: Optional[Union[_models.DeviceConfiguration, IO]]=None, **kwargs: Any) -> _models.ProductList:
        if False:
            return 10
        "Returns a list of products.\n\n        :param resource_group: Name of the resource group. Required.\n        :type resource_group: str\n        :param registration_name: Name of the Azure Stack registration. Required.\n        :type registration_name: str\n        :param product_name: Name of the product. Required.\n        :type product_name: str\n        :param device_configuration: Device configuration. Is either a model type or a IO type. Default\n         value is None.\n        :type device_configuration: ~azure.mgmt.azurestack.models.DeviceConfiguration or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ProductList or the result of cls(response)\n        :rtype: ~azure.mgmt.azurestack.models.ProductList\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2022-06-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.ProductList] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(device_configuration, (IO, bytes)):
            _content = device_configuration
        elif device_configuration is not None:
            _json = self._serialize.body(device_configuration, 'DeviceConfiguration')
        else:
            _json = None
        request = build_get_products_request(resource_group=resource_group, registration_name=registration_name, product_name=product_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.get_products.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('ProductList', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get_products.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroup}/providers/Microsoft.AzureStack/registrations/{registrationName}/products/{productName}/getProducts'}

    @overload
    def get_product(self, resource_group: str, registration_name: str, product_name: str, device_configuration: Optional[_models.DeviceConfiguration]=None, *, content_type: str='application/json', **kwargs: Any) -> _models.Product:
        if False:
            for i in range(10):
                print('nop')
        'Returns the specified product.\n\n        :param resource_group: Name of the resource group. Required.\n        :type resource_group: str\n        :param registration_name: Name of the Azure Stack registration. Required.\n        :type registration_name: str\n        :param product_name: Name of the product. Required.\n        :type product_name: str\n        :param device_configuration: Device configuration. Default value is None.\n        :type device_configuration: ~azure.mgmt.azurestack.models.DeviceConfiguration\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Product or the result of cls(response)\n        :rtype: ~azure.mgmt.azurestack.models.Product\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def get_product(self, resource_group: str, registration_name: str, product_name: str, device_configuration: Optional[IO]=None, *, content_type: str='application/json', **kwargs: Any) -> _models.Product:
        if False:
            i = 10
            return i + 15
        'Returns the specified product.\n\n        :param resource_group: Name of the resource group. Required.\n        :type resource_group: str\n        :param registration_name: Name of the Azure Stack registration. Required.\n        :type registration_name: str\n        :param product_name: Name of the product. Required.\n        :type product_name: str\n        :param device_configuration: Device configuration. Default value is None.\n        :type device_configuration: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Product or the result of cls(response)\n        :rtype: ~azure.mgmt.azurestack.models.Product\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def get_product(self, resource_group: str, registration_name: str, product_name: str, device_configuration: Optional[Union[_models.DeviceConfiguration, IO]]=None, **kwargs: Any) -> _models.Product:
        if False:
            return 10
        "Returns the specified product.\n\n        :param resource_group: Name of the resource group. Required.\n        :type resource_group: str\n        :param registration_name: Name of the Azure Stack registration. Required.\n        :type registration_name: str\n        :param product_name: Name of the product. Required.\n        :type product_name: str\n        :param device_configuration: Device configuration. Is either a model type or a IO type. Default\n         value is None.\n        :type device_configuration: ~azure.mgmt.azurestack.models.DeviceConfiguration or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Product or the result of cls(response)\n        :rtype: ~azure.mgmt.azurestack.models.Product\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2022-06-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.Product] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(device_configuration, (IO, bytes)):
            _content = device_configuration
        elif device_configuration is not None:
            _json = self._serialize.body(device_configuration, 'DeviceConfiguration')
        else:
            _json = None
        request = build_get_product_request(resource_group=resource_group, registration_name=registration_name, product_name=product_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.get_product.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('Product', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get_product.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroup}/providers/Microsoft.AzureStack/registrations/{registrationName}/products/{productName}/getProduct'}

    @overload
    def upload_log(self, resource_group: str, registration_name: str, product_name: str, marketplace_product_log_update: Optional[_models.MarketplaceProductLogUpdate]=None, *, content_type: str='application/json', **kwargs: Any) -> _models.ProductLog:
        if False:
            print('Hello World!')
        'Returns the specified product.\n\n        :param resource_group: Name of the resource group. Required.\n        :type resource_group: str\n        :param registration_name: Name of the Azure Stack registration. Required.\n        :type registration_name: str\n        :param product_name: Name of the product. Required.\n        :type product_name: str\n        :param marketplace_product_log_update: Update details for product log. Default value is None.\n        :type marketplace_product_log_update: ~azure.mgmt.azurestack.models.MarketplaceProductLogUpdate\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ProductLog or the result of cls(response)\n        :rtype: ~azure.mgmt.azurestack.models.ProductLog\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def upload_log(self, resource_group: str, registration_name: str, product_name: str, marketplace_product_log_update: Optional[IO]=None, *, content_type: str='application/json', **kwargs: Any) -> _models.ProductLog:
        if False:
            for i in range(10):
                print('nop')
        'Returns the specified product.\n\n        :param resource_group: Name of the resource group. Required.\n        :type resource_group: str\n        :param registration_name: Name of the Azure Stack registration. Required.\n        :type registration_name: str\n        :param product_name: Name of the product. Required.\n        :type product_name: str\n        :param marketplace_product_log_update: Update details for product log. Default value is None.\n        :type marketplace_product_log_update: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ProductLog or the result of cls(response)\n        :rtype: ~azure.mgmt.azurestack.models.ProductLog\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def upload_log(self, resource_group: str, registration_name: str, product_name: str, marketplace_product_log_update: Optional[Union[_models.MarketplaceProductLogUpdate, IO]]=None, **kwargs: Any) -> _models.ProductLog:
        if False:
            while True:
                i = 10
        "Returns the specified product.\n\n        :param resource_group: Name of the resource group. Required.\n        :type resource_group: str\n        :param registration_name: Name of the Azure Stack registration. Required.\n        :type registration_name: str\n        :param product_name: Name of the product. Required.\n        :type product_name: str\n        :param marketplace_product_log_update: Update details for product log. Is either a model type\n         or a IO type. Default value is None.\n        :type marketplace_product_log_update: ~azure.mgmt.azurestack.models.MarketplaceProductLogUpdate\n         or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ProductLog or the result of cls(response)\n        :rtype: ~azure.mgmt.azurestack.models.ProductLog\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2022-06-01'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.ProductLog] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(marketplace_product_log_update, (IO, bytes)):
            _content = marketplace_product_log_update
        elif marketplace_product_log_update is not None:
            _json = self._serialize.body(marketplace_product_log_update, 'MarketplaceProductLogUpdate')
        else:
            _json = None
        request = build_upload_log_request(resource_group=resource_group, registration_name=registration_name, product_name=product_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.upload_log.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('ProductLog', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    upload_log.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroup}/providers/Microsoft.AzureStack/registrations/{registrationName}/products/{productName}/uploadProductLog'}