import sys
from typing import Any, Callable, Dict, Iterable, Optional, TypeVar
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

def build_list_by_customer_request(billing_account_name: str, customer_name: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version = kwargs.pop('api_version', _params.pop('api-version', '2020-05-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/providers/Microsoft.Billing/billingAccounts/{billingAccountName}/customers/{customerName}/billingPermissions')
    path_format_arguments = {'billingAccountName': _SERIALIZER.url('billing_account_name', billing_account_name, 'str'), 'customerName': _SERIALIZER.url('customer_name', customer_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_by_billing_account_request(billing_account_name: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version = kwargs.pop('api_version', _params.pop('api-version', '2020-05-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/providers/Microsoft.Billing/billingAccounts/{billingAccountName}/billingPermissions')
    path_format_arguments = {'billingAccountName': _SERIALIZER.url('billing_account_name', billing_account_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_by_invoice_sections_request(billing_account_name: str, billing_profile_name: str, invoice_section_name: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version = kwargs.pop('api_version', _params.pop('api-version', '2020-05-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/providers/Microsoft.Billing/billingAccounts/{billingAccountName}/billingProfiles/{billingProfileName}/invoiceSections/{invoiceSectionName}/billingPermissions')
    path_format_arguments = {'billingAccountName': _SERIALIZER.url('billing_account_name', billing_account_name, 'str'), 'billingProfileName': _SERIALIZER.url('billing_profile_name', billing_profile_name, 'str'), 'invoiceSectionName': _SERIALIZER.url('invoice_section_name', invoice_section_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_by_billing_profile_request(billing_account_name: str, billing_profile_name: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version = kwargs.pop('api_version', _params.pop('api-version', '2020-05-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/providers/Microsoft.Billing/billingAccounts/{billingAccountName}/billingProfiles/{billingProfileName}/billingPermissions')
    path_format_arguments = {'billingAccountName': _SERIALIZER.url('billing_account_name', billing_account_name, 'str'), 'billingProfileName': _SERIALIZER.url('billing_profile_name', billing_profile_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

class BillingPermissionsOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.billing.BillingManagementClient`'s
        :attr:`billing_permissions` attribute.
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
    def list_by_customer(self, billing_account_name: str, customer_name: str, **kwargs: Any) -> Iterable['_models.BillingPermissionsProperties']:
        if False:
            for i in range(10):
                print('nop')
        'Lists the billing permissions the caller has for a customer.\n\n        :param billing_account_name: The ID that uniquely identifies a billing account. Required.\n        :type billing_account_name: str\n        :param customer_name: The ID that uniquely identifies a customer. Required.\n        :type customer_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either BillingPermissionsProperties or the result of\n         cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.billing.models.BillingPermissionsProperties]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', '2020-05-01'))
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                i = 10
                return i + 15
            if not next_link:
                request = build_list_by_customer_request(billing_account_name=billing_account_name, customer_name=customer_name, api_version=api_version, template_url=self.list_by_customer.metadata['url'], headers=_headers, params=_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
            else:
                request = HttpRequest('GET', next_link)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
                request.method = 'GET'
            return request

        def extract_data(pipeline_response):
            if False:
                for i in range(10):
                    print('nop')
            deserialized = self._deserialize('BillingPermissionsListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                while True:
                    i = 10
            request = prepare_request(next_link)
            pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_by_customer.metadata = {'url': '/providers/Microsoft.Billing/billingAccounts/{billingAccountName}/customers/{customerName}/billingPermissions'}

    @distributed_trace
    def list_by_billing_account(self, billing_account_name: str, **kwargs: Any) -> Iterable['_models.BillingPermissionsProperties']:
        if False:
            while True:
                i = 10
        'Lists the billing permissions the caller has on a billing account.\n\n        :param billing_account_name: The ID that uniquely identifies a billing account. Required.\n        :type billing_account_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either BillingPermissionsProperties or the result of\n         cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.billing.models.BillingPermissionsProperties]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', '2020-05-01'))
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                i = 10
                return i + 15
            if not next_link:
                request = build_list_by_billing_account_request(billing_account_name=billing_account_name, api_version=api_version, template_url=self.list_by_billing_account.metadata['url'], headers=_headers, params=_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
            else:
                request = HttpRequest('GET', next_link)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
                request.method = 'GET'
            return request

        def extract_data(pipeline_response):
            if False:
                print('Hello World!')
            deserialized = self._deserialize('BillingPermissionsListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                i = 10
                return i + 15
            request = prepare_request(next_link)
            pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_by_billing_account.metadata = {'url': '/providers/Microsoft.Billing/billingAccounts/{billingAccountName}/billingPermissions'}

    @distributed_trace
    def list_by_invoice_sections(self, billing_account_name: str, billing_profile_name: str, invoice_section_name: str, **kwargs: Any) -> Iterable['_models.BillingPermissionsProperties']:
        if False:
            while True:
                i = 10
        'Lists the billing permissions the caller has on an invoice section.\n\n        :param billing_account_name: The ID that uniquely identifies a billing account. Required.\n        :type billing_account_name: str\n        :param billing_profile_name: The ID that uniquely identifies a billing profile. Required.\n        :type billing_profile_name: str\n        :param invoice_section_name: The ID that uniquely identifies an invoice section. Required.\n        :type invoice_section_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either BillingPermissionsProperties or the result of\n         cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.billing.models.BillingPermissionsProperties]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', '2020-05-01'))
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                for i in range(10):
                    print('nop')
            if not next_link:
                request = build_list_by_invoice_sections_request(billing_account_name=billing_account_name, billing_profile_name=billing_profile_name, invoice_section_name=invoice_section_name, api_version=api_version, template_url=self.list_by_invoice_sections.metadata['url'], headers=_headers, params=_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
            else:
                request = HttpRequest('GET', next_link)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
                request.method = 'GET'
            return request

        def extract_data(pipeline_response):
            if False:
                while True:
                    i = 10
            deserialized = self._deserialize('BillingPermissionsListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                while True:
                    i = 10
            request = prepare_request(next_link)
            pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_by_invoice_sections.metadata = {'url': '/providers/Microsoft.Billing/billingAccounts/{billingAccountName}/billingProfiles/{billingProfileName}/invoiceSections/{invoiceSectionName}/billingPermissions'}

    @distributed_trace
    def list_by_billing_profile(self, billing_account_name: str, billing_profile_name: str, **kwargs: Any) -> Iterable['_models.BillingPermissionsProperties']:
        if False:
            return 10
        'Lists the billing permissions the caller has on a billing profile.\n\n        :param billing_account_name: The ID that uniquely identifies a billing account. Required.\n        :type billing_account_name: str\n        :param billing_profile_name: The ID that uniquely identifies a billing profile. Required.\n        :type billing_profile_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either BillingPermissionsProperties or the result of\n         cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.billing.models.BillingPermissionsProperties]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', '2020-05-01'))
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                return 10
            if not next_link:
                request = build_list_by_billing_profile_request(billing_account_name=billing_account_name, billing_profile_name=billing_profile_name, api_version=api_version, template_url=self.list_by_billing_profile.metadata['url'], headers=_headers, params=_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
            else:
                request = HttpRequest('GET', next_link)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
                request.method = 'GET'
            return request

        def extract_data(pipeline_response):
            if False:
                return 10
            deserialized = self._deserialize('BillingPermissionsListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                while True:
                    i = 10
            request = prepare_request(next_link)
            pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_by_billing_profile.metadata = {'url': '/providers/Microsoft.Billing/billingAccounts/{billingAccountName}/billingProfiles/{billingProfileName}/billingPermissions'}