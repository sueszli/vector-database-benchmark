import sys
from typing import Any, Callable, Dict, IO, Iterable, Optional, TypeVar, Union, cast, overload
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
from .._vendor import _convert_request, _format_url_section
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_list_request(*, expand: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version = kwargs.pop('api_version', _params.pop('api-version', '2020-05-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/providers/Microsoft.Billing/billingAccounts')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if expand is not None:
        _params['$expand'] = _SERIALIZER.query('expand', expand, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_request(billing_account_name: str, *, expand: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version = kwargs.pop('api_version', _params.pop('api-version', '2020-05-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/providers/Microsoft.Billing/billingAccounts/{billingAccountName}')
    path_format_arguments = {'billingAccountName': _SERIALIZER.url('billing_account_name', billing_account_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if expand is not None:
        _params['$expand'] = _SERIALIZER.query('expand', expand, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_update_request(billing_account_name: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version = kwargs.pop('api_version', _params.pop('api-version', '2020-05-01'))
    content_type = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/providers/Microsoft.Billing/billingAccounts/{billingAccountName}')
    path_format_arguments = {'billingAccountName': _SERIALIZER.url('billing_account_name', billing_account_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PATCH', url=_url, params=_params, headers=_headers, **kwargs)

def build_list_invoice_sections_by_create_subscription_permission_request(billing_account_name: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version = kwargs.pop('api_version', _params.pop('api-version', '2020-05-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/providers/Microsoft.Billing/billingAccounts/{billingAccountName}/listInvoiceSectionsWithCreateSubscriptionPermission')
    path_format_arguments = {'billingAccountName': _SERIALIZER.url('billing_account_name', billing_account_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

class BillingAccountsOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.billing.BillingManagementClient`'s
        :attr:`billing_accounts` attribute.
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
    def list(self, expand: Optional[str]=None, **kwargs: Any) -> Iterable['_models.BillingAccount']:
        if False:
            for i in range(10):
                print('nop')
        'Lists the billing accounts that a user has access to.\n\n        :param expand: May be used to expand the soldTo, invoice sections and billing profiles. Default\n         value is None.\n        :type expand: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either BillingAccount or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.billing.models.BillingAccount]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', '2020-05-01'))
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                while True:
                    i = 10
            if not next_link:
                request = build_list_request(expand=expand, api_version=api_version, template_url=self.list.metadata['url'], headers=_headers, params=_params)
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
                i = 10
                return i + 15
            deserialized = self._deserialize('BillingAccountListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                print('Hello World!')
            request = prepare_request(next_link)
            pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list.metadata = {'url': '/providers/Microsoft.Billing/billingAccounts'}

    @distributed_trace
    def get(self, billing_account_name: str, expand: Optional[str]=None, **kwargs: Any) -> _models.BillingAccount:
        if False:
            for i in range(10):
                print('nop')
        'Gets a billing account by its ID.\n\n        :param billing_account_name: The ID that uniquely identifies a billing account. Required.\n        :type billing_account_name: str\n        :param expand: May be used to expand the soldTo, invoice sections and billing profiles. Default\n         value is None.\n        :type expand: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: BillingAccount or the result of cls(response)\n        :rtype: ~azure.mgmt.billing.models.BillingAccount\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', '2020-05-01'))
        cls = kwargs.pop('cls', None)
        request = build_get_request(billing_account_name=billing_account_name, expand=expand, api_version=api_version, template_url=self.get.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('BillingAccount', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get.metadata = {'url': '/providers/Microsoft.Billing/billingAccounts/{billingAccountName}'}

    def _update_initial(self, billing_account_name: str, parameters: Union[_models.BillingAccountUpdateRequest, IO], **kwargs: Any) -> Optional[_models.BillingAccount]:
        if False:
            print('Hello World!')
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', '2020-05-01'))
        content_type = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(parameters, (IO, bytes)):
            _content = parameters
        else:
            _json = self._serialize.body(parameters, 'BillingAccountUpdateRequest')
        request = build_update_request(billing_account_name=billing_account_name, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self._update_initial.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('BillingAccount', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    _update_initial.metadata = {'url': '/providers/Microsoft.Billing/billingAccounts/{billingAccountName}'}

    @overload
    def begin_update(self, billing_account_name: str, parameters: _models.BillingAccountUpdateRequest, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.BillingAccount]:
        if False:
            i = 10
            return i + 15
        'Updates the properties of a billing account. Currently, displayName and address can be updated.\n        The operation is supported only for billing accounts with agreement type Microsoft Customer\n        Agreement.\n\n        :param billing_account_name: The ID that uniquely identifies a billing account. Required.\n        :type billing_account_name: str\n        :param parameters: Request parameters that are provided to the update billing account\n         operation. Required.\n        :type parameters: ~azure.mgmt.billing.models.BillingAccountUpdateRequest\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either BillingAccount or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.billing.models.BillingAccount]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_update(self, billing_account_name: str, parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.BillingAccount]:
        if False:
            i = 10
            return i + 15
        'Updates the properties of a billing account. Currently, displayName and address can be updated.\n        The operation is supported only for billing accounts with agreement type Microsoft Customer\n        Agreement.\n\n        :param billing_account_name: The ID that uniquely identifies a billing account. Required.\n        :type billing_account_name: str\n        :param parameters: Request parameters that are provided to the update billing account\n         operation. Required.\n        :type parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either BillingAccount or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.billing.models.BillingAccount]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_update(self, billing_account_name: str, parameters: Union[_models.BillingAccountUpdateRequest, IO], **kwargs: Any) -> LROPoller[_models.BillingAccount]:
        if False:
            i = 10
            return i + 15
        "Updates the properties of a billing account. Currently, displayName and address can be updated.\n        The operation is supported only for billing accounts with agreement type Microsoft Customer\n        Agreement.\n\n        :param billing_account_name: The ID that uniquely identifies a billing account. Required.\n        :type billing_account_name: str\n        :param parameters: Request parameters that are provided to the update billing account\n         operation. Is either a model type or a IO type. Required.\n        :type parameters: ~azure.mgmt.billing.models.BillingAccountUpdateRequest or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this\n         operation to not poll, or pass in your own initialized polling object for a personal polling\n         strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either BillingAccount or the result of\n         cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.mgmt.billing.models.BillingAccount]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', '2020-05-01'))
        content_type = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls = kwargs.pop('cls', None)
        polling = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._update_initial(billing_account_name=billing_account_name, parameters=parameters, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                i = 10
                return i + 15
            deserialized = self._deserialize('BillingAccount', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        if polling is True:
            polling_method = cast(PollingMethod, ARMPolling(lro_delay, lro_options={'final-state-via': 'azure-async-operation'}, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_update.metadata = {'url': '/providers/Microsoft.Billing/billingAccounts/{billingAccountName}'}

    @distributed_trace
    def list_invoice_sections_by_create_subscription_permission(self, billing_account_name: str, **kwargs: Any) -> Iterable['_models.InvoiceSectionWithCreateSubPermission']:
        if False:
            for i in range(10):
                print('nop')
        'Lists the invoice sections for which the user has permission to create Azure subscriptions. The\n        operation is supported only for billing accounts with agreement type Microsoft Customer\n        Agreement.\n\n        :param billing_account_name: The ID that uniquely identifies a billing account. Required.\n        :type billing_account_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either InvoiceSectionWithCreateSubPermission or the\n         result of cls(response)\n        :rtype:\n         ~azure.core.paging.ItemPaged[~azure.mgmt.billing.models.InvoiceSectionWithCreateSubPermission]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
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
                request = build_list_invoice_sections_by_create_subscription_permission_request(billing_account_name=billing_account_name, api_version=api_version, template_url=self.list_invoice_sections_by_create_subscription_permission.metadata['url'], headers=_headers, params=_params)
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
            deserialized = self._deserialize('InvoiceSectionListWithCreateSubPermissionResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                print('Hello World!')
            request = prepare_request(next_link)
            pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_invoice_sections_by_create_subscription_permission.metadata = {'url': '/providers/Microsoft.Billing/billingAccounts/{billingAccountName}/listInvoiceSectionsWithCreateSubscriptionPermission'}