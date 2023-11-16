import sys
from typing import Any, Callable, Dict, IO, Iterable, Optional, TypeVar, Union, overload
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

def build_list_by_billing_profile_request(billing_account_name: str, billing_profile_name: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version = kwargs.pop('api_version', _params.pop('api-version', '2020-05-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/providers/Microsoft.Billing/billingAccounts/{billingAccountName}/billingProfiles/{billingProfileName}/instructions')
    path_format_arguments = {'billingAccountName': _SERIALIZER.url('billing_account_name', billing_account_name, 'str'), 'billingProfileName': _SERIALIZER.url('billing_profile_name', billing_profile_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_request(billing_account_name: str, billing_profile_name: str, instruction_name: str, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version = kwargs.pop('api_version', _params.pop('api-version', '2020-05-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/providers/Microsoft.Billing/billingAccounts/{billingAccountName}/billingProfiles/{billingProfileName}/instructions/{instructionName}')
    path_format_arguments = {'billingAccountName': _SERIALIZER.url('billing_account_name', billing_account_name, 'str'), 'billingProfileName': _SERIALIZER.url('billing_profile_name', billing_profile_name, 'str'), 'instructionName': _SERIALIZER.url('instruction_name', instruction_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_put_request(billing_account_name: str, billing_profile_name: str, instruction_name: str, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version = kwargs.pop('api_version', _params.pop('api-version', '2020-05-01'))
    content_type = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/providers/Microsoft.Billing/billingAccounts/{billingAccountName}/billingProfiles/{billingProfileName}/instructions/{instructionName}')
    path_format_arguments = {'billingAccountName': _SERIALIZER.url('billing_account_name', billing_account_name, 'str'), 'billingProfileName': _SERIALIZER.url('billing_profile_name', billing_profile_name, 'str'), 'instructionName': _SERIALIZER.url('instruction_name', instruction_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

class InstructionsOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.billing.BillingManagementClient`'s
        :attr:`instructions` attribute.
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
    def list_by_billing_profile(self, billing_account_name: str, billing_profile_name: str, **kwargs: Any) -> Iterable['_models.Instruction']:
        if False:
            i = 10
            return i + 15
        'Lists the instructions by billing profile id.\n\n        :param billing_account_name: The ID that uniquely identifies a billing account. Required.\n        :type billing_account_name: str\n        :param billing_profile_name: The ID that uniquely identifies a billing profile. Required.\n        :type billing_profile_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either Instruction or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.billing.models.Instruction]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
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
                print('Hello World!')
            deserialized = self._deserialize('InstructionListResult', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                for i in range(10):
                    print('nop')
            request = prepare_request(next_link)
            pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_by_billing_profile.metadata = {'url': '/providers/Microsoft.Billing/billingAccounts/{billingAccountName}/billingProfiles/{billingProfileName}/instructions'}

    @distributed_trace
    def get(self, billing_account_name: str, billing_profile_name: str, instruction_name: str, **kwargs: Any) -> _models.Instruction:
        if False:
            while True:
                i = 10
        'Get the instruction by name. These are custom billing instructions and are only applicable for\n        certain customers.\n\n        :param billing_account_name: The ID that uniquely identifies a billing account. Required.\n        :type billing_account_name: str\n        :param billing_profile_name: The ID that uniquely identifies a billing profile. Required.\n        :type billing_profile_name: str\n        :param instruction_name: Instruction Name. Required.\n        :type instruction_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Instruction or the result of cls(response)\n        :rtype: ~azure.mgmt.billing.models.Instruction\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', '2020-05-01'))
        cls = kwargs.pop('cls', None)
        request = build_get_request(billing_account_name=billing_account_name, billing_profile_name=billing_profile_name, instruction_name=instruction_name, api_version=api_version, template_url=self.get.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('Instruction', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get.metadata = {'url': '/providers/Microsoft.Billing/billingAccounts/{billingAccountName}/billingProfiles/{billingProfileName}/instructions/{instructionName}'}

    @overload
    def put(self, billing_account_name: str, billing_profile_name: str, instruction_name: str, parameters: _models.Instruction, *, content_type: str='application/json', **kwargs: Any) -> _models.Instruction:
        if False:
            for i in range(10):
                print('nop')
        'Creates or updates an instruction. These are custom billing instructions and are only\n        applicable for certain customers.\n\n        :param billing_account_name: The ID that uniquely identifies a billing account. Required.\n        :type billing_account_name: str\n        :param billing_profile_name: The ID that uniquely identifies a billing profile. Required.\n        :type billing_profile_name: str\n        :param instruction_name: Instruction Name. Required.\n        :type instruction_name: str\n        :param parameters: The new instruction. Required.\n        :type parameters: ~azure.mgmt.billing.models.Instruction\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Instruction or the result of cls(response)\n        :rtype: ~azure.mgmt.billing.models.Instruction\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def put(self, billing_account_name: str, billing_profile_name: str, instruction_name: str, parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> _models.Instruction:
        if False:
            for i in range(10):
                print('nop')
        'Creates or updates an instruction. These are custom billing instructions and are only\n        applicable for certain customers.\n\n        :param billing_account_name: The ID that uniquely identifies a billing account. Required.\n        :type billing_account_name: str\n        :param billing_profile_name: The ID that uniquely identifies a billing profile. Required.\n        :type billing_profile_name: str\n        :param instruction_name: Instruction Name. Required.\n        :type instruction_name: str\n        :param parameters: The new instruction. Required.\n        :type parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Instruction or the result of cls(response)\n        :rtype: ~azure.mgmt.billing.models.Instruction\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def put(self, billing_account_name: str, billing_profile_name: str, instruction_name: str, parameters: Union[_models.Instruction, IO], **kwargs: Any) -> _models.Instruction:
        if False:
            print('Hello World!')
        "Creates or updates an instruction. These are custom billing instructions and are only\n        applicable for certain customers.\n\n        :param billing_account_name: The ID that uniquely identifies a billing account. Required.\n        :type billing_account_name: str\n        :param billing_profile_name: The ID that uniquely identifies a billing profile. Required.\n        :type billing_profile_name: str\n        :param instruction_name: Instruction Name. Required.\n        :type instruction_name: str\n        :param parameters: The new instruction. Is either a model type or a IO type. Required.\n        :type parameters: ~azure.mgmt.billing.models.Instruction or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Instruction or the result of cls(response)\n        :rtype: ~azure.mgmt.billing.models.Instruction\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
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
            _json = self._serialize.body(parameters, 'Instruction')
        request = build_put_request(billing_account_name=billing_account_name, billing_profile_name=billing_profile_name, instruction_name=instruction_name, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.put.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('Instruction', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    put.metadata = {'url': '/providers/Microsoft.Billing/billingAccounts/{billingAccountName}/billingProfiles/{billingProfileName}/instructions/{instructionName}'}