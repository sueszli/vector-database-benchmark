import sys
from typing import Any, Callable, Dict, IO, Optional, TypeVar, Union, overload
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceExistsError, ResourceNotFoundError, ResourceNotModifiedError, map_error
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import HttpResponse
from azure.core.rest import HttpRequest
from azure.core.tracing.decorator import distributed_trace
from azure.core.utils import case_insensitive_dict
from azure.mgmt.core.exceptions import ARMErrorFormat
from .. import models as _models
from .._serialization import Serializer
from .._vendor import _convert_request
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_validate_request(**kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version = kwargs.pop('api_version', _params.pop('api-version', '2020-05-01'))
    content_type = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/providers/Microsoft.Billing/validateAddress')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

class AddressOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.billing.BillingManagementClient`'s
        :attr:`address` attribute.
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

    @overload
    def validate(self, address: _models.AddressDetails, *, content_type: str='application/json', **kwargs: Any) -> _models.ValidateAddressResponse:
        if False:
            print('Hello World!')
        'Validates an address. Use the operation to validate an address before using it as soldTo or a\n        billTo address.\n\n        :param address: Required.\n        :type address: ~azure.mgmt.billing.models.AddressDetails\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ValidateAddressResponse or the result of cls(response)\n        :rtype: ~azure.mgmt.billing.models.ValidateAddressResponse\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def validate(self, address: IO, *, content_type: str='application/json', **kwargs: Any) -> _models.ValidateAddressResponse:
        if False:
            i = 10
            return i + 15
        'Validates an address. Use the operation to validate an address before using it as soldTo or a\n        billTo address.\n\n        :param address: Required.\n        :type address: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ValidateAddressResponse or the result of cls(response)\n        :rtype: ~azure.mgmt.billing.models.ValidateAddressResponse\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def validate(self, address: Union[_models.AddressDetails, IO], **kwargs: Any) -> _models.ValidateAddressResponse:
        if False:
            i = 10
            return i + 15
        "Validates an address. Use the operation to validate an address before using it as soldTo or a\n        billTo address.\n\n        :param address: Is either a model type or a IO type. Required.\n        :type address: ~azure.mgmt.billing.models.AddressDetails or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ValidateAddressResponse or the result of cls(response)\n        :rtype: ~azure.mgmt.billing.models.ValidateAddressResponse\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
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
        if isinstance(address, (IO, bytes)):
            _content = address
        else:
            _json = self._serialize.body(address, 'AddressDetails')
        request = build_validate_request(api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.validate.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('ValidateAddressResponse', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    validate.metadata = {'url': '/providers/Microsoft.Billing/validateAddress'}