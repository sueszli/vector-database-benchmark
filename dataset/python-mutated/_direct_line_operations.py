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
from .._vendor import _convert_request, _format_url_section
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_regenerate_keys_request(resource_group_name: str, resource_name: str, channel_name: Union[str, _models.RegenerateKeysChannelName], subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: Literal['2022-09-15'] = kwargs.pop('api_version', _params.pop('api-version', '2022-09-15'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.BotService/botServices/{resourceName}/channels/{channelName}/regeneratekeys')
    path_format_arguments = {'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=64, min_length=2, pattern='^[a-zA-Z0-9][a-zA-Z0-9_.-]*$'), 'resourceName': _SERIALIZER.url('resource_name', resource_name, 'str', max_length=64, min_length=2, pattern='^[a-zA-Z0-9][a-zA-Z0-9_.-]*$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'channelName': _SERIALIZER.url('channel_name', channel_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

class DirectLineOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.botservice.AzureBotService`'s
        :attr:`direct_line` attribute.
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
    def regenerate_keys(self, resource_group_name: str, resource_name: str, channel_name: Union[str, _models.RegenerateKeysChannelName], parameters: _models.SiteInfo, *, content_type: str='application/json', **kwargs: Any) -> _models.BotChannel:
        if False:
            return 10
        'Regenerates secret keys and returns them for the DirectLine Channel of a particular BotService\n        resource.\n\n        :param resource_group_name: The name of the Bot resource group in the user subscription.\n         Required.\n        :type resource_group_name: str\n        :param resource_name: The name of the Bot resource. Required.\n        :type resource_name: str\n        :param channel_name: The name of the Channel resource for which keys are to be regenerated.\n         Known values are: "WebChatChannel" and "DirectLineChannel". Required.\n        :type channel_name: str or ~azure.mgmt.botservice.models.RegenerateKeysChannelName\n        :param parameters: The parameters to provide for the created bot. Required.\n        :type parameters: ~azure.mgmt.botservice.models.SiteInfo\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: BotChannel or the result of cls(response)\n        :rtype: ~azure.mgmt.botservice.models.BotChannel\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def regenerate_keys(self, resource_group_name: str, resource_name: str, channel_name: Union[str, _models.RegenerateKeysChannelName], parameters: IO, *, content_type: str='application/json', **kwargs: Any) -> _models.BotChannel:
        if False:
            return 10
        'Regenerates secret keys and returns them for the DirectLine Channel of a particular BotService\n        resource.\n\n        :param resource_group_name: The name of the Bot resource group in the user subscription.\n         Required.\n        :type resource_group_name: str\n        :param resource_name: The name of the Bot resource. Required.\n        :type resource_name: str\n        :param channel_name: The name of the Channel resource for which keys are to be regenerated.\n         Known values are: "WebChatChannel" and "DirectLineChannel". Required.\n        :type channel_name: str or ~azure.mgmt.botservice.models.RegenerateKeysChannelName\n        :param parameters: The parameters to provide for the created bot. Required.\n        :type parameters: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: BotChannel or the result of cls(response)\n        :rtype: ~azure.mgmt.botservice.models.BotChannel\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def regenerate_keys(self, resource_group_name: str, resource_name: str, channel_name: Union[str, _models.RegenerateKeysChannelName], parameters: Union[_models.SiteInfo, IO], **kwargs: Any) -> _models.BotChannel:
        if False:
            for i in range(10):
                print('nop')
        'Regenerates secret keys and returns them for the DirectLine Channel of a particular BotService\n        resource.\n\n        :param resource_group_name: The name of the Bot resource group in the user subscription.\n         Required.\n        :type resource_group_name: str\n        :param resource_name: The name of the Bot resource. Required.\n        :type resource_name: str\n        :param channel_name: The name of the Channel resource for which keys are to be regenerated.\n         Known values are: "WebChatChannel" and "DirectLineChannel". Required.\n        :type channel_name: str or ~azure.mgmt.botservice.models.RegenerateKeysChannelName\n        :param parameters: The parameters to provide for the created bot. Is either a model type or a\n         IO type. Required.\n        :type parameters: ~azure.mgmt.botservice.models.SiteInfo or IO\n        :keyword content_type: Body Parameter content-type. Known values are: \'application/json\'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: BotChannel or the result of cls(response)\n        :rtype: ~azure.mgmt.botservice.models.BotChannel\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: Literal['2022-09-15'] = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.BotChannel] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(parameters, (IO, bytes)):
            _content = parameters
        else:
            _json = self._serialize.body(parameters, 'SiteInfo')
        request = build_regenerate_keys_request(resource_group_name=resource_group_name, resource_name=resource_name, channel_name=channel_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.regenerate_keys.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.Error, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('BotChannel', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    regenerate_keys.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.BotService/botServices/{resourceName}/channels/{channelName}/regeneratekeys'}