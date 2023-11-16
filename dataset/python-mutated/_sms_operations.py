from typing import TYPE_CHECKING
import warnings
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceExistsError, ResourceNotFoundError, map_error
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import HttpRequest, HttpResponse
from .. import models as _models
if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Generic, Optional, TypeVar
    T = TypeVar('T')
    ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]

class SmsOperations(object):
    """SmsOperations operations.

    You should not instantiate this class directly. Instead, you should create a Client instance that
    instantiates it for you and attaches it as an attribute.

    :ivar models: Alias to model classes used in this operation group.
    :type models: ~azure.communication.sms.models
    :param client: Client for service requests.
    :param config: Configuration of service client.
    :param serializer: An object model serializer.
    :param deserializer: An object model deserializer.
    """
    models = _models

    def __init__(self, client, config, serializer, deserializer):
        if False:
            i = 10
            return i + 15
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self._config = config

    def send(self, send_message_request, **kwargs):
        if False:
            while True:
                i = 10
        'Sends a SMS message from a phone number that belongs to the authenticated account.\n\n        Sends a SMS message from a phone number that belongs to the authenticated account.\n\n        :param send_message_request: Represents the body of the send message request.\n        :type send_message_request: ~azure.communication.sms.models.SendMessageRequest\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: SmsSendResponse, or the result of cls(response)\n        :rtype: ~azure.communication.sms.models.SmsSendResponse\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2021-03-07'
        content_type = kwargs.pop('content_type', 'application/json')
        accept = 'application/json'
        url = self.send.metadata['url']
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        body_content_kwargs = {}
        body_content = self._serialize.body(send_message_request, 'SendMessageRequest')
        body_content_kwargs['content'] = body_content
        request = self._client.post(url, query_parameters, header_parameters, **body_content_kwargs)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [202]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response)
        deserialized = self._deserialize('SmsSendResponse', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    send.metadata = {'url': '/sms'}