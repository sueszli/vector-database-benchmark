from copy import deepcopy
from typing import TYPE_CHECKING
from msrest import Deserializer, Serializer
from azure.core import PipelineClient
from . import models
from ._configuration import CommunicationNetworkTraversalClientConfiguration
from .operations import CommunicationNetworkTraversalOperations
if TYPE_CHECKING:
    from typing import Any
    from azure.core.rest import HttpRequest, HttpResponse

class CommunicationNetworkTraversalClient(object):
    """Azure Communication Network Traversal Service.

    :ivar communication_network_traversal: CommunicationNetworkTraversalOperations operations
    :vartype communication_network_traversal:
     azure.communication.networktraversal.operations.CommunicationNetworkTraversalOperations
    :param endpoint: The communication resource, for example
     https://my-resource.communication.azure.com.
    :type endpoint: str
    :keyword api_version: Api Version. Default value is "2022-03-01-preview". Note that overriding
     this default value may result in unsupported behavior.
    :paramtype api_version: str
    """

    def __init__(self, endpoint, **kwargs):
        if False:
            return 10
        _base_url = '{endpoint}'
        self._config = CommunicationNetworkTraversalClientConfiguration(endpoint=endpoint, **kwargs)
        self._client = PipelineClient(base_url=_base_url, config=self._config, **kwargs)
        client_models = {k: v for (k, v) in models.__dict__.items() if isinstance(v, type)}
        self._serialize = Serializer(client_models)
        self._deserialize = Deserializer(client_models)
        self._serialize.client_side_validation = False
        self.communication_network_traversal = CommunicationNetworkTraversalOperations(self._client, self._config, self._serialize, self._deserialize)

    def _send_request(self, request, **kwargs):
        if False:
            return 10
        'Runs the network request through the client\'s chained policies.\n\n        >>> from azure.core.rest import HttpRequest\n        >>> request = HttpRequest("GET", "https://www.example.org/")\n        <HttpRequest [GET], url: \'https://www.example.org/\'>\n        >>> response = client._send_request(request)\n        <HttpResponse: 200 OK>\n\n        For more information on this code flow, see https://aka.ms/azsdk/python/protocol/quickstart\n\n        :param request: The network request you want to make. Required.\n        :type request: ~azure.core.rest.HttpRequest\n        :keyword bool stream: Whether the response payload will be streamed. Defaults to False.\n        :return: The response of your network call. Does not do error handling on your response.\n        :rtype: ~azure.core.rest.HttpResponse\n        '
        request_copy = deepcopy(request)
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
        request_copy.url = self._client.format_url(request_copy.url, **path_format_arguments)
        return self._client.send_request(request_copy, **kwargs)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        self._client.close()

    def __enter__(self):
        if False:
            return 10
        self._client.__enter__()
        return self

    def __exit__(self, *exc_details):
        if False:
            return 10
        self._client.__exit__(*exc_details)