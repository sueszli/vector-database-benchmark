from msrest.pipeline import ClientRawResponse
from .. import models

class EndpointKeysOperations(object):
    """EndpointKeysOperations operations.

    :param client: Client for service requests.
    :param config: Configuration of service client.
    :param serializer: An object model serializer.
    :param deserializer: An object model deserializer.
    """
    models = models

    def __init__(self, client, config, serializer, deserializer):
        if False:
            print('Hello World!')
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self.config = config

    def get_keys(self, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Gets endpoint keys for an endpoint.\n\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: EndpointKeysDTO or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.knowledge.qnamaker.models.EndpointKeysDTO or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.knowledge.qnamaker.models.ErrorResponseException>`\n        '
        url = self.get_keys.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('EndpointKeysDTO', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_keys.metadata = {'url': '/endpointkeys'}

    def refresh_keys(self, key_type, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Re-generates an endpoint key.\n\n        :param key_type: Type of Key\n        :type key_type: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: EndpointKeysDTO or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.knowledge.qnamaker.models.EndpointKeysDTO or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.knowledge.qnamaker.models.ErrorResponseException>`\n        '
        url = self.refresh_keys.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'keyType': self._serialize.url('key_type', key_type, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.patch(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('EndpointKeysDTO', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    refresh_keys.metadata = {'url': '/endpointkeys/{keyType}'}