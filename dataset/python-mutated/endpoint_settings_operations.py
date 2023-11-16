from msrest.pipeline import ClientRawResponse
from .. import models

class EndpointSettingsOperations(object):
    """EndpointSettingsOperations operations.

    :param client: Client for service requests.
    :param config: Configuration of service client.
    :param serializer: An object model serializer.
    :param deserializer: An object model deserializer.
    """
    models = models

    def __init__(self, client, config, serializer, deserializer):
        if False:
            while True:
                i = 10
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self.config = config

    def get_settings(self, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Gets endpoint settings for an endpoint.\n\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: EndpointSettingsDTO or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.knowledge.qnamaker.models.EndpointSettingsDTO\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.knowledge.qnamaker.models.ErrorResponseException>`\n        '
        url = self.get_settings.metadata['url']
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
            deserialized = self._deserialize('EndpointSettingsDTO', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_settings.metadata = {'url': '/endpointSettings'}

    def update_settings(self, active_learning=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Updates endpoint settings for an endpoint.\n\n        :param active_learning: Active Learning settings of the endpoint.\n        :type active_learning:\n         ~azure.cognitiveservices.knowledge.qnamaker.models.EndpointSettingsDTOActiveLearning\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.knowledge.qnamaker.models.ErrorResponseException>`\n        '
        endpoint_settings_payload = models.EndpointSettingsDTO(active_learning=active_learning)
        url = self.update_settings.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(endpoint_settings_payload, 'EndpointSettingsDTO')
        request = self._client.patch(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [204]:
            raise models.ErrorResponseException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    update_settings.metadata = {'url': '/endpointSettings'}