from msrest.pipeline import ClientRawResponse
from .. import models

class AlterationsOperations(object):
    """AlterationsOperations operations.

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

    def get(self, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Download alterations from runtime.\n\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: WordAlterationsDTO or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.knowledge.qnamaker.models.WordAlterationsDTO\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.knowledge.qnamaker.models.ErrorResponseException>`\n        '
        url = self.get.metadata['url']
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
            deserialized = self._deserialize('WordAlterationsDTO', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get.metadata = {'url': '/alterations'}

    def replace(self, word_alterations, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Replace alterations data.\n\n        :param word_alterations: Collection of word alterations.\n        :type word_alterations:\n         list[~azure.cognitiveservices.knowledge.qnamaker.models.AlterationsDTO]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.knowledge.qnamaker.models.ErrorResponseException>`\n        '
        word_alterations1 = models.WordAlterationsDTO(word_alterations=word_alterations)
        url = self.replace.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(word_alterations1, 'WordAlterationsDTO')
        request = self._client.put(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [204]:
            raise models.ErrorResponseException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    replace.metadata = {'url': '/alterations'}

    def get_alterations_for_kb(self, kb_id, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Download alterations per Knowledgebase (QnAMaker Managed).\n\n        :param kb_id: Knowledgebase id.\n        :type kb_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: WordAlterationsDTO or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.knowledge.qnamaker.models.WordAlterationsDTO\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.knowledge.qnamaker.models.ErrorResponseException>`\n        '
        url = self.get_alterations_for_kb.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'kbId': self._serialize.url('kb_id', kb_id, 'str')}
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
            deserialized = self._deserialize('WordAlterationsDTO', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_alterations_for_kb.metadata = {'url': '/alterations/{kbId}'}

    def replace_alterations_for_kb(self, kb_id, word_alterations, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Replace alterations data per Knowledgebase (QnAMaker Managed).\n\n        :param kb_id: Knowledgebase id.\n        :type kb_id: str\n        :param word_alterations: Collection of word alterations.\n        :type word_alterations:\n         list[~azure.cognitiveservices.knowledge.qnamaker.models.AlterationsDTO]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.knowledge.qnamaker.models.ErrorResponseException>`\n        '
        word_alterations1 = models.WordAlterationsDTO(word_alterations=word_alterations)
        url = self.replace_alterations_for_kb.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'kbId': self._serialize.url('kb_id', kb_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(word_alterations1, 'WordAlterationsDTO')
        request = self._client.put(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [204]:
            raise models.ErrorResponseException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    replace_alterations_for_kb.metadata = {'url': '/alterations/{kbId}'}