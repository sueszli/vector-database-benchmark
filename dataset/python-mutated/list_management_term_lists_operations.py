from msrest.pipeline import ClientRawResponse
from .. import models

class ListManagementTermListsOperations(object):
    """ListManagementTermListsOperations operations.

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

    def get_details(self, list_id, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Returns list Id details of the term list with list Id equal to list Id\n        passed.\n\n        :param list_id: List Id of the image list.\n        :type list_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: TermList or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.contentmoderator.models.TermList or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.get_details.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'listId': self._serialize.url('list_id', list_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('TermList', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_details.metadata = {'url': '/contentmoderator/lists/v1.0/termlists/{listId}'}

    def delete(self, list_id, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Deletes term list with the list Id equal to list Id passed.\n\n        :param list_id: List Id of the image list.\n        :type list_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: str or ClientRawResponse if raw=true\n        :rtype: str or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.delete.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'listId': self._serialize.url('list_id', list_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.delete(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('str', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    delete.metadata = {'url': '/contentmoderator/lists/v1.0/termlists/{listId}'}

    def update(self, list_id, content_type, body, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Updates an Term List.\n\n        :param list_id: List Id of the image list.\n        :type list_id: str\n        :param content_type: The content type.\n        :type content_type: str\n        :param body: Schema of the body.\n        :type body:\n         ~azure.cognitiveservices.vision.contentmoderator.models.Body\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: TermList or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.contentmoderator.models.TermList or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.update.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'listId': self._serialize.url('list_id', list_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        body_content = self._serialize.body(body, 'Body')
        request = self._client.put(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('TermList', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    update.metadata = {'url': '/contentmoderator/lists/v1.0/termlists/{listId}'}

    def create(self, content_type, body, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Creates a Term List.\n\n        :param content_type: The content type.\n        :type content_type: str\n        :param body: Schema of the body.\n        :type body:\n         ~azure.cognitiveservices.vision.contentmoderator.models.Body\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: TermList or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.contentmoderator.models.TermList or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.create.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        body_content = self._serialize.body(body, 'Body')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('TermList', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    create.metadata = {'url': '/contentmoderator/lists/v1.0/termlists'}

    def get_all_term_lists(self, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'gets all the Term Lists.\n\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.vision.contentmoderator.models.TermList]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.get_all_term_lists.metadata['url']
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
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('[TermList]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_all_term_lists.metadata = {'url': '/contentmoderator/lists/v1.0/termlists'}

    def refresh_index_method(self, list_id, language, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Refreshes the index of the list with list Id equal to list ID passed.\n\n        :param list_id: List Id of the image list.\n        :type list_id: str\n        :param language: Language of the terms.\n        :type language: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: RefreshIndex or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.contentmoderator.models.RefreshIndex\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.refresh_index_method.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'listId': self._serialize.url('list_id', list_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['language'] = self._serialize.query('language', language, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.post(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('RefreshIndex', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    refresh_index_method.metadata = {'url': '/contentmoderator/lists/v1.0/termlists/{listId}/RefreshIndex'}