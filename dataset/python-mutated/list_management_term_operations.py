from msrest.pipeline import ClientRawResponse
from .. import models

class ListManagementTermOperations(object):
    """ListManagementTermOperations operations.

    :param client: Client for service requests.
    :param config: Configuration of service client.
    :param serializer: An object model serializer.
    :param deserializer: An object model deserializer.
    """
    models = models

    def __init__(self, client, config, serializer, deserializer):
        if False:
            return 10
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self.config = config

    def add_term(self, list_id, term, language, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Add a term to the term list with list Id equal to list Id passed.\n\n        :param list_id: List Id of the image list.\n        :type list_id: str\n        :param term: Term to be deleted\n        :type term: str\n        :param language: Language of the terms.\n        :type language: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: object or ClientRawResponse if raw=true\n        :rtype: object or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.add_term.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'listId': self._serialize.url('list_id', list_id, 'str'), 'term': self._serialize.url('term', term, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['language'] = self._serialize.query('language', language, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.post(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [201]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 201:
            deserialized = self._deserialize('object', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    add_term.metadata = {'url': '/contentmoderator/lists/v1.0/termlists/{listId}/terms/{term}'}

    def delete_term(self, list_id, term, language, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Deletes a term from the list with list Id equal to the list Id passed.\n\n        :param list_id: List Id of the image list.\n        :type list_id: str\n        :param term: Term to be deleted\n        :type term: str\n        :param language: Language of the terms.\n        :type language: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: str or ClientRawResponse if raw=true\n        :rtype: str or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.delete_term.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'listId': self._serialize.url('list_id', list_id, 'str'), 'term': self._serialize.url('term', term, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['language'] = self._serialize.query('language', language, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.delete(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [204]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 204:
            deserialized = self._deserialize('str', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    delete_term.metadata = {'url': '/contentmoderator/lists/v1.0/termlists/{listId}/terms/{term}'}

    def get_all_terms(self, list_id, language, offset=None, limit=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Gets all terms from the list with list Id equal to the list Id passed.\n\n        :param list_id: List Id of the image list.\n        :type list_id: str\n        :param language: Language of the terms.\n        :type language: str\n        :param offset: The pagination start index.\n        :type offset: int\n        :param limit: The max limit.\n        :type limit: int\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Terms or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.contentmoderator.models.Terms\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.get_all_terms.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'listId': self._serialize.url('list_id', list_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['language'] = self._serialize.query('language', language, 'str')
        if offset is not None:
            query_parameters['offset'] = self._serialize.query('offset', offset, 'int')
        if limit is not None:
            query_parameters['limit'] = self._serialize.query('limit', limit, 'int')
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
            deserialized = self._deserialize('Terms', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_all_terms.metadata = {'url': '/contentmoderator/lists/v1.0/termlists/{listId}/terms'}

    def delete_all_terms(self, list_id, language, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Deletes all terms from the list with list Id equal to the list Id\n        passed.\n\n        :param list_id: List Id of the image list.\n        :type list_id: str\n        :param language: Language of the terms.\n        :type language: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: str or ClientRawResponse if raw=true\n        :rtype: str or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.delete_all_terms.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'listId': self._serialize.url('list_id', list_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['language'] = self._serialize.query('language', language, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.delete(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [204]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 204:
            deserialized = self._deserialize('str', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    delete_all_terms.metadata = {'url': '/contentmoderator/lists/v1.0/termlists/{listId}/terms'}