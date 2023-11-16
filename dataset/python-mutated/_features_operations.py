from msrest.pipeline import ClientRawResponse
from .. import models

class FeaturesOperations(object):
    """FeaturesOperations operations.

    You should not instantiate directly this class, but create a Client instance that will create it for you and attach it as attribute.

    :param client: Client for service requests.
    :param config: Configuration of service client.
    :param serializer: An object model serializer.
    :param deserializer: An object model deserializer.
    """
    models = models

    def __init__(self, client, config, serializer, deserializer):
        if False:
            for i in range(10):
                print('nop')
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self.config = config

    def add_phrase_list(self, app_id, version_id, phraselist_create_object, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Creates a new phraselist feature in a version of the application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param phraselist_create_object: A Phraselist object containing Name,\n         comma-separated Phrases and the isExchangeable boolean. Default value\n         for isExchangeable is true.\n        :type phraselist_create_object:\n         ~azure.cognitiveservices.language.luis.authoring.models.PhraselistCreateObject\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: int or ClientRawResponse if raw=true\n        :rtype: int or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.add_phrase_list.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(phraselist_create_object, 'PhraselistCreateObject')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [201]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 201:
            deserialized = self._deserialize('int', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    add_phrase_list.metadata = {'url': '/apps/{appId}/versions/{versionId}/phraselists'}

    def list_phrase_lists(self, app_id, version_id, skip=0, take=100, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Gets all the phraselist features in a version of the application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param skip: The number of entries to skip. Default value is 0.\n        :type skip: int\n        :param take: The number of entries to return. Maximum page size is\n         500. Default is 100.\n        :type take: int\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.language.luis.authoring.models.PhraseListFeatureInfo]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.list_phrase_lists.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if skip is not None:
            query_parameters['skip'] = self._serialize.query('skip', skip, 'int', minimum=0)
        if take is not None:
            query_parameters['take'] = self._serialize.query('take', take, 'int', maximum=500, minimum=0)
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
            deserialized = self._deserialize('[PhraseListFeatureInfo]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    list_phrase_lists.metadata = {'url': '/apps/{appId}/versions/{versionId}/phraselists'}

    def list(self, app_id, version_id, skip=0, take=100, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Gets all the extraction phraselist and pattern features in a version of\n        the application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param skip: The number of entries to skip. Default value is 0.\n        :type skip: int\n        :param take: The number of entries to return. Maximum page size is\n         500. Default is 100.\n        :type take: int\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: FeaturesResponseObject or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.FeaturesResponseObject\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.list.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if skip is not None:
            query_parameters['skip'] = self._serialize.query('skip', skip, 'int', minimum=0)
        if take is not None:
            query_parameters['take'] = self._serialize.query('take', take, 'int', maximum=500, minimum=0)
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
            deserialized = self._deserialize('FeaturesResponseObject', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    list.metadata = {'url': '/apps/{appId}/versions/{versionId}/features'}

    def get_phrase_list(self, app_id, version_id, phraselist_id, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Gets phraselist feature info in a version of the application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param phraselist_id: The ID of the feature to be retrieved.\n        :type phraselist_id: int\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: PhraseListFeatureInfo or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.PhraseListFeatureInfo\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.get_phrase_list.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str'), 'phraselistId': self._serialize.url('phraselist_id', phraselist_id, 'int')}
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
            deserialized = self._deserialize('PhraseListFeatureInfo', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_phrase_list.metadata = {'url': '/apps/{appId}/versions/{versionId}/phraselists/{phraselistId}'}

    def update_phrase_list(self, app_id, version_id, phraselist_id, phraselist_update_object=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Updates the phrases, the state and the name of the phraselist feature\n        in a version of the application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param phraselist_id: The ID of the feature to be updated.\n        :type phraselist_id: int\n        :param phraselist_update_object: The new values for: - Just a boolean\n         called IsActive, in which case the status of the feature will be\n         changed. - Name, Pattern, Mode, and a boolean called IsActive to\n         update the feature.\n        :type phraselist_update_object:\n         ~azure.cognitiveservices.language.luis.authoring.models.PhraselistUpdateObject\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: OperationStatus or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.OperationStatus\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.update_phrase_list.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str'), 'phraselistId': self._serialize.url('phraselist_id', phraselist_id, 'int')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        if phraselist_update_object is not None:
            body_content = self._serialize.body(phraselist_update_object, 'PhraselistUpdateObject')
        else:
            body_content = None
        request = self._client.put(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('OperationStatus', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    update_phrase_list.metadata = {'url': '/apps/{appId}/versions/{versionId}/phraselists/{phraselistId}'}

    def delete_phrase_list(self, app_id, version_id, phraselist_id, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Deletes a phraselist feature from a version of the application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param phraselist_id: The ID of the feature to be deleted.\n        :type phraselist_id: int\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: OperationStatus or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.OperationStatus\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.delete_phrase_list.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str'), 'phraselistId': self._serialize.url('phraselist_id', phraselist_id, 'int')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.delete(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('OperationStatus', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    delete_phrase_list.metadata = {'url': '/apps/{appId}/versions/{versionId}/phraselists/{phraselistId}'}

    def add_intent_feature(self, app_id, version_id, intent_id, feature_relation_create_object, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Adds a new feature relation to be used by the intent in a version of\n        the application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param intent_id: The intent classifier ID.\n        :type intent_id: str\n        :param feature_relation_create_object: A Feature relation information\n         object.\n        :type feature_relation_create_object:\n         ~azure.cognitiveservices.language.luis.authoring.models.ModelFeatureInformation\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: OperationStatus or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.OperationStatus\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.add_intent_feature.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str'), 'intentId': self._serialize.url('intent_id', intent_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(feature_relation_create_object, 'ModelFeatureInformation')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('OperationStatus', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    add_intent_feature.metadata = {'url': '/apps/{appId}/versions/{versionId}/intents/{intentId}/features'}

    def add_entity_feature(self, app_id, version_id, entity_id, feature_relation_create_object, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Adds a new feature relation to be used by the entity in a version of\n        the application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param entity_id: The entity extractor ID.\n        :type entity_id: str\n        :param feature_relation_create_object: A Feature relation information\n         object.\n        :type feature_relation_create_object:\n         ~azure.cognitiveservices.language.luis.authoring.models.ModelFeatureInformation\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: OperationStatus or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.OperationStatus\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.add_entity_feature.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str'), 'entityId': self._serialize.url('entity_id', entity_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(feature_relation_create_object, 'ModelFeatureInformation')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('OperationStatus', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    add_entity_feature.metadata = {'url': '/apps/{appId}/versions/{versionId}/entities/{entityId}/features'}