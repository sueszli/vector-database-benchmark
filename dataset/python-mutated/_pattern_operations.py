from msrest.pipeline import ClientRawResponse
from .. import models

class PatternOperations(object):
    """PatternOperations operations.

    You should not instantiate directly this class, but create a Client instance that will create it for you and attach it as attribute.

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

    def add_pattern(self, app_id, version_id, pattern=None, intent=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        "Adds a pattern to a version of the application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param pattern: The pattern text.\n        :type pattern: str\n        :param intent: The intent's name which the pattern belongs to.\n        :type intent: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: PatternRuleInfo or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.PatternRuleInfo\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        "
        pattern1 = models.PatternRuleCreateObject(pattern=pattern, intent=intent)
        url = self.add_pattern.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(pattern1, 'PatternRuleCreateObject')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [201]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 201:
            deserialized = self._deserialize('PatternRuleInfo', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    add_pattern.metadata = {'url': '/apps/{appId}/versions/{versionId}/patternrule'}

    def list_patterns(self, app_id, version_id, skip=0, take=100, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Gets patterns in a version of the application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param skip: The number of entries to skip. Default value is 0.\n        :type skip: int\n        :param take: The number of entries to return. Maximum page size is\n         500. Default is 100.\n        :type take: int\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.language.luis.authoring.models.PatternRuleInfo]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.list_patterns.metadata['url']
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
            deserialized = self._deserialize('[PatternRuleInfo]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    list_patterns.metadata = {'url': '/apps/{appId}/versions/{versionId}/patternrules'}

    def update_patterns(self, app_id, version_id, patterns, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Updates patterns in a version of the application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param patterns: An array represents the patterns.\n        :type patterns:\n         list[~azure.cognitiveservices.language.luis.authoring.models.PatternRuleUpdateObject]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.language.luis.authoring.models.PatternRuleInfo]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.update_patterns.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(patterns, '[PatternRuleUpdateObject]')
        request = self._client.put(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('[PatternRuleInfo]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    update_patterns.metadata = {'url': '/apps/{appId}/versions/{versionId}/patternrules'}

    def batch_add_patterns(self, app_id, version_id, patterns, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Adds a batch of patterns in a version of the application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param patterns: A JSON array containing patterns.\n        :type patterns:\n         list[~azure.cognitiveservices.language.luis.authoring.models.PatternRuleCreateObject]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.language.luis.authoring.models.PatternRuleInfo]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.batch_add_patterns.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(patterns, '[PatternRuleCreateObject]')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [201]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 201:
            deserialized = self._deserialize('[PatternRuleInfo]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    batch_add_patterns.metadata = {'url': '/apps/{appId}/versions/{versionId}/patternrules'}

    def delete_patterns(self, app_id, version_id, pattern_ids, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Deletes a list of patterns in a version of the application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param pattern_ids: The patterns IDs.\n        :type pattern_ids: list[str]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: OperationStatus or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.OperationStatus\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.delete_patterns.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(pattern_ids, '[str]')
        request = self._client.delete(url, query_parameters, header_parameters, body_content)
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
    delete_patterns.metadata = {'url': '/apps/{appId}/versions/{versionId}/patternrules'}

    def update_pattern(self, app_id, version_id, pattern_id, pattern, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Updates a pattern in a version of the application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param pattern_id: The pattern ID.\n        :type pattern_id: str\n        :param pattern: An object representing a pattern.\n        :type pattern:\n         ~azure.cognitiveservices.language.luis.authoring.models.PatternRuleUpdateObject\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: PatternRuleInfo or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.PatternRuleInfo\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.update_pattern.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str'), 'patternId': self._serialize.url('pattern_id', pattern_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(pattern, 'PatternRuleUpdateObject')
        request = self._client.put(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('PatternRuleInfo', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    update_pattern.metadata = {'url': '/apps/{appId}/versions/{versionId}/patternrules/{patternId}'}

    def delete_pattern(self, app_id, version_id, pattern_id, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Deletes the pattern with the specified ID from a version of the\n        application..\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param pattern_id: The pattern ID.\n        :type pattern_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: OperationStatus or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.OperationStatus\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.delete_pattern.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str'), 'patternId': self._serialize.url('pattern_id', pattern_id, 'str')}
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
    delete_pattern.metadata = {'url': '/apps/{appId}/versions/{versionId}/patternrules/{patternId}'}

    def list_intent_patterns(self, app_id, version_id, intent_id, skip=0, take=100, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Returns patterns for the specific intent in a version of the\n        application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param intent_id: The intent classifier ID.\n        :type intent_id: str\n        :param skip: The number of entries to skip. Default value is 0.\n        :type skip: int\n        :param take: The number of entries to return. Maximum page size is\n         500. Default is 100.\n        :type take: int\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.language.luis.authoring.models.PatternRuleInfo]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.list_intent_patterns.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str'), 'intentId': self._serialize.url('intent_id', intent_id, 'str')}
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
            deserialized = self._deserialize('[PatternRuleInfo]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    list_intent_patterns.metadata = {'url': '/apps/{appId}/versions/{versionId}/intents/{intentId}/patternrules'}