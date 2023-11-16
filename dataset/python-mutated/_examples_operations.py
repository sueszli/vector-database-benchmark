from msrest.pipeline import ClientRawResponse
from .. import models

class ExamplesOperations(object):
    """ExamplesOperations operations.

    You should not instantiate directly this class, but create a Client instance that will create it for you and attach it as attribute.

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

    def add(self, app_id, version_id, example_label_object, enable_nested_children=False, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Adds a labeled example utterance in a version of the application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param example_label_object: A labeled example utterance with the\n         expected intent and entities.\n        :type example_label_object:\n         ~azure.cognitiveservices.language.luis.authoring.models.ExampleLabelObject\n        :param enable_nested_children: Toggles nested/flat format\n        :type enable_nested_children: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: LabelExampleResponse or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.LabelExampleResponse\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.add.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if enable_nested_children is not None:
            query_parameters['enableNestedChildren'] = self._serialize.query('enable_nested_children', enable_nested_children, 'bool')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(example_label_object, 'ExampleLabelObject')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [201]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 201:
            deserialized = self._deserialize('LabelExampleResponse', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    add.metadata = {'url': '/apps/{appId}/versions/{versionId}/example'}

    def batch(self, app_id, version_id, example_label_object_array, enable_nested_children=False, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Adds a batch of labeled example utterances to a version of the\n        application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param example_label_object_array: Array of example utterances.\n        :type example_label_object_array:\n         list[~azure.cognitiveservices.language.luis.authoring.models.ExampleLabelObject]\n        :param enable_nested_children: Toggles nested/flat format\n        :type enable_nested_children: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.language.luis.authoring.models.BatchLabelExample]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.batch.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if enable_nested_children is not None:
            query_parameters['enableNestedChildren'] = self._serialize.query('enable_nested_children', enable_nested_children, 'bool')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(example_label_object_array, '[ExampleLabelObject]')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [201, 207]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 201:
            deserialized = self._deserialize('[BatchLabelExample]', response)
        if response.status_code == 207:
            deserialized = self._deserialize('[BatchLabelExample]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    batch.metadata = {'url': '/apps/{appId}/versions/{versionId}/examples'}

    def list(self, app_id, version_id, skip=0, take=100, enable_nested_children=False, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Returns example utterances to be reviewed from a version of the\n        application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param skip: The number of entries to skip. Default value is 0.\n        :type skip: int\n        :param take: The number of entries to return. Maximum page size is\n         500. Default is 100.\n        :type take: int\n        :param enable_nested_children: Toggles nested/flat format\n        :type enable_nested_children: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.language.luis.authoring.models.LabeledUtterance]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.list.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if skip is not None:
            query_parameters['skip'] = self._serialize.query('skip', skip, 'int', minimum=0)
        if take is not None:
            query_parameters['take'] = self._serialize.query('take', take, 'int', maximum=500, minimum=0)
        if enable_nested_children is not None:
            query_parameters['enableNestedChildren'] = self._serialize.query('enable_nested_children', enable_nested_children, 'bool')
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
            deserialized = self._deserialize('[LabeledUtterance]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    list.metadata = {'url': '/apps/{appId}/versions/{versionId}/examples'}

    def delete(self, app_id, version_id, example_id, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Deletes the labeled example utterances with the specified ID from a\n        version of the application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param example_id: The example ID.\n        :type example_id: int\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: OperationStatus or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.OperationStatus\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.delete.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str'), 'exampleId': self._serialize.url('example_id', example_id, 'int')}
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
    delete.metadata = {'url': '/apps/{appId}/versions/{versionId}/examples/{exampleId}'}