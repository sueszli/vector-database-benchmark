from msrest.pipeline import ClientRawResponse
from msrest.exceptions import HttpOperationError
from .. import models

class VersionsOperations(object):
    """VersionsOperations operations.

    You should not instantiate directly this class, but create a Client instance that will create it for you and attach it as attribute.

    :param client: Client for service requests.
    :param config: Configuration of service client.
    :param serializer: An object model serializer.
    :param deserializer: An object model deserializer.
    :ivar format: Lu format extension. Constant value: "lu".
    """
    models = models

    def __init__(self, client, config, serializer, deserializer):
        if False:
            print('Hello World!')
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self.config = config
        self.format = 'lu'

    def clone(self, app_id, version_id, version=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Creates a new version from the selected version.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param version: The new version for the cloned model.\n        :type version: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: str or ClientRawResponse if raw=true\n        :rtype: str or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        version_clone_object = models.TaskUpdateObject(version=version)
        url = self.clone.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(version_clone_object, 'TaskUpdateObject')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [201]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 201:
            deserialized = self._deserialize('str', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    clone.metadata = {'url': '/apps/{appId}/versions/{versionId}/clone'}

    def list(self, app_id, skip=0, take=100, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Gets a list of versions for this application ID.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param skip: The number of entries to skip. Default value is 0.\n        :type skip: int\n        :param take: The number of entries to return. Maximum page size is\n         500. Default is 100.\n        :type take: int\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.language.luis.authoring.models.VersionInfo]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.list.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str')}
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
            deserialized = self._deserialize('[VersionInfo]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    list.metadata = {'url': '/apps/{appId}/versions'}

    def get(self, app_id, version_id, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Gets the version information such as date created, last modified date,\n        endpoint URL, count of intents and entities, training and publishing\n        status.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: VersionInfo or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.VersionInfo or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.get.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str')}
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
            deserialized = self._deserialize('VersionInfo', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get.metadata = {'url': '/apps/{appId}/versions/{versionId}/'}

    def update(self, app_id, version_id, version=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Updates the name or description of the application version.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param version: The new version for the cloned model.\n        :type version: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: OperationStatus or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.OperationStatus\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        version_update_object = models.TaskUpdateObject(version=version)
        url = self.update.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(version_update_object, 'TaskUpdateObject')
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
    update.metadata = {'url': '/apps/{appId}/versions/{versionId}/'}

    def delete(self, app_id, version_id, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Deletes an application version.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: OperationStatus or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.OperationStatus\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.delete.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str')}
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
    delete.metadata = {'url': '/apps/{appId}/versions/{versionId}/'}

    def export(self, app_id, version_id, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Exports a LUIS application to JSON format.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: LuisApp or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.LuisApp or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.export.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str')}
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
            deserialized = self._deserialize('LuisApp', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    export.metadata = {'url': '/apps/{appId}/versions/{versionId}/export'}

    def import_method(self, app_id, luis_app, version_id=None, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Imports a new version into a LUIS application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param luis_app: A LUIS application structure.\n        :type luis_app:\n         ~azure.cognitiveservices.language.luis.authoring.models.LuisApp\n        :param version_id: The new versionId to import. If not specified, the\n         versionId will be read from the imported object.\n        :type version_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: str or ClientRawResponse if raw=true\n        :rtype: str or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.import_method.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if version_id is not None:
            query_parameters['versionId'] = self._serialize.query('version_id', version_id, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(luis_app, 'LuisApp')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [201]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 201:
            deserialized = self._deserialize('str', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    import_method.metadata = {'url': '/apps/{appId}/versions/import'}

    def delete_unlabelled_utterance(self, app_id, version_id, utterance, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Deleted an unlabelled utterance in a version of the application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param utterance: The utterance text to delete.\n        :type utterance: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: OperationStatus or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.OperationStatus\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.delete_unlabelled_utterance.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(utterance, 'str')
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
    delete_unlabelled_utterance.metadata = {'url': '/apps/{appId}/versions/{versionId}/suggest'}

    def import_v2_app(self, app_id, luis_app_v2, version_id=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Imports a new version into a LUIS application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param luis_app_v2: A LUIS application structure.\n        :type luis_app_v2:\n         ~azure.cognitiveservices.language.luis.authoring.models.LuisAppV2\n        :param version_id: The new versionId to import. If not specified, the\n         versionId will be read from the imported object.\n        :type version_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: str or ClientRawResponse if raw=true\n        :rtype: str or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.import_v2_app.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if version_id is not None:
            query_parameters['versionId'] = self._serialize.query('version_id', version_id, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(luis_app_v2, 'LuisAppV2')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [201]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 201:
            deserialized = self._deserialize('str', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    import_v2_app.metadata = {'url': '/apps/{appId}/versions/import'}

    def import_lu_format(self, app_id, luis_app_lu, version_id=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Imports a new version into a LUIS application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param luis_app_lu: An LU representing the LUIS application structure.\n        :type luis_app_lu: str\n        :param version_id: The new versionId to import. If not specified, the\n         versionId will be read from the imported object.\n        :type version_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: str or ClientRawResponse if raw=true\n        :rtype: str or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.import_lu_format.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if version_id is not None:
            query_parameters['versionId'] = self._serialize.query('version_id', version_id, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'text/plain'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(luis_app_lu, 'str')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [201]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 201:
            deserialized = self._deserialize('str', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    import_lu_format.metadata = {'url': '/apps/{appId}/versions/import'}

    def export_lu_format(self, app_id, version_id, custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            i = 10
            return i + 15
        'Exports a LUIS application to text format.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: object or ClientRawResponse if raw=true\n        :rtype: Generator or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`HttpOperationError<msrest.exceptions.HttpOperationError>`\n        '
        url = self.export_lu_format.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['format'] = self._serialize.query('self.format', self.format, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/octet-stream'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=True, **operation_config)
        if response.status_code not in [200]:
            raise HttpOperationError(self._deserialize, response)
        deserialized = self._client.stream_download(response, callback)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    export_lu_format.metadata = {'url': '/apps/{appId}/versions/{versionId}/export'}