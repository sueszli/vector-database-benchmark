from msrest.pipeline import ClientRawResponse
from msrest.exceptions import HttpOperationError
from .. import models

class AppsOperations(object):
    """AppsOperations operations.

    You should not instantiate directly this class, but create a Client instance that will create it for you and attach it as attribute.

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

    def add(self, application_create_object, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Creates a new LUIS app.\n\n        :param application_create_object: An application containing Name,\n         Description (optional), Culture, Usage Scenario (optional), Domain\n         (optional) and initial version ID (optional) of the application.\n         Default value for the version ID is "0.1". Note: the culture cannot be\n         changed after the app is created.\n        :type application_create_object:\n         ~azure.cognitiveservices.language.luis.authoring.models.ApplicationCreateObject\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: str or ClientRawResponse if raw=true\n        :rtype: str or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.add.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(application_create_object, 'ApplicationCreateObject')
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
    add.metadata = {'url': '/apps/'}

    def list(self, skip=0, take=100, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        "Lists all of the user's applications.\n\n        :param skip: The number of entries to skip. Default value is 0.\n        :type skip: int\n        :param take: The number of entries to return. Maximum page size is\n         500. Default is 100.\n        :type take: int\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.language.luis.authoring.models.ApplicationInfoResponse]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        "
        url = self.list.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
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
            deserialized = self._deserialize('[ApplicationInfoResponse]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    list.metadata = {'url': '/apps/'}

    def import_method(self, luis_app, app_name=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        "Imports an application to LUIS, the application's structure is included\n        in the request body.\n\n        :param luis_app: A LUIS application structure.\n        :type luis_app:\n         ~azure.cognitiveservices.language.luis.authoring.models.LuisApp\n        :param app_name: The application name to create. If not specified, the\n         application name will be read from the imported object. If the\n         application name already exists, an error is returned.\n        :type app_name: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: str or ClientRawResponse if raw=true\n        :rtype: str or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        "
        url = self.import_method.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if app_name is not None:
            query_parameters['appName'] = self._serialize.query('app_name', app_name, 'str')
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
    import_method.metadata = {'url': '/apps/import'}

    def list_cortana_endpoints(self, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Gets the endpoint URLs for the prebuilt Cortana applications.\n\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: PersonalAssistantsResponse or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.PersonalAssistantsResponse\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.list_cortana_endpoints.metadata['url']
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
            deserialized = self._deserialize('PersonalAssistantsResponse', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    list_cortana_endpoints.metadata = {'url': '/apps/assistants'}

    def list_domains(self, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Gets the available application domains.\n\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype: list[str] or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.list_domains.metadata['url']
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
            deserialized = self._deserialize('[str]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    list_domains.metadata = {'url': '/apps/domains'}

    def list_usage_scenarios(self, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Gets the application available usage scenarios.\n\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype: list[str] or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.list_usage_scenarios.metadata['url']
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
            deserialized = self._deserialize('[str]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    list_usage_scenarios.metadata = {'url': '/apps/usagescenarios'}

    def list_supported_cultures(self, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Gets a list of supported cultures. Cultures are equivalent to the\n        written language and locale. For example,"en-us" represents the U.S.\n        variation of English.\n\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.language.luis.authoring.models.AvailableCulture]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.list_supported_cultures.metadata['url']
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
            deserialized = self._deserialize('[AvailableCulture]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    list_supported_cultures.metadata = {'url': '/apps/cultures'}

    def download_query_logs(self, app_id, custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            print('Hello World!')
        "Gets the logs of the past month's endpoint queries for the application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: object or ClientRawResponse if raw=true\n        :rtype: Generator or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`HttpOperationError<msrest.exceptions.HttpOperationError>`\n        "
        url = self.download_query_logs.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
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
    download_query_logs.metadata = {'url': '/apps/{appId}/querylogs'}

    def get(self, app_id, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Gets the application info.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ApplicationInfoResponse or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.ApplicationInfoResponse\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.get.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str')}
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
            deserialized = self._deserialize('ApplicationInfoResponse', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get.metadata = {'url': '/apps/{appId}'}

    def update(self, app_id, name=None, description=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        "Updates the name or description of the application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param name: The application's new name.\n        :type name: str\n        :param description: The application's new description.\n        :type description: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: OperationStatus or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.OperationStatus\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        "
        application_update_object = models.ApplicationUpdateObject(name=name, description=description)
        url = self.update.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(application_update_object, 'ApplicationUpdateObject')
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
    update.metadata = {'url': '/apps/{appId}'}

    def delete(self, app_id, force=False, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Deletes an application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param force: A flag to indicate whether to force an operation.\n        :type force: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: OperationStatus or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.OperationStatus\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.delete.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if force is not None:
            query_parameters['force'] = self._serialize.query('force', force, 'bool')
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
    delete.metadata = {'url': '/apps/{appId}'}

    def publish(self, app_id, version_id=None, is_staging=False, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Publishes a specific version of the application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID to publish.\n        :type version_id: str\n        :param is_staging: Indicates if the staging slot should be used,\n         instead of the Production one.\n        :type is_staging: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ProductionOrStagingEndpointInfo or ClientRawResponse if\n         raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.ProductionOrStagingEndpointInfo\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        application_publish_object = models.ApplicationPublishObject(version_id=version_id, is_staging=is_staging)
        url = self.publish.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(application_publish_object, 'ApplicationPublishObject')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [201, 207]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 201:
            deserialized = self._deserialize('ProductionOrStagingEndpointInfo', response)
        if response.status_code == 207:
            deserialized = self._deserialize('ProductionOrStagingEndpointInfo', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    publish.metadata = {'url': '/apps/{appId}/publish'}

    def get_settings(self, app_id, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        "Get the application settings including 'UseAllTrainingData'.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ApplicationSettings or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.ApplicationSettings\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        "
        url = self.get_settings.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str')}
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
            deserialized = self._deserialize('ApplicationSettings', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_settings.metadata = {'url': '/apps/{appId}/settings'}

    def update_settings(self, app_id, is_public=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        "Updates the application settings including 'UseAllTrainingData'.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param is_public: Setting your application as public allows other\n         people to use your application's endpoint using their own keys.\n        :type is_public: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: OperationStatus or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.OperationStatus\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        "
        application_setting_update_object = models.ApplicationSettingUpdateObject(is_public=is_public)
        url = self.update_settings.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(application_setting_update_object, 'ApplicationSettingUpdateObject')
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
    update_settings.metadata = {'url': '/apps/{appId}/settings'}

    def get_publish_settings(self, app_id, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        "Get the application publish settings including 'UseAllTrainingData'.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: PublishSettings or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.PublishSettings\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        "
        url = self.get_publish_settings.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str')}
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
            deserialized = self._deserialize('PublishSettings', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_publish_settings.metadata = {'url': '/apps/{appId}/publishsettings'}

    def update_publish_settings(self, app_id, publish_setting_update_object, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        "Updates the application publish settings including\n        'UseAllTrainingData'.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param publish_setting_update_object: An object containing the new\n         publish application settings.\n        :type publish_setting_update_object:\n         ~azure.cognitiveservices.language.luis.authoring.models.PublishSettingUpdateObject\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: OperationStatus or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.OperationStatus\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        "
        url = self.update_publish_settings.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(publish_setting_update_object, 'PublishSettingUpdateObject')
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
    update_publish_settings.metadata = {'url': '/apps/{appId}/publishsettings'}

    def list_endpoints(self, app_id, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Returns the available endpoint deployment regions and URLs.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: dict or ClientRawResponse if raw=true\n        :rtype: dict[str, str] or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.list_endpoints.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str')}
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
            deserialized = self._deserialize('{str}', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    list_endpoints.metadata = {'url': '/apps/{appId}/endpoints'}

    def list_available_custom_prebuilt_domains(self, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Gets all the available custom prebuilt domains for all cultures.\n\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.language.luis.authoring.models.PrebuiltDomain]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.list_available_custom_prebuilt_domains.metadata['url']
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
            deserialized = self._deserialize('[PrebuiltDomain]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    list_available_custom_prebuilt_domains.metadata = {'url': '/apps/customprebuiltdomains'}

    def add_custom_prebuilt_domain(self, domain_name=None, culture=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Adds a prebuilt domain along with its intent and entity models as a new\n        application.\n\n        :param domain_name: The domain name.\n        :type domain_name: str\n        :param culture: The culture of the new domain.\n        :type culture: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: str or ClientRawResponse if raw=true\n        :rtype: str or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        prebuilt_domain_create_object = models.PrebuiltDomainCreateObject(domain_name=domain_name, culture=culture)
        url = self.add_custom_prebuilt_domain.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(prebuilt_domain_create_object, 'PrebuiltDomainCreateObject')
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
    add_custom_prebuilt_domain.metadata = {'url': '/apps/customprebuiltdomains'}

    def list_available_custom_prebuilt_domains_for_culture(self, culture, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Gets all the available prebuilt domains for a specific culture.\n\n        :param culture: Culture.\n        :type culture: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.language.luis.authoring.models.PrebuiltDomain]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.list_available_custom_prebuilt_domains_for_culture.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'culture': self._serialize.url('culture', culture, 'str')}
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
            deserialized = self._deserialize('[PrebuiltDomain]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    list_available_custom_prebuilt_domains_for_culture.metadata = {'url': '/apps/customprebuiltdomains/{culture}'}

    def package_published_application_as_gzip(self, app_id, slot_name, custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'package - Gets published LUIS application package in binary stream GZip\n        format.\n\n        Packages a published LUIS application as a GZip file to be used in the\n        LUIS container.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param slot_name: The publishing slot name.\n        :type slot_name: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: object or ClientRawResponse if raw=true\n        :rtype: Generator or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.package_published_application_as_gzip.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'slotName': self._serialize.url('slot_name', slot_name, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=True, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = self._client.stream_download(response, callback)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    package_published_application_as_gzip.metadata = {'url': '/package/{appId}/slot/{slotName}/gzip'}

    def package_trained_application_as_gzip(self, app_id, version_id, custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            while True:
                i = 10
        'package - Gets trained LUIS application package in binary stream GZip\n        format.\n\n        Packages trained LUIS application as GZip file to be used in the LUIS\n        container.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The version ID.\n        :type version_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: object or ClientRawResponse if raw=true\n        :rtype: Generator or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        '
        url = self.package_trained_application_as_gzip.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=True, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = self._client.stream_download(response, callback)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    package_trained_application_as_gzip.metadata = {'url': '/package/{appId}/versions/{versionId}/gzip'}

    def import_v2_app(self, luis_app_v2, app_name=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        "Imports an application to LUIS, the application's structure is included\n        in the request body.\n\n        :param luis_app_v2: A LUIS application structure.\n        :type luis_app_v2:\n         ~azure.cognitiveservices.language.luis.authoring.models.LuisAppV2\n        :param app_name: The application name to create. If not specified, the\n         application name will be read from the imported object. If the\n         application name already exists, an error is returned.\n        :type app_name: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: str or ClientRawResponse if raw=true\n        :rtype: str or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        "
        url = self.import_v2_app.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if app_name is not None:
            query_parameters['appName'] = self._serialize.query('app_name', app_name, 'str')
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
    import_v2_app.metadata = {'url': '/apps/import'}

    def import_lu_format(self, luis_app_lu, app_name=None, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        "Imports an application to LUIS, the application's structure is included\n        in the request body.\n\n        :param luis_app_lu: A LUIS application structure.\n        :type luis_app_lu: str\n        :param app_name: The application name to create. If not specified, the\n         application name will be read from the imported object. If the\n         application name already exists, an error is returned.\n        :type app_name: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: str or ClientRawResponse if raw=true\n        :rtype: str or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        "
        url = self.import_lu_format.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if app_name is not None:
            query_parameters['appName'] = self._serialize.query('app_name', app_name, 'str')
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
    import_lu_format.metadata = {'url': '/apps/import'}