from msrest.pipeline import ClientRawResponse
from .. import models

class AzureAccountsOperations(object):
    """AzureAccountsOperations operations.

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

    def assign_to_app(self, app_id, arm_token=None, azure_account_info_object=None, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        "apps - Assign a LUIS Azure account to an application.\n\n        Assigns an Azure account to the application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param arm_token: The custom arm token header to use; containing the\n         user's ARM token used to validate azure accounts information.\n        :type arm_token: str\n        :param azure_account_info_object: The Azure account information\n         object.\n        :type azure_account_info_object:\n         ~azure.cognitiveservices.language.luis.authoring.models.AzureAccountInfoObject\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: OperationStatus or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.OperationStatus\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        "
        url = self.assign_to_app.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        if arm_token is not None:
            header_parameters['ArmToken'] = self._serialize.header('arm_token', arm_token, 'str')
        if azure_account_info_object is not None:
            body_content = self._serialize.body(azure_account_info_object, 'AzureAccountInfoObject')
        else:
            body_content = None
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [201]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 201:
            deserialized = self._deserialize('OperationStatus', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    assign_to_app.metadata = {'url': '/apps/{appId}/azureaccounts'}

    def get_assigned(self, app_id, arm_token=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        "apps - Get LUIS Azure accounts assigned to the application.\n\n        Gets the LUIS Azure accounts assigned to the application for the user\n        using his ARM token.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param arm_token: The custom arm token header to use; containing the\n         user's ARM token used to validate azure accounts information.\n        :type arm_token: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.language.luis.authoring.models.AzureAccountInfoObject]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        "
        url = self.get_assigned.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        if arm_token is not None:
            header_parameters['ArmToken'] = self._serialize.header('arm_token', arm_token, 'str')
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('[AzureAccountInfoObject]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_assigned.metadata = {'url': '/apps/{appId}/azureaccounts'}

    def remove_from_app(self, app_id, arm_token=None, azure_account_info_object=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        "apps - Removes an assigned LUIS Azure account from an application.\n\n        Removes assigned Azure account from the application.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param arm_token: The custom arm token header to use; containing the\n         user's ARM token used to validate azure accounts information.\n        :type arm_token: str\n        :param azure_account_info_object: The Azure account information\n         object.\n        :type azure_account_info_object:\n         ~azure.cognitiveservices.language.luis.authoring.models.AzureAccountInfoObject\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: OperationStatus or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.authoring.models.OperationStatus\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        "
        url = self.remove_from_app.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        if arm_token is not None:
            header_parameters['ArmToken'] = self._serialize.header('arm_token', arm_token, 'str')
        if azure_account_info_object is not None:
            body_content = self._serialize.body(azure_account_info_object, 'AzureAccountInfoObject')
        else:
            body_content = None
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
    remove_from_app.metadata = {'url': '/apps/{appId}/azureaccounts'}

    def list_user_luis_accounts(self, arm_token=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        "user - Get LUIS Azure accounts.\n\n        Gets the LUIS Azure accounts for the user using his ARM token.\n\n        :param arm_token: The custom arm token header to use; containing the\n         user's ARM token used to validate azure accounts information.\n        :type arm_token: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.language.luis.authoring.models.AzureAccountInfoObject]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.luis.authoring.models.ErrorResponseException>`\n        "
        url = self.list_user_luis_accounts.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        if arm_token is not None:
            header_parameters['ArmToken'] = self._serialize.header('arm_token', arm_token, 'str')
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('[AzureAccountInfoObject]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    list_user_luis_accounts.metadata = {'url': '/azureaccounts'}