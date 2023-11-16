from msrest.pipeline import ClientRawResponse
from msrestazure.azure_exceptions import CloudError
from msrestazure.azure_operation import AzureOperationPoller
import uuid
from .. import models

class DatabaseAccountsOperations(object):
    """DatabaseAccountsOperations operations.

    :param client: Client for service requests.
    :param config: Configuration of service client.
    :param serializer: An object model serializer.
    :param deserializer: An objec model deserializer.
    :ivar api_version: Version of the API to be used with the client request. The current version is 2015-04-08. Constant value: "2015-04-08".
    """

    def __init__(self, client, config, serializer, deserializer):
        if False:
            return 10
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self.api_version = '2015-04-08'
        self.config = config

    def get(self, resource_group_name, account_name, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Retrieves the properties of an existing Azure DocumentDB database\n        account.\n\n        :param resource_group_name: Name of an Azure resource group.\n        :type resource_group_name: str\n        :param account_name: DocumentDB database account name.\n        :type account_name: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :rtype: :class:`DatabaseAccount\n         <azure.mgmt.documentdb.models.DatabaseAccount>`\n        :rtype: :class:`ClientRawResponse<msrest.pipeline.ClientRawResponse>`\n         if raw=true\n        :raises: :class:`CloudError<msrestazure.azure_exceptions.CloudError>`\n        '
        url = '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}'
        path_format_arguments = {'subscriptionId': self._serialize.url('self.config.subscription_id', self.config.subscription_id, 'str'), 'resourceGroupName': self._serialize.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'accountName': self._serialize.url('account_name', account_name, 'str', max_length=50, min_length=3)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if self.config.generate_client_request_id:
            header_parameters['x-ms-client-request-id'] = str(uuid.uuid1())
        if custom_headers:
            header_parameters.update(custom_headers)
        if self.config.accept_language is not None:
            header_parameters['accept-language'] = self._serialize.header('self.config.accept_language', self.config.accept_language, 'str')
        request = self._client.get(url, query_parameters)
        response = self._client.send(request, header_parameters, **operation_config)
        if response.status_code not in [200]:
            exp = CloudError(response)
            exp.request_id = response.headers.get('x-ms-request-id')
            raise exp
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('DatabaseAccount', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized

    def patch(self, resource_group_name, account_name, tags, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Patches the properties of an existing Azure DocumentDB database\n        account.\n\n        :param resource_group_name: Name of an Azure resource group.\n        :type resource_group_name: str\n        :param account_name: DocumentDB database account name.\n        :type account_name: str\n        :param tags:\n        :type tags: dict\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :rtype:\n         :class:`AzureOperationPoller<msrestazure.azure_operation.AzureOperationPoller>`\n         instance that returns :class:`DatabaseAccount\n         <azure.mgmt.documentdb.models.DatabaseAccount>`\n        :rtype: :class:`ClientRawResponse<msrest.pipeline.ClientRawResponse>`\n         if raw=true\n        :raises: :class:`CloudError<msrestazure.azure_exceptions.CloudError>`\n        '
        update_parameters = models.DatabaseAccountPatchParameters(tags=tags)
        url = '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}'
        path_format_arguments = {'subscriptionId': self._serialize.url('self.config.subscription_id', self.config.subscription_id, 'str'), 'resourceGroupName': self._serialize.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'accountName': self._serialize.url('account_name', account_name, 'str', max_length=50, min_length=3)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if self.config.generate_client_request_id:
            header_parameters['x-ms-client-request-id'] = str(uuid.uuid1())
        if custom_headers:
            header_parameters.update(custom_headers)
        if self.config.accept_language is not None:
            header_parameters['accept-language'] = self._serialize.header('self.config.accept_language', self.config.accept_language, 'str')
        body_content = self._serialize.body(update_parameters, 'DatabaseAccountPatchParameters')

        def long_running_send():
            if False:
                for i in range(10):
                    print('nop')
            request = self._client.patch(url, query_parameters)
            return self._client.send(request, header_parameters, body_content, **operation_config)

        def get_long_running_status(status_link, headers=None):
            if False:
                while True:
                    i = 10
            request = self._client.get(status_link)
            if headers:
                request.headers.update(headers)
            return self._client.send(request, header_parameters, **operation_config)

        def get_long_running_output(response):
            if False:
                return 10
            if response.status_code not in [200]:
                exp = CloudError(response)
                exp.request_id = response.headers.get('x-ms-request-id')
                raise exp
            deserialized = None
            if response.status_code == 200:
                deserialized = self._deserialize('DatabaseAccount', response)
            if raw:
                client_raw_response = ClientRawResponse(deserialized, response)
                return client_raw_response
            return deserialized
        if raw:
            response = long_running_send()
            return get_long_running_output(response)
        long_running_operation_timeout = operation_config.get('long_running_operation_timeout', self.config.long_running_operation_timeout)
        return AzureOperationPoller(long_running_send, get_long_running_output, get_long_running_status, long_running_operation_timeout)

    def create_or_update(self, resource_group_name, account_name, create_update_parameters, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Creates or updates an Azure DocumentDB database account.\n\n        :param resource_group_name: Name of an Azure resource group.\n        :type resource_group_name: str\n        :param account_name: DocumentDB database account name.\n        :type account_name: str\n        :param create_update_parameters: The parameters to provide for the\n         current database account.\n        :type create_update_parameters:\n         :class:`DatabaseAccountCreateUpdateParameters\n         <azure.mgmt.documentdb.models.DatabaseAccountCreateUpdateParameters>`\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :rtype:\n         :class:`AzureOperationPoller<msrestazure.azure_operation.AzureOperationPoller>`\n         instance that returns :class:`DatabaseAccount\n         <azure.mgmt.documentdb.models.DatabaseAccount>`\n        :rtype: :class:`ClientRawResponse<msrest.pipeline.ClientRawResponse>`\n         if raw=true\n        :raises: :class:`CloudError<msrestazure.azure_exceptions.CloudError>`\n        '
        url = '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}'
        path_format_arguments = {'subscriptionId': self._serialize.url('self.config.subscription_id', self.config.subscription_id, 'str'), 'resourceGroupName': self._serialize.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'accountName': self._serialize.url('account_name', account_name, 'str', max_length=50, min_length=3)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if self.config.generate_client_request_id:
            header_parameters['x-ms-client-request-id'] = str(uuid.uuid1())
        if custom_headers:
            header_parameters.update(custom_headers)
        if self.config.accept_language is not None:
            header_parameters['accept-language'] = self._serialize.header('self.config.accept_language', self.config.accept_language, 'str')
        body_content = self._serialize.body(create_update_parameters, 'DatabaseAccountCreateUpdateParameters')

        def long_running_send():
            if False:
                i = 10
                return i + 15
            request = self._client.put(url, query_parameters)
            return self._client.send(request, header_parameters, body_content, **operation_config)

        def get_long_running_status(status_link, headers=None):
            if False:
                return 10
            request = self._client.get(status_link)
            if headers:
                request.headers.update(headers)
            return self._client.send(request, header_parameters, **operation_config)

        def get_long_running_output(response):
            if False:
                i = 10
                return i + 15
            if response.status_code not in [200]:
                exp = CloudError(response)
                exp.request_id = response.headers.get('x-ms-request-id')
                raise exp
            deserialized = None
            if response.status_code == 200:
                deserialized = self._deserialize('DatabaseAccount', response)
            if raw:
                client_raw_response = ClientRawResponse(deserialized, response)
                return client_raw_response
            return deserialized
        if raw:
            response = long_running_send()
            return get_long_running_output(response)
        long_running_operation_timeout = operation_config.get('long_running_operation_timeout', self.config.long_running_operation_timeout)
        return AzureOperationPoller(long_running_send, get_long_running_output, get_long_running_status, long_running_operation_timeout)

    def delete(self, resource_group_name, account_name, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Deletes an existing Azure DocumentDB database account.\n\n        :param resource_group_name: Name of an Azure resource group.\n        :type resource_group_name: str\n        :param account_name: DocumentDB database account name.\n        :type account_name: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :rtype:\n         :class:`AzureOperationPoller<msrestazure.azure_operation.AzureOperationPoller>`\n         instance that returns None\n        :rtype: :class:`ClientRawResponse<msrest.pipeline.ClientRawResponse>`\n         if raw=true\n        :raises: :class:`CloudError<msrestazure.azure_exceptions.CloudError>`\n        '
        url = '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}'
        path_format_arguments = {'subscriptionId': self._serialize.url('self.config.subscription_id', self.config.subscription_id, 'str'), 'resourceGroupName': self._serialize.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'accountName': self._serialize.url('account_name', account_name, 'str', max_length=50, min_length=3)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if self.config.generate_client_request_id:
            header_parameters['x-ms-client-request-id'] = str(uuid.uuid1())
        if custom_headers:
            header_parameters.update(custom_headers)
        if self.config.accept_language is not None:
            header_parameters['accept-language'] = self._serialize.header('self.config.accept_language', self.config.accept_language, 'str')

        def long_running_send():
            if False:
                while True:
                    i = 10
            request = self._client.delete(url, query_parameters)
            return self._client.send(request, header_parameters, **operation_config)

        def get_long_running_status(status_link, headers=None):
            if False:
                i = 10
                return i + 15
            request = self._client.get(status_link)
            if headers:
                request.headers.update(headers)
            return self._client.send(request, header_parameters, **operation_config)

        def get_long_running_output(response):
            if False:
                return 10
            if response.status_code not in [202, 204]:
                exp = CloudError(response)
                exp.request_id = response.headers.get('x-ms-request-id')
                raise exp
            if raw:
                client_raw_response = ClientRawResponse(None, response)
                return client_raw_response
        if raw:
            response = long_running_send()
            return get_long_running_output(response)
        long_running_operation_timeout = operation_config.get('long_running_operation_timeout', self.config.long_running_operation_timeout)
        return AzureOperationPoller(long_running_send, get_long_running_output, get_long_running_status, long_running_operation_timeout)

    def failover_priority_change(self, resource_group_name, account_name, failover_policies=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Changes the failover priority for the Azure DocumentDB database\n        account. A failover priority of 0 indicates a write region. The maximum\n        value for a failover priority = (total number of regions - 1). Failover\n        priority values must be unique for each of the regions in which the\n        database account exists.\n\n        :param resource_group_name: Name of an Azure resource group.\n        :type resource_group_name: str\n        :param account_name: DocumentDB database account name.\n        :type account_name: str\n        :param failover_policies: List of failover policies.\n        :type failover_policies: list of :class:`FailoverPolicy\n         <azure.mgmt.documentdb.models.FailoverPolicy>`\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :rtype:\n         :class:`AzureOperationPoller<msrestazure.azure_operation.AzureOperationPoller>`\n         instance that returns None\n        :rtype: :class:`ClientRawResponse<msrest.pipeline.ClientRawResponse>`\n         if raw=true\n        :raises: :class:`CloudError<msrestazure.azure_exceptions.CloudError>`\n        '
        failover_parameters = models.FailoverPolicies(failover_policies=failover_policies)
        url = '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/failoverPriorityChange'
        path_format_arguments = {'subscriptionId': self._serialize.url('self.config.subscription_id', self.config.subscription_id, 'str'), 'resourceGroupName': self._serialize.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'accountName': self._serialize.url('account_name', account_name, 'str', max_length=50, min_length=3)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if self.config.generate_client_request_id:
            header_parameters['x-ms-client-request-id'] = str(uuid.uuid1())
        if custom_headers:
            header_parameters.update(custom_headers)
        if self.config.accept_language is not None:
            header_parameters['accept-language'] = self._serialize.header('self.config.accept_language', self.config.accept_language, 'str')
        body_content = self._serialize.body(failover_parameters, 'FailoverPolicies')

        def long_running_send():
            if False:
                for i in range(10):
                    print('nop')
            request = self._client.post(url, query_parameters)
            return self._client.send(request, header_parameters, body_content, **operation_config)

        def get_long_running_status(status_link, headers=None):
            if False:
                while True:
                    i = 10
            request = self._client.get(status_link)
            if headers:
                request.headers.update(headers)
            return self._client.send(request, header_parameters, **operation_config)

        def get_long_running_output(response):
            if False:
                for i in range(10):
                    print('nop')
            if response.status_code not in [202, 204]:
                exp = CloudError(response)
                exp.request_id = response.headers.get('x-ms-request-id')
                raise exp
            if raw:
                client_raw_response = ClientRawResponse(None, response)
                return client_raw_response
        if raw:
            response = long_running_send()
            return get_long_running_output(response)
        long_running_operation_timeout = operation_config.get('long_running_operation_timeout', self.config.long_running_operation_timeout)
        return AzureOperationPoller(long_running_send, get_long_running_output, get_long_running_status, long_running_operation_timeout)

    def list(self, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Lists all the Azure DocumentDB database accounts available under the\n        subscription.\n\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :rtype: :class:`DatabaseAccountPaged\n         <azure.mgmt.documentdb.models.DatabaseAccountPaged>`\n        :raises: :class:`CloudError<msrestazure.azure_exceptions.CloudError>`\n        '

        def internal_paging(next_link=None, raw=False):
            if False:
                for i in range(10):
                    print('nop')
            if not next_link:
                url = '/subscriptions/{subscriptionId}/providers/Microsoft.DocumentDB/databaseAccounts'
                path_format_arguments = {'subscriptionId': self._serialize.url('self.config.subscription_id', self.config.subscription_id, 'str')}
                url = self._client.format_url(url, **path_format_arguments)
                query_parameters = {}
                query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
            else:
                url = next_link
                query_parameters = {}
            header_parameters = {}
            header_parameters['Content-Type'] = 'application/json; charset=utf-8'
            if self.config.generate_client_request_id:
                header_parameters['x-ms-client-request-id'] = str(uuid.uuid1())
            if custom_headers:
                header_parameters.update(custom_headers)
            if self.config.accept_language is not None:
                header_parameters['accept-language'] = self._serialize.header('self.config.accept_language', self.config.accept_language, 'str')
            request = self._client.get(url, query_parameters)
            response = self._client.send(request, header_parameters, **operation_config)
            if response.status_code not in [200]:
                exp = CloudError(response)
                exp.request_id = response.headers.get('x-ms-request-id')
                raise exp
            return response
        deserialized = models.DatabaseAccountPaged(internal_paging, self._deserialize.dependencies)
        if raw:
            header_dict = {}
            client_raw_response = models.DatabaseAccountPaged(internal_paging, self._deserialize.dependencies, header_dict)
            return client_raw_response
        return deserialized

    def list_by_resource_group(self, resource_group_name, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Lists all the Azure DocumentDB database accounts available under the\n        given resource group.\n\n        :param resource_group_name: Name of an Azure resource group.\n        :type resource_group_name: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :rtype: :class:`DatabaseAccountPaged\n         <azure.mgmt.documentdb.models.DatabaseAccountPaged>`\n        :raises: :class:`CloudError<msrestazure.azure_exceptions.CloudError>`\n        '

        def internal_paging(next_link=None, raw=False):
            if False:
                for i in range(10):
                    print('nop')
            if not next_link:
                url = '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts'
                path_format_arguments = {'resourceGroupName': self._serialize.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'subscriptionId': self._serialize.url('self.config.subscription_id', self.config.subscription_id, 'str')}
                url = self._client.format_url(url, **path_format_arguments)
                query_parameters = {}
                query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
            else:
                url = next_link
                query_parameters = {}
            header_parameters = {}
            header_parameters['Content-Type'] = 'application/json; charset=utf-8'
            if self.config.generate_client_request_id:
                header_parameters['x-ms-client-request-id'] = str(uuid.uuid1())
            if custom_headers:
                header_parameters.update(custom_headers)
            if self.config.accept_language is not None:
                header_parameters['accept-language'] = self._serialize.header('self.config.accept_language', self.config.accept_language, 'str')
            request = self._client.get(url, query_parameters)
            response = self._client.send(request, header_parameters, **operation_config)
            if response.status_code not in [200]:
                exp = CloudError(response)
                exp.request_id = response.headers.get('x-ms-request-id')
                raise exp
            return response
        deserialized = models.DatabaseAccountPaged(internal_paging, self._deserialize.dependencies)
        if raw:
            header_dict = {}
            client_raw_response = models.DatabaseAccountPaged(internal_paging, self._deserialize.dependencies, header_dict)
            return client_raw_response
        return deserialized

    def list_keys(self, resource_group_name, account_name, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Lists the access keys for the specified Azure DocumentDB database\n        account.\n\n        :param resource_group_name: Name of an Azure resource group.\n        :type resource_group_name: str\n        :param account_name: DocumentDB database account name.\n        :type account_name: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :rtype: :class:`DatabaseAccountListKeysResult\n         <azure.mgmt.documentdb.models.DatabaseAccountListKeysResult>`\n        :rtype: :class:`ClientRawResponse<msrest.pipeline.ClientRawResponse>`\n         if raw=true\n        :raises: :class:`CloudError<msrestazure.azure_exceptions.CloudError>`\n        '
        url = '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/listKeys'
        path_format_arguments = {'subscriptionId': self._serialize.url('self.config.subscription_id', self.config.subscription_id, 'str'), 'resourceGroupName': self._serialize.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'accountName': self._serialize.url('account_name', account_name, 'str', max_length=50, min_length=3)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if self.config.generate_client_request_id:
            header_parameters['x-ms-client-request-id'] = str(uuid.uuid1())
        if custom_headers:
            header_parameters.update(custom_headers)
        if self.config.accept_language is not None:
            header_parameters['accept-language'] = self._serialize.header('self.config.accept_language', self.config.accept_language, 'str')
        request = self._client.post(url, query_parameters)
        response = self._client.send(request, header_parameters, **operation_config)
        if response.status_code not in [200]:
            exp = CloudError(response)
            exp.request_id = response.headers.get('x-ms-request-id')
            raise exp
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('DatabaseAccountListKeysResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized

    def list_connection_strings(self, resource_group_name, account_name, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Lists the connection strings for the specified Azure DocumentDB\n        database account.\n\n        :param resource_group_name: Name of an Azure resource group.\n        :type resource_group_name: str\n        :param account_name: DocumentDB database account name.\n        :type account_name: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :rtype: :class:`DatabaseAccountListConnectionStringsResult\n         <azure.mgmt.documentdb.models.DatabaseAccountListConnectionStringsResult>`\n        :rtype: :class:`ClientRawResponse<msrest.pipeline.ClientRawResponse>`\n         if raw=true\n        :raises: :class:`CloudError<msrestazure.azure_exceptions.CloudError>`\n        '
        url = '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/listConnectionStrings'
        path_format_arguments = {'subscriptionId': self._serialize.url('self.config.subscription_id', self.config.subscription_id, 'str'), 'resourceGroupName': self._serialize.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'accountName': self._serialize.url('account_name', account_name, 'str', max_length=50, min_length=3)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if self.config.generate_client_request_id:
            header_parameters['x-ms-client-request-id'] = str(uuid.uuid1())
        if custom_headers:
            header_parameters.update(custom_headers)
        if self.config.accept_language is not None:
            header_parameters['accept-language'] = self._serialize.header('self.config.accept_language', self.config.accept_language, 'str')
        request = self._client.post(url, query_parameters)
        response = self._client.send(request, header_parameters, **operation_config)
        if response.status_code not in [200]:
            exp = CloudError(response)
            exp.request_id = response.headers.get('x-ms-request-id')
            raise exp
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('DatabaseAccountListConnectionStringsResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized

    def list_read_only_keys(self, resource_group_name, account_name, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Lists the read-only access keys for the specified Azure DocumentDB\n        database account.\n\n        :param resource_group_name: Name of an Azure resource group.\n        :type resource_group_name: str\n        :param account_name: DocumentDB database account name.\n        :type account_name: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :rtype: :class:`DatabaseAccountListReadOnlyKeysResult\n         <azure.mgmt.documentdb.models.DatabaseAccountListReadOnlyKeysResult>`\n        :rtype: :class:`ClientRawResponse<msrest.pipeline.ClientRawResponse>`\n         if raw=true\n        :raises: :class:`CloudError<msrestazure.azure_exceptions.CloudError>`\n        '
        url = '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/readonlykeys'
        path_format_arguments = {'subscriptionId': self._serialize.url('self.config.subscription_id', self.config.subscription_id, 'str'), 'resourceGroupName': self._serialize.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'accountName': self._serialize.url('account_name', account_name, 'str', max_length=50, min_length=3)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if self.config.generate_client_request_id:
            header_parameters['x-ms-client-request-id'] = str(uuid.uuid1())
        if custom_headers:
            header_parameters.update(custom_headers)
        if self.config.accept_language is not None:
            header_parameters['accept-language'] = self._serialize.header('self.config.accept_language', self.config.accept_language, 'str')
        request = self._client.get(url, query_parameters)
        response = self._client.send(request, header_parameters, **operation_config)
        if response.status_code not in [200]:
            exp = CloudError(response)
            exp.request_id = response.headers.get('x-ms-request-id')
            raise exp
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('DatabaseAccountListReadOnlyKeysResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized

    def regenerate_key(self, resource_group_name, account_name, key_kind, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        "Regenerates an access key for the specified Azure DocumentDB database\n        account.\n\n        :param resource_group_name: Name of an Azure resource group.\n        :type resource_group_name: str\n        :param account_name: DocumentDB database account name.\n        :type account_name: str\n        :param key_kind: The access key to regenerate. Possible values\n         include: 'primary', 'secondary', 'primaryReadonly',\n         'secondaryReadonly'\n        :type key_kind: str or :class:`KeyKind\n         <azure.mgmt.documentdb.models.KeyKind>`\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :rtype:\n         :class:`AzureOperationPoller<msrestazure.azure_operation.AzureOperationPoller>`\n         instance that returns None\n        :rtype: :class:`ClientRawResponse<msrest.pipeline.ClientRawResponse>`\n         if raw=true\n        :raises: :class:`CloudError<msrestazure.azure_exceptions.CloudError>`\n        "
        key_to_regenerate = models.DatabaseAccountRegenerateKeyParameters(key_kind=key_kind)
        url = '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DocumentDB/databaseAccounts/{accountName}/regenerateKey'
        path_format_arguments = {'subscriptionId': self._serialize.url('self.config.subscription_id', self.config.subscription_id, 'str'), 'resourceGroupName': self._serialize.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'accountName': self._serialize.url('account_name', account_name, 'str', max_length=50, min_length=3)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if self.config.generate_client_request_id:
            header_parameters['x-ms-client-request-id'] = str(uuid.uuid1())
        if custom_headers:
            header_parameters.update(custom_headers)
        if self.config.accept_language is not None:
            header_parameters['accept-language'] = self._serialize.header('self.config.accept_language', self.config.accept_language, 'str')
        body_content = self._serialize.body(key_to_regenerate, 'DatabaseAccountRegenerateKeyParameters')

        def long_running_send():
            if False:
                print('Hello World!')
            request = self._client.post(url, query_parameters)
            return self._client.send(request, header_parameters, body_content, **operation_config)

        def get_long_running_status(status_link, headers=None):
            if False:
                return 10
            request = self._client.get(status_link)
            if headers:
                request.headers.update(headers)
            return self._client.send(request, header_parameters, **operation_config)

        def get_long_running_output(response):
            if False:
                for i in range(10):
                    print('nop')
            if response.status_code not in [200, 202]:
                exp = CloudError(response)
                exp.request_id = response.headers.get('x-ms-request-id')
                raise exp
            if raw:
                client_raw_response = ClientRawResponse(None, response)
                return client_raw_response
        if raw:
            response = long_running_send()
            return get_long_running_output(response)
        long_running_operation_timeout = operation_config.get('long_running_operation_timeout', self.config.long_running_operation_timeout)
        return AzureOperationPoller(long_running_send, get_long_running_output, get_long_running_status, long_running_operation_timeout)

    def check_name_exists(self, account_name, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        "Checks that the Azure DocumentDB account name already exists. A valid\n        account name may contain only lowercase letters, numbers, and the '-'\n        character, and must be between 3 and 50 characters.\n\n        :param account_name: DocumentDB database account name.\n        :type account_name: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :rtype: bool\n        :rtype: :class:`ClientRawResponse<msrest.pipeline.ClientRawResponse>`\n         if raw=true\n        :raises: :class:`CloudError<msrestazure.azure_exceptions.CloudError>`\n        "
        url = '/providers/Microsoft.DocumentDB/databaseAccountNames/{accountName}'
        path_format_arguments = {'accountName': self._serialize.url('account_name', account_name, 'str', max_length=50, min_length=3)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if self.config.generate_client_request_id:
            header_parameters['x-ms-client-request-id'] = str(uuid.uuid1())
        if custom_headers:
            header_parameters.update(custom_headers)
        if self.config.accept_language is not None:
            header_parameters['accept-language'] = self._serialize.header('self.config.accept_language', self.config.accept_language, 'str')
        request = self._client.head(url, query_parameters)
        response = self._client.send(request, header_parameters, **operation_config)
        if response.status_code not in [200, 404]:
            exp = CloudError(response)
            exp.request_id = response.headers.get('x-ms-request-id')
            raise exp
        deserialized = response.status_code == 200
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized