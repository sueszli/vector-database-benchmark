import uuid
from msrest.pipeline import ClientRawResponse
from .. import models

class AccountOperations(object):
    """AccountOperations operations.

    You should not instantiate directly this class, but create a Client instance that will create it for you and attach it as attribute.

    :param client: Client for service requests.
    :param config: Configuration of service client.
    :param serializer: An object model serializer.
    :param deserializer: An object model deserializer.
    :ivar api_version: The API version to use for the request. Constant value: "2023-05-01.17.0".
    """
    models = models

    def __init__(self, client, config, serializer, deserializer):
        if False:
            i = 10
            return i + 15
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self.api_version = '2023-05-01.17.0'
        self.config = config

    def list_supported_images(self, account_list_supported_images_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Lists all Virtual Machine Images supported by the Azure Batch service.\n\n        :param account_list_supported_images_options: Additional parameters\n         for the operation\n        :type account_list_supported_images_options:\n         ~azure.batch.models.AccountListSupportedImagesOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: An iterator like instance of ImageInformation\n        :rtype:\n         ~azure.batch.models.ImageInformationPaged[~azure.batch.models.ImageInformation]\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        filter = None
        if account_list_supported_images_options is not None:
            filter = account_list_supported_images_options.filter
        max_results = None
        if account_list_supported_images_options is not None:
            max_results = account_list_supported_images_options.max_results
        timeout = None
        if account_list_supported_images_options is not None:
            timeout = account_list_supported_images_options.timeout
        client_request_id = None
        if account_list_supported_images_options is not None:
            client_request_id = account_list_supported_images_options.client_request_id
        return_client_request_id = None
        if account_list_supported_images_options is not None:
            return_client_request_id = account_list_supported_images_options.return_client_request_id
        ocp_date = None
        if account_list_supported_images_options is not None:
            ocp_date = account_list_supported_images_options.ocp_date

        def prepare_request(next_link=None):
            if False:
                i = 10
                return i + 15
            if not next_link:
                url = self.list_supported_images.metadata['url']
                path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True)}
                url = self._client.format_url(url, **path_format_arguments)
                query_parameters = {}
                query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
                if filter is not None:
                    query_parameters['$filter'] = self._serialize.query('filter', filter, 'str')
                if max_results is not None:
                    query_parameters['maxresults'] = self._serialize.query('max_results', max_results, 'int', maximum=1000, minimum=1)
                if timeout is not None:
                    query_parameters['timeout'] = self._serialize.query('timeout', timeout, 'int')
            else:
                url = next_link
                query_parameters = {}
            header_parameters = {}
            header_parameters['Accept'] = 'application/json'
            if self.config.generate_client_request_id:
                header_parameters['client-request-id'] = str(uuid.uuid1())
            if custom_headers:
                header_parameters.update(custom_headers)
            if self.config.accept_language is not None:
                header_parameters['accept-language'] = self._serialize.header('self.config.accept_language', self.config.accept_language, 'str')
            if client_request_id is not None:
                header_parameters['client-request-id'] = self._serialize.header('client_request_id', client_request_id, 'str')
            if return_client_request_id is not None:
                header_parameters['return-client-request-id'] = self._serialize.header('return_client_request_id', return_client_request_id, 'bool')
            if ocp_date is not None:
                header_parameters['ocp-date'] = self._serialize.header('ocp_date', ocp_date, 'rfc-1123')
            request = self._client.get(url, query_parameters, header_parameters)
            return request

        def internal_paging(next_link=None):
            if False:
                print('Hello World!')
            request = prepare_request(next_link)
            response = self._client.send(request, stream=False, **operation_config)
            if response.status_code not in [200]:
                raise models.BatchErrorException(self._deserialize, response)
            return response
        header_dict = None
        if raw:
            header_dict = {}
        deserialized = models.ImageInformationPaged(internal_paging, self._deserialize.dependencies, header_dict)
        return deserialized
    list_supported_images.metadata = {'url': '/supportedimages'}

    def list_pool_node_counts(self, account_list_pool_node_counts_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Gets the number of Compute Nodes in each state, grouped by Pool. Note\n        that the numbers returned may not always be up to date. If you need\n        exact node counts, use a list query.\n\n        :param account_list_pool_node_counts_options: Additional parameters\n         for the operation\n        :type account_list_pool_node_counts_options:\n         ~azure.batch.models.AccountListPoolNodeCountsOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: An iterator like instance of PoolNodeCounts\n        :rtype:\n         ~azure.batch.models.PoolNodeCountsPaged[~azure.batch.models.PoolNodeCounts]\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        filter = None
        if account_list_pool_node_counts_options is not None:
            filter = account_list_pool_node_counts_options.filter
        max_results = None
        if account_list_pool_node_counts_options is not None:
            max_results = account_list_pool_node_counts_options.max_results
        timeout = None
        if account_list_pool_node_counts_options is not None:
            timeout = account_list_pool_node_counts_options.timeout
        client_request_id = None
        if account_list_pool_node_counts_options is not None:
            client_request_id = account_list_pool_node_counts_options.client_request_id
        return_client_request_id = None
        if account_list_pool_node_counts_options is not None:
            return_client_request_id = account_list_pool_node_counts_options.return_client_request_id
        ocp_date = None
        if account_list_pool_node_counts_options is not None:
            ocp_date = account_list_pool_node_counts_options.ocp_date

        def prepare_request(next_link=None):
            if False:
                i = 10
                return i + 15
            if not next_link:
                url = self.list_pool_node_counts.metadata['url']
                path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True)}
                url = self._client.format_url(url, **path_format_arguments)
                query_parameters = {}
                query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
                if filter is not None:
                    query_parameters['$filter'] = self._serialize.query('filter', filter, 'str')
                if max_results is not None:
                    query_parameters['maxresults'] = self._serialize.query('max_results', max_results, 'int', maximum=10, minimum=1)
                if timeout is not None:
                    query_parameters['timeout'] = self._serialize.query('timeout', timeout, 'int')
            else:
                url = next_link
                query_parameters = {}
            header_parameters = {}
            header_parameters['Accept'] = 'application/json'
            if self.config.generate_client_request_id:
                header_parameters['client-request-id'] = str(uuid.uuid1())
            if custom_headers:
                header_parameters.update(custom_headers)
            if self.config.accept_language is not None:
                header_parameters['accept-language'] = self._serialize.header('self.config.accept_language', self.config.accept_language, 'str')
            if client_request_id is not None:
                header_parameters['client-request-id'] = self._serialize.header('client_request_id', client_request_id, 'str')
            if return_client_request_id is not None:
                header_parameters['return-client-request-id'] = self._serialize.header('return_client_request_id', return_client_request_id, 'bool')
            if ocp_date is not None:
                header_parameters['ocp-date'] = self._serialize.header('ocp_date', ocp_date, 'rfc-1123')
            request = self._client.get(url, query_parameters, header_parameters)
            return request

        def internal_paging(next_link=None):
            if False:
                print('Hello World!')
            request = prepare_request(next_link)
            response = self._client.send(request, stream=False, **operation_config)
            if response.status_code not in [200]:
                raise models.BatchErrorException(self._deserialize, response)
            return response
        header_dict = None
        if raw:
            header_dict = {}
        deserialized = models.PoolNodeCountsPaged(internal_paging, self._deserialize.dependencies, header_dict)
        return deserialized
    list_pool_node_counts.metadata = {'url': '/nodecounts'}