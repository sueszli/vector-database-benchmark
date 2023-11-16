import uuid
from msrest.pipeline import ClientRawResponse
from .. import models

class ApplicationOperations(object):
    """ApplicationOperations operations.

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
            print('Hello World!')
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self.api_version = '2023-05-01.17.0'
        self.config = config

    def list(self, application_list_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Lists all of the applications available in the specified Account.\n\n        This operation returns only Applications and versions that are\n        available for use on Compute Nodes; that is, that can be used in an\n        Package reference. For administrator information about applications and\n        versions that are not yet available to Compute Nodes, use the Azure\n        portal or the Azure Resource Manager API.\n\n        :param application_list_options: Additional parameters for the\n         operation\n        :type application_list_options:\n         ~azure.batch.models.ApplicationListOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: An iterator like instance of ApplicationSummary\n        :rtype:\n         ~azure.batch.models.ApplicationSummaryPaged[~azure.batch.models.ApplicationSummary]\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        max_results = None
        if application_list_options is not None:
            max_results = application_list_options.max_results
        timeout = None
        if application_list_options is not None:
            timeout = application_list_options.timeout
        client_request_id = None
        if application_list_options is not None:
            client_request_id = application_list_options.client_request_id
        return_client_request_id = None
        if application_list_options is not None:
            return_client_request_id = application_list_options.return_client_request_id
        ocp_date = None
        if application_list_options is not None:
            ocp_date = application_list_options.ocp_date

        def prepare_request(next_link=None):
            if False:
                while True:
                    i = 10
            if not next_link:
                url = self.list.metadata['url']
                path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True)}
                url = self._client.format_url(url, **path_format_arguments)
                query_parameters = {}
                query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
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
                for i in range(10):
                    print('nop')
            request = prepare_request(next_link)
            response = self._client.send(request, stream=False, **operation_config)
            if response.status_code not in [200]:
                raise models.BatchErrorException(self._deserialize, response)
            return response
        header_dict = None
        if raw:
            header_dict = {}
        deserialized = models.ApplicationSummaryPaged(internal_paging, self._deserialize.dependencies, header_dict)
        return deserialized
    list.metadata = {'url': '/applications'}

    def get(self, application_id, application_get_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Gets information about the specified Application.\n\n        This operation returns only Applications and versions that are\n        available for use on Compute Nodes; that is, that can be used in an\n        Package reference. For administrator information about Applications and\n        versions that are not yet available to Compute Nodes, use the Azure\n        portal or the Azure Resource Manager API.\n\n        :param application_id: The ID of the Application.\n        :type application_id: str\n        :param application_get_options: Additional parameters for the\n         operation\n        :type application_get_options:\n         ~azure.batch.models.ApplicationGetOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ApplicationSummary or ClientRawResponse if raw=true\n        :rtype: ~azure.batch.models.ApplicationSummary or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if application_get_options is not None:
            timeout = application_get_options.timeout
        client_request_id = None
        if application_get_options is not None:
            client_request_id = application_get_options.client_request_id
        return_client_request_id = None
        if application_get_options is not None:
            return_client_request_id = application_get_options.return_client_request_id
        ocp_date = None
        if application_get_options is not None:
            ocp_date = application_get_options.ocp_date
        url = self.get.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'applicationId': self._serialize.url('application_id', application_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
        if timeout is not None:
            query_parameters['timeout'] = self._serialize.query('timeout', timeout, 'int')
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
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.BatchErrorException(self._deserialize, response)
        header_dict = {}
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ApplicationSummary', response)
            header_dict = {'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123'}
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            client_raw_response.add_headers(header_dict)
            return client_raw_response
        return deserialized
    get.metadata = {'url': '/applications/{applicationId}'}