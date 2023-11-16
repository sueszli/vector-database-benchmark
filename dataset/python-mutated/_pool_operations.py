import uuid
from msrest.pipeline import ClientRawResponse
from .. import models

class PoolOperations(object):
    """PoolOperations operations.

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
            while True:
                i = 10
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self.api_version = '2023-05-01.17.0'
        self.config = config

    def list_usage_metrics(self, pool_list_usage_metrics_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Lists the usage metrics, aggregated by Pool across individual time\n        intervals, for the specified Account.\n\n        If you do not specify a $filter clause including a poolId, the response\n        includes all Pools that existed in the Account in the time range of the\n        returned aggregation intervals. If you do not specify a $filter clause\n        including a startTime or endTime these filters default to the start and\n        end times of the last aggregation interval currently available; that\n        is, only the last aggregation interval is returned.\n\n        :param pool_list_usage_metrics_options: Additional parameters for the\n         operation\n        :type pool_list_usage_metrics_options:\n         ~azure.batch.models.PoolListUsageMetricsOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: An iterator like instance of PoolUsageMetrics\n        :rtype:\n         ~azure.batch.models.PoolUsageMetricsPaged[~azure.batch.models.PoolUsageMetrics]\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        start_time = None
        if pool_list_usage_metrics_options is not None:
            start_time = pool_list_usage_metrics_options.start_time
        end_time = None
        if pool_list_usage_metrics_options is not None:
            end_time = pool_list_usage_metrics_options.end_time
        filter = None
        if pool_list_usage_metrics_options is not None:
            filter = pool_list_usage_metrics_options.filter
        max_results = None
        if pool_list_usage_metrics_options is not None:
            max_results = pool_list_usage_metrics_options.max_results
        timeout = None
        if pool_list_usage_metrics_options is not None:
            timeout = pool_list_usage_metrics_options.timeout
        client_request_id = None
        if pool_list_usage_metrics_options is not None:
            client_request_id = pool_list_usage_metrics_options.client_request_id
        return_client_request_id = None
        if pool_list_usage_metrics_options is not None:
            return_client_request_id = pool_list_usage_metrics_options.return_client_request_id
        ocp_date = None
        if pool_list_usage_metrics_options is not None:
            ocp_date = pool_list_usage_metrics_options.ocp_date

        def prepare_request(next_link=None):
            if False:
                return 10
            if not next_link:
                url = self.list_usage_metrics.metadata['url']
                path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True)}
                url = self._client.format_url(url, **path_format_arguments)
                query_parameters = {}
                query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
                if start_time is not None:
                    query_parameters['starttime'] = self._serialize.query('start_time', start_time, 'iso-8601')
                if end_time is not None:
                    query_parameters['endtime'] = self._serialize.query('end_time', end_time, 'iso-8601')
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
                return 10
            request = prepare_request(next_link)
            response = self._client.send(request, stream=False, **operation_config)
            if response.status_code not in [200]:
                raise models.BatchErrorException(self._deserialize, response)
            return response
        header_dict = None
        if raw:
            header_dict = {}
        deserialized = models.PoolUsageMetricsPaged(internal_paging, self._deserialize.dependencies, header_dict)
        return deserialized
    list_usage_metrics.metadata = {'url': '/poolusagemetrics'}

    def add(self, pool, pool_add_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Adds a Pool to the specified Account.\n\n        When naming Pools, avoid including sensitive information such as user\n        names or secret project names. This information may appear in telemetry\n        logs accessible to Microsoft Support engineers.\n\n        :param pool: The Pool to be added.\n        :type pool: ~azure.batch.models.PoolAddParameter\n        :param pool_add_options: Additional parameters for the operation\n        :type pool_add_options: ~azure.batch.models.PoolAddOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if pool_add_options is not None:
            timeout = pool_add_options.timeout
        client_request_id = None
        if pool_add_options is not None:
            client_request_id = pool_add_options.client_request_id
        return_client_request_id = None
        if pool_add_options is not None:
            return_client_request_id = pool_add_options.return_client_request_id
        ocp_date = None
        if pool_add_options is not None:
            ocp_date = pool_add_options.ocp_date
        url = self.add.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
        if timeout is not None:
            query_parameters['timeout'] = self._serialize.query('timeout', timeout, 'int')
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; odata=minimalmetadata; charset=utf-8'
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
        body_content = self._serialize.body(pool, 'PoolAddParameter')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [201]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123', 'DataServiceId': 'str'})
            return client_raw_response
    add.metadata = {'url': '/pools'}

    def list(self, pool_list_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Lists all of the Pools in the specified Account.\n\n        :param pool_list_options: Additional parameters for the operation\n        :type pool_list_options: ~azure.batch.models.PoolListOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: An iterator like instance of CloudPool\n        :rtype:\n         ~azure.batch.models.CloudPoolPaged[~azure.batch.models.CloudPool]\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        filter = None
        if pool_list_options is not None:
            filter = pool_list_options.filter
        select = None
        if pool_list_options is not None:
            select = pool_list_options.select
        expand = None
        if pool_list_options is not None:
            expand = pool_list_options.expand
        max_results = None
        if pool_list_options is not None:
            max_results = pool_list_options.max_results
        timeout = None
        if pool_list_options is not None:
            timeout = pool_list_options.timeout
        client_request_id = None
        if pool_list_options is not None:
            client_request_id = pool_list_options.client_request_id
        return_client_request_id = None
        if pool_list_options is not None:
            return_client_request_id = pool_list_options.return_client_request_id
        ocp_date = None
        if pool_list_options is not None:
            ocp_date = pool_list_options.ocp_date

        def prepare_request(next_link=None):
            if False:
                print('Hello World!')
            if not next_link:
                url = self.list.metadata['url']
                path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True)}
                url = self._client.format_url(url, **path_format_arguments)
                query_parameters = {}
                query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
                if filter is not None:
                    query_parameters['$filter'] = self._serialize.query('filter', filter, 'str')
                if select is not None:
                    query_parameters['$select'] = self._serialize.query('select', select, 'str')
                if expand is not None:
                    query_parameters['$expand'] = self._serialize.query('expand', expand, 'str')
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
        deserialized = models.CloudPoolPaged(internal_paging, self._deserialize.dependencies, header_dict)
        return deserialized
    list.metadata = {'url': '/pools'}

    def delete(self, pool_id, pool_delete_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Deletes a Pool from the specified Account.\n\n        When you request that a Pool be deleted, the following actions occur:\n        the Pool state is set to deleting; any ongoing resize operation on the\n        Pool are stopped; the Batch service starts resizing the Pool to zero\n        Compute Nodes; any Tasks running on existing Compute Nodes are\n        terminated and requeued (as if a resize Pool operation had been\n        requested with the default requeue option); finally, the Pool is\n        removed from the system. Because running Tasks are requeued, the user\n        can rerun these Tasks by updating their Job to target a different Pool.\n        The Tasks can then run on the new Pool. If you want to override the\n        requeue behavior, then you should call resize Pool explicitly to shrink\n        the Pool to zero size before deleting the Pool. If you call an Update,\n        Patch or Delete API on a Pool in the deleting state, it will fail with\n        HTTP status code 409 with error code PoolBeingDeleted.\n\n        :param pool_id: The ID of the Pool to delete.\n        :type pool_id: str\n        :param pool_delete_options: Additional parameters for the operation\n        :type pool_delete_options: ~azure.batch.models.PoolDeleteOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if pool_delete_options is not None:
            timeout = pool_delete_options.timeout
        client_request_id = None
        if pool_delete_options is not None:
            client_request_id = pool_delete_options.client_request_id
        return_client_request_id = None
        if pool_delete_options is not None:
            return_client_request_id = pool_delete_options.return_client_request_id
        ocp_date = None
        if pool_delete_options is not None:
            ocp_date = pool_delete_options.ocp_date
        if_match = None
        if pool_delete_options is not None:
            if_match = pool_delete_options.if_match
        if_none_match = None
        if pool_delete_options is not None:
            if_none_match = pool_delete_options.if_none_match
        if_modified_since = None
        if pool_delete_options is not None:
            if_modified_since = pool_delete_options.if_modified_since
        if_unmodified_since = None
        if pool_delete_options is not None:
            if_unmodified_since = pool_delete_options.if_unmodified_since
        url = self.delete.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
        if timeout is not None:
            query_parameters['timeout'] = self._serialize.query('timeout', timeout, 'int')
        header_parameters = {}
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
        if if_match is not None:
            header_parameters['If-Match'] = self._serialize.header('if_match', if_match, 'str')
        if if_none_match is not None:
            header_parameters['If-None-Match'] = self._serialize.header('if_none_match', if_none_match, 'str')
        if if_modified_since is not None:
            header_parameters['If-Modified-Since'] = self._serialize.header('if_modified_since', if_modified_since, 'rfc-1123')
        if if_unmodified_since is not None:
            header_parameters['If-Unmodified-Since'] = self._serialize.header('if_unmodified_since', if_unmodified_since, 'rfc-1123')
        request = self._client.delete(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [202]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str'})
            return client_raw_response
    delete.metadata = {'url': '/pools/{poolId}'}

    def exists(self, pool_id, pool_exists_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Gets basic properties of a Pool.\n\n        :param pool_id: The ID of the Pool to get.\n        :type pool_id: str\n        :param pool_exists_options: Additional parameters for the operation\n        :type pool_exists_options: ~azure.batch.models.PoolExistsOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: bool or ClientRawResponse if raw=true\n        :rtype: bool or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if pool_exists_options is not None:
            timeout = pool_exists_options.timeout
        client_request_id = None
        if pool_exists_options is not None:
            client_request_id = pool_exists_options.client_request_id
        return_client_request_id = None
        if pool_exists_options is not None:
            return_client_request_id = pool_exists_options.return_client_request_id
        ocp_date = None
        if pool_exists_options is not None:
            ocp_date = pool_exists_options.ocp_date
        if_match = None
        if pool_exists_options is not None:
            if_match = pool_exists_options.if_match
        if_none_match = None
        if pool_exists_options is not None:
            if_none_match = pool_exists_options.if_none_match
        if_modified_since = None
        if pool_exists_options is not None:
            if_modified_since = pool_exists_options.if_modified_since
        if_unmodified_since = None
        if pool_exists_options is not None:
            if_unmodified_since = pool_exists_options.if_unmodified_since
        url = self.exists.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
        if timeout is not None:
            query_parameters['timeout'] = self._serialize.query('timeout', timeout, 'int')
        header_parameters = {}
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
        if if_match is not None:
            header_parameters['If-Match'] = self._serialize.header('if_match', if_match, 'str')
        if if_none_match is not None:
            header_parameters['If-None-Match'] = self._serialize.header('if_none_match', if_none_match, 'str')
        if if_modified_since is not None:
            header_parameters['If-Modified-Since'] = self._serialize.header('if_modified_since', if_modified_since, 'rfc-1123')
        if if_unmodified_since is not None:
            header_parameters['If-Unmodified-Since'] = self._serialize.header('if_unmodified_since', if_unmodified_since, 'rfc-1123')
        request = self._client.head(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200, 404]:
            raise models.BatchErrorException(self._deserialize, response)
        deserialized = response.status_code == 200
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123'})
            return client_raw_response
        return deserialized
    exists.metadata = {'url': '/pools/{poolId}'}

    def get(self, pool_id, pool_get_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Gets information about the specified Pool.\n\n        :param pool_id: The ID of the Pool to get.\n        :type pool_id: str\n        :param pool_get_options: Additional parameters for the operation\n        :type pool_get_options: ~azure.batch.models.PoolGetOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: CloudPool or ClientRawResponse if raw=true\n        :rtype: ~azure.batch.models.CloudPool or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        select = None
        if pool_get_options is not None:
            select = pool_get_options.select
        expand = None
        if pool_get_options is not None:
            expand = pool_get_options.expand
        timeout = None
        if pool_get_options is not None:
            timeout = pool_get_options.timeout
        client_request_id = None
        if pool_get_options is not None:
            client_request_id = pool_get_options.client_request_id
        return_client_request_id = None
        if pool_get_options is not None:
            return_client_request_id = pool_get_options.return_client_request_id
        ocp_date = None
        if pool_get_options is not None:
            ocp_date = pool_get_options.ocp_date
        if_match = None
        if pool_get_options is not None:
            if_match = pool_get_options.if_match
        if_none_match = None
        if pool_get_options is not None:
            if_none_match = pool_get_options.if_none_match
        if_modified_since = None
        if pool_get_options is not None:
            if_modified_since = pool_get_options.if_modified_since
        if_unmodified_since = None
        if pool_get_options is not None:
            if_unmodified_since = pool_get_options.if_unmodified_since
        url = self.get.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
        if select is not None:
            query_parameters['$select'] = self._serialize.query('select', select, 'str')
        if expand is not None:
            query_parameters['$expand'] = self._serialize.query('expand', expand, 'str')
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
        if if_match is not None:
            header_parameters['If-Match'] = self._serialize.header('if_match', if_match, 'str')
        if if_none_match is not None:
            header_parameters['If-None-Match'] = self._serialize.header('if_none_match', if_none_match, 'str')
        if if_modified_since is not None:
            header_parameters['If-Modified-Since'] = self._serialize.header('if_modified_since', if_modified_since, 'rfc-1123')
        if if_unmodified_since is not None:
            header_parameters['If-Unmodified-Since'] = self._serialize.header('if_unmodified_since', if_unmodified_since, 'rfc-1123')
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.BatchErrorException(self._deserialize, response)
        header_dict = {}
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('CloudPool', response)
            header_dict = {'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123'}
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            client_raw_response.add_headers(header_dict)
            return client_raw_response
        return deserialized
    get.metadata = {'url': '/pools/{poolId}'}

    def patch(self, pool_id, pool_patch_parameter, pool_patch_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Updates the properties of the specified Pool.\n\n        This only replaces the Pool properties specified in the request. For\n        example, if the Pool has a StartTask associated with it, and a request\n        does not specify a StartTask element, then the Pool keeps the existing\n        StartTask.\n\n        :param pool_id: The ID of the Pool to update.\n        :type pool_id: str\n        :param pool_patch_parameter: The parameters for the request.\n        :type pool_patch_parameter: ~azure.batch.models.PoolPatchParameter\n        :param pool_patch_options: Additional parameters for the operation\n        :type pool_patch_options: ~azure.batch.models.PoolPatchOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if pool_patch_options is not None:
            timeout = pool_patch_options.timeout
        client_request_id = None
        if pool_patch_options is not None:
            client_request_id = pool_patch_options.client_request_id
        return_client_request_id = None
        if pool_patch_options is not None:
            return_client_request_id = pool_patch_options.return_client_request_id
        ocp_date = None
        if pool_patch_options is not None:
            ocp_date = pool_patch_options.ocp_date
        if_match = None
        if pool_patch_options is not None:
            if_match = pool_patch_options.if_match
        if_none_match = None
        if pool_patch_options is not None:
            if_none_match = pool_patch_options.if_none_match
        if_modified_since = None
        if pool_patch_options is not None:
            if_modified_since = pool_patch_options.if_modified_since
        if_unmodified_since = None
        if pool_patch_options is not None:
            if_unmodified_since = pool_patch_options.if_unmodified_since
        url = self.patch.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
        if timeout is not None:
            query_parameters['timeout'] = self._serialize.query('timeout', timeout, 'int')
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; odata=minimalmetadata; charset=utf-8'
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
        if if_match is not None:
            header_parameters['If-Match'] = self._serialize.header('if_match', if_match, 'str')
        if if_none_match is not None:
            header_parameters['If-None-Match'] = self._serialize.header('if_none_match', if_none_match, 'str')
        if if_modified_since is not None:
            header_parameters['If-Modified-Since'] = self._serialize.header('if_modified_since', if_modified_since, 'rfc-1123')
        if if_unmodified_since is not None:
            header_parameters['If-Unmodified-Since'] = self._serialize.header('if_unmodified_since', if_unmodified_since, 'rfc-1123')
        body_content = self._serialize.body(pool_patch_parameter, 'PoolPatchParameter')
        request = self._client.patch(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123', 'DataServiceId': 'str'})
            return client_raw_response
    patch.metadata = {'url': '/pools/{poolId}'}

    def disable_auto_scale(self, pool_id, pool_disable_auto_scale_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Disables automatic scaling for a Pool.\n\n        :param pool_id: The ID of the Pool on which to disable automatic\n         scaling.\n        :type pool_id: str\n        :param pool_disable_auto_scale_options: Additional parameters for the\n         operation\n        :type pool_disable_auto_scale_options:\n         ~azure.batch.models.PoolDisableAutoScaleOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if pool_disable_auto_scale_options is not None:
            timeout = pool_disable_auto_scale_options.timeout
        client_request_id = None
        if pool_disable_auto_scale_options is not None:
            client_request_id = pool_disable_auto_scale_options.client_request_id
        return_client_request_id = None
        if pool_disable_auto_scale_options is not None:
            return_client_request_id = pool_disable_auto_scale_options.return_client_request_id
        ocp_date = None
        if pool_disable_auto_scale_options is not None:
            ocp_date = pool_disable_auto_scale_options.ocp_date
        url = self.disable_auto_scale.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
        if timeout is not None:
            query_parameters['timeout'] = self._serialize.query('timeout', timeout, 'int')
        header_parameters = {}
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
        request = self._client.post(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123', 'DataServiceId': 'str'})
            return client_raw_response
    disable_auto_scale.metadata = {'url': '/pools/{poolId}/disableautoscale'}

    def enable_auto_scale(self, pool_id, auto_scale_formula=None, auto_scale_evaluation_interval=None, pool_enable_auto_scale_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Enables automatic scaling for a Pool.\n\n        You cannot enable automatic scaling on a Pool if a resize operation is\n        in progress on the Pool. If automatic scaling of the Pool is currently\n        disabled, you must specify a valid autoscale formula as part of the\n        request. If automatic scaling of the Pool is already enabled, you may\n        specify a new autoscale formula and/or a new evaluation interval. You\n        cannot call this API for the same Pool more than once every 30 seconds.\n\n        :param pool_id: The ID of the Pool on which to enable automatic\n         scaling.\n        :type pool_id: str\n        :param auto_scale_formula: The formula is checked for validity before\n         it is applied to the Pool. If the formula is not valid, the Batch\n         service rejects the request with detailed error information. For more\n         information about specifying this formula, see Automatically scale\n         Compute Nodes in an Azure Batch Pool\n         (https://azure.microsoft.com/en-us/documentation/articles/batch-automatic-scaling).\n        :type auto_scale_formula: str\n        :param auto_scale_evaluation_interval: The default value is 15\n         minutes. The minimum and maximum value are 5 minutes and 168 hours\n         respectively. If you specify a value less than 5 minutes or greater\n         than 168 hours, the Batch service rejects the request with an invalid\n         property value error; if you are calling the REST API directly, the\n         HTTP status code is 400 (Bad Request). If you specify a new interval,\n         then the existing autoscale evaluation schedule will be stopped and a\n         new autoscale evaluation schedule will be started, with its starting\n         time being the time when this request was issued.\n        :type auto_scale_evaluation_interval: timedelta\n        :param pool_enable_auto_scale_options: Additional parameters for the\n         operation\n        :type pool_enable_auto_scale_options:\n         ~azure.batch.models.PoolEnableAutoScaleOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if pool_enable_auto_scale_options is not None:
            timeout = pool_enable_auto_scale_options.timeout
        client_request_id = None
        if pool_enable_auto_scale_options is not None:
            client_request_id = pool_enable_auto_scale_options.client_request_id
        return_client_request_id = None
        if pool_enable_auto_scale_options is not None:
            return_client_request_id = pool_enable_auto_scale_options.return_client_request_id
        ocp_date = None
        if pool_enable_auto_scale_options is not None:
            ocp_date = pool_enable_auto_scale_options.ocp_date
        if_match = None
        if pool_enable_auto_scale_options is not None:
            if_match = pool_enable_auto_scale_options.if_match
        if_none_match = None
        if pool_enable_auto_scale_options is not None:
            if_none_match = pool_enable_auto_scale_options.if_none_match
        if_modified_since = None
        if pool_enable_auto_scale_options is not None:
            if_modified_since = pool_enable_auto_scale_options.if_modified_since
        if_unmodified_since = None
        if pool_enable_auto_scale_options is not None:
            if_unmodified_since = pool_enable_auto_scale_options.if_unmodified_since
        pool_enable_auto_scale_parameter = models.PoolEnableAutoScaleParameter(auto_scale_formula=auto_scale_formula, auto_scale_evaluation_interval=auto_scale_evaluation_interval)
        url = self.enable_auto_scale.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
        if timeout is not None:
            query_parameters['timeout'] = self._serialize.query('timeout', timeout, 'int')
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; odata=minimalmetadata; charset=utf-8'
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
        if if_match is not None:
            header_parameters['If-Match'] = self._serialize.header('if_match', if_match, 'str')
        if if_none_match is not None:
            header_parameters['If-None-Match'] = self._serialize.header('if_none_match', if_none_match, 'str')
        if if_modified_since is not None:
            header_parameters['If-Modified-Since'] = self._serialize.header('if_modified_since', if_modified_since, 'rfc-1123')
        if if_unmodified_since is not None:
            header_parameters['If-Unmodified-Since'] = self._serialize.header('if_unmodified_since', if_unmodified_since, 'rfc-1123')
        body_content = self._serialize.body(pool_enable_auto_scale_parameter, 'PoolEnableAutoScaleParameter')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123', 'DataServiceId': 'str'})
            return client_raw_response
    enable_auto_scale.metadata = {'url': '/pools/{poolId}/enableautoscale'}

    def evaluate_auto_scale(self, pool_id, auto_scale_formula, pool_evaluate_auto_scale_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        "Gets the result of evaluating an automatic scaling formula on the Pool.\n\n        This API is primarily for validating an autoscale formula, as it simply\n        returns the result without applying the formula to the Pool. The Pool\n        must have auto scaling enabled in order to evaluate a formula.\n\n        :param pool_id: The ID of the Pool on which to evaluate the automatic\n         scaling formula.\n        :type pool_id: str\n        :param auto_scale_formula: The formula is validated and its results\n         calculated, but it is not applied to the Pool. To apply the formula to\n         the Pool, 'Enable automatic scaling on a Pool'. For more information\n         about specifying this formula, see Automatically scale Compute Nodes\n         in an Azure Batch Pool\n         (https://azure.microsoft.com/en-us/documentation/articles/batch-automatic-scaling).\n        :type auto_scale_formula: str\n        :param pool_evaluate_auto_scale_options: Additional parameters for the\n         operation\n        :type pool_evaluate_auto_scale_options:\n         ~azure.batch.models.PoolEvaluateAutoScaleOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: AutoScaleRun or ClientRawResponse if raw=true\n        :rtype: ~azure.batch.models.AutoScaleRun or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        "
        timeout = None
        if pool_evaluate_auto_scale_options is not None:
            timeout = pool_evaluate_auto_scale_options.timeout
        client_request_id = None
        if pool_evaluate_auto_scale_options is not None:
            client_request_id = pool_evaluate_auto_scale_options.client_request_id
        return_client_request_id = None
        if pool_evaluate_auto_scale_options is not None:
            return_client_request_id = pool_evaluate_auto_scale_options.return_client_request_id
        ocp_date = None
        if pool_evaluate_auto_scale_options is not None:
            ocp_date = pool_evaluate_auto_scale_options.ocp_date
        pool_evaluate_auto_scale_parameter = models.PoolEvaluateAutoScaleParameter(auto_scale_formula=auto_scale_formula)
        url = self.evaluate_auto_scale.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
        if timeout is not None:
            query_parameters['timeout'] = self._serialize.query('timeout', timeout, 'int')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; odata=minimalmetadata; charset=utf-8'
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
        body_content = self._serialize.body(pool_evaluate_auto_scale_parameter, 'PoolEvaluateAutoScaleParameter')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.BatchErrorException(self._deserialize, response)
        header_dict = {}
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('AutoScaleRun', response)
            header_dict = {'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123', 'DataServiceId': 'str'}
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            client_raw_response.add_headers(header_dict)
            return client_raw_response
        return deserialized
    evaluate_auto_scale.metadata = {'url': '/pools/{poolId}/evaluateautoscale'}

    def resize(self, pool_id, pool_resize_parameter, pool_resize_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        "Changes the number of Compute Nodes that are assigned to a Pool.\n\n        You can only resize a Pool when its allocation state is steady. If the\n        Pool is already resizing, the request fails with status code 409. When\n        you resize a Pool, the Pool's allocation state changes from steady to\n        resizing. You cannot resize Pools which are configured for automatic\n        scaling. If you try to do this, the Batch service returns an error 409.\n        If you resize a Pool downwards, the Batch service chooses which Compute\n        Nodes to remove. To remove specific Compute Nodes, use the Pool remove\n        Compute Nodes API instead.\n\n        :param pool_id: The ID of the Pool to resize.\n        :type pool_id: str\n        :param pool_resize_parameter: The parameters for the request.\n        :type pool_resize_parameter: ~azure.batch.models.PoolResizeParameter\n        :param pool_resize_options: Additional parameters for the operation\n        :type pool_resize_options: ~azure.batch.models.PoolResizeOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        "
        timeout = None
        if pool_resize_options is not None:
            timeout = pool_resize_options.timeout
        client_request_id = None
        if pool_resize_options is not None:
            client_request_id = pool_resize_options.client_request_id
        return_client_request_id = None
        if pool_resize_options is not None:
            return_client_request_id = pool_resize_options.return_client_request_id
        ocp_date = None
        if pool_resize_options is not None:
            ocp_date = pool_resize_options.ocp_date
        if_match = None
        if pool_resize_options is not None:
            if_match = pool_resize_options.if_match
        if_none_match = None
        if pool_resize_options is not None:
            if_none_match = pool_resize_options.if_none_match
        if_modified_since = None
        if pool_resize_options is not None:
            if_modified_since = pool_resize_options.if_modified_since
        if_unmodified_since = None
        if pool_resize_options is not None:
            if_unmodified_since = pool_resize_options.if_unmodified_since
        url = self.resize.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
        if timeout is not None:
            query_parameters['timeout'] = self._serialize.query('timeout', timeout, 'int')
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; odata=minimalmetadata; charset=utf-8'
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
        if if_match is not None:
            header_parameters['If-Match'] = self._serialize.header('if_match', if_match, 'str')
        if if_none_match is not None:
            header_parameters['If-None-Match'] = self._serialize.header('if_none_match', if_none_match, 'str')
        if if_modified_since is not None:
            header_parameters['If-Modified-Since'] = self._serialize.header('if_modified_since', if_modified_since, 'rfc-1123')
        if if_unmodified_since is not None:
            header_parameters['If-Unmodified-Since'] = self._serialize.header('if_unmodified_since', if_unmodified_since, 'rfc-1123')
        body_content = self._serialize.body(pool_resize_parameter, 'PoolResizeParameter')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [202]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123', 'DataServiceId': 'str'})
            return client_raw_response
    resize.metadata = {'url': '/pools/{poolId}/resize'}

    def stop_resize(self, pool_id, pool_stop_resize_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Stops an ongoing resize operation on the Pool.\n\n        This does not restore the Pool to its previous state before the resize\n        operation: it only stops any further changes being made, and the Pool\n        maintains its current state. After stopping, the Pool stabilizes at the\n        number of Compute Nodes it was at when the stop operation was done.\n        During the stop operation, the Pool allocation state changes first to\n        stopping and then to steady. A resize operation need not be an explicit\n        resize Pool request; this API can also be used to halt the initial\n        sizing of the Pool when it is created.\n\n        :param pool_id: The ID of the Pool whose resizing you want to stop.\n        :type pool_id: str\n        :param pool_stop_resize_options: Additional parameters for the\n         operation\n        :type pool_stop_resize_options:\n         ~azure.batch.models.PoolStopResizeOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if pool_stop_resize_options is not None:
            timeout = pool_stop_resize_options.timeout
        client_request_id = None
        if pool_stop_resize_options is not None:
            client_request_id = pool_stop_resize_options.client_request_id
        return_client_request_id = None
        if pool_stop_resize_options is not None:
            return_client_request_id = pool_stop_resize_options.return_client_request_id
        ocp_date = None
        if pool_stop_resize_options is not None:
            ocp_date = pool_stop_resize_options.ocp_date
        if_match = None
        if pool_stop_resize_options is not None:
            if_match = pool_stop_resize_options.if_match
        if_none_match = None
        if pool_stop_resize_options is not None:
            if_none_match = pool_stop_resize_options.if_none_match
        if_modified_since = None
        if pool_stop_resize_options is not None:
            if_modified_since = pool_stop_resize_options.if_modified_since
        if_unmodified_since = None
        if pool_stop_resize_options is not None:
            if_unmodified_since = pool_stop_resize_options.if_unmodified_since
        url = self.stop_resize.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
        if timeout is not None:
            query_parameters['timeout'] = self._serialize.query('timeout', timeout, 'int')
        header_parameters = {}
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
        if if_match is not None:
            header_parameters['If-Match'] = self._serialize.header('if_match', if_match, 'str')
        if if_none_match is not None:
            header_parameters['If-None-Match'] = self._serialize.header('if_none_match', if_none_match, 'str')
        if if_modified_since is not None:
            header_parameters['If-Modified-Since'] = self._serialize.header('if_modified_since', if_modified_since, 'rfc-1123')
        if if_unmodified_since is not None:
            header_parameters['If-Unmodified-Since'] = self._serialize.header('if_unmodified_since', if_unmodified_since, 'rfc-1123')
        request = self._client.post(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [202]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123', 'DataServiceId': 'str'})
            return client_raw_response
    stop_resize.metadata = {'url': '/pools/{poolId}/stopresize'}

    def update_properties(self, pool_id, pool_update_properties_parameter, pool_update_properties_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Updates the properties of the specified Pool.\n\n        This fully replaces all the updatable properties of the Pool. For\n        example, if the Pool has a StartTask associated with it and if\n        StartTask is not specified with this request, then the Batch service\n        will remove the existing StartTask.\n\n        :param pool_id: The ID of the Pool to update.\n        :type pool_id: str\n        :param pool_update_properties_parameter: The parameters for the\n         request.\n        :type pool_update_properties_parameter:\n         ~azure.batch.models.PoolUpdatePropertiesParameter\n        :param pool_update_properties_options: Additional parameters for the\n         operation\n        :type pool_update_properties_options:\n         ~azure.batch.models.PoolUpdatePropertiesOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if pool_update_properties_options is not None:
            timeout = pool_update_properties_options.timeout
        client_request_id = None
        if pool_update_properties_options is not None:
            client_request_id = pool_update_properties_options.client_request_id
        return_client_request_id = None
        if pool_update_properties_options is not None:
            return_client_request_id = pool_update_properties_options.return_client_request_id
        ocp_date = None
        if pool_update_properties_options is not None:
            ocp_date = pool_update_properties_options.ocp_date
        url = self.update_properties.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
        if timeout is not None:
            query_parameters['timeout'] = self._serialize.query('timeout', timeout, 'int')
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; odata=minimalmetadata; charset=utf-8'
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
        body_content = self._serialize.body(pool_update_properties_parameter, 'PoolUpdatePropertiesParameter')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [204]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123', 'DataServiceId': 'str'})
            return client_raw_response
    update_properties.metadata = {'url': '/pools/{poolId}/updateproperties'}

    def remove_nodes(self, pool_id, node_remove_parameter, pool_remove_nodes_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Removes Compute Nodes from the specified Pool.\n\n        This operation can only run when the allocation state of the Pool is\n        steady. When this operation runs, the allocation state changes from\n        steady to resizing. Each request may remove up to 100 nodes.\n\n        :param pool_id: The ID of the Pool from which you want to remove\n         Compute Nodes.\n        :type pool_id: str\n        :param node_remove_parameter: The parameters for the request.\n        :type node_remove_parameter: ~azure.batch.models.NodeRemoveParameter\n        :param pool_remove_nodes_options: Additional parameters for the\n         operation\n        :type pool_remove_nodes_options:\n         ~azure.batch.models.PoolRemoveNodesOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if pool_remove_nodes_options is not None:
            timeout = pool_remove_nodes_options.timeout
        client_request_id = None
        if pool_remove_nodes_options is not None:
            client_request_id = pool_remove_nodes_options.client_request_id
        return_client_request_id = None
        if pool_remove_nodes_options is not None:
            return_client_request_id = pool_remove_nodes_options.return_client_request_id
        ocp_date = None
        if pool_remove_nodes_options is not None:
            ocp_date = pool_remove_nodes_options.ocp_date
        if_match = None
        if pool_remove_nodes_options is not None:
            if_match = pool_remove_nodes_options.if_match
        if_none_match = None
        if pool_remove_nodes_options is not None:
            if_none_match = pool_remove_nodes_options.if_none_match
        if_modified_since = None
        if pool_remove_nodes_options is not None:
            if_modified_since = pool_remove_nodes_options.if_modified_since
        if_unmodified_since = None
        if pool_remove_nodes_options is not None:
            if_unmodified_since = pool_remove_nodes_options.if_unmodified_since
        url = self.remove_nodes.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
        if timeout is not None:
            query_parameters['timeout'] = self._serialize.query('timeout', timeout, 'int')
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; odata=minimalmetadata; charset=utf-8'
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
        if if_match is not None:
            header_parameters['If-Match'] = self._serialize.header('if_match', if_match, 'str')
        if if_none_match is not None:
            header_parameters['If-None-Match'] = self._serialize.header('if_none_match', if_none_match, 'str')
        if if_modified_since is not None:
            header_parameters['If-Modified-Since'] = self._serialize.header('if_modified_since', if_modified_since, 'rfc-1123')
        if if_unmodified_since is not None:
            header_parameters['If-Unmodified-Since'] = self._serialize.header('if_unmodified_since', if_unmodified_since, 'rfc-1123')
        body_content = self._serialize.body(node_remove_parameter, 'NodeRemoveParameter')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [202]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123', 'DataServiceId': 'str'})
            return client_raw_response
    remove_nodes.metadata = {'url': '/pools/{poolId}/removenodes'}