import uuid
from msrest.pipeline import ClientRawResponse
from .. import models

class TaskOperations(object):
    """TaskOperations operations.

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

    def add(self, job_id, task, task_add_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Adds a Task to the specified Job.\n\n        The maximum lifetime of a Task from addition to completion is 180 days.\n        If a Task has not completed within 180 days of being added it will be\n        terminated by the Batch service and left in whatever state it was in at\n        that time.\n\n        :param job_id: The ID of the Job to which the Task is to be added.\n        :type job_id: str\n        :param task: The Task to be added.\n        :type task: ~azure.batch.models.TaskAddParameter\n        :param task_add_options: Additional parameters for the operation\n        :type task_add_options: ~azure.batch.models.TaskAddOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if task_add_options is not None:
            timeout = task_add_options.timeout
        client_request_id = None
        if task_add_options is not None:
            client_request_id = task_add_options.client_request_id
        return_client_request_id = None
        if task_add_options is not None:
            return_client_request_id = task_add_options.return_client_request_id
        ocp_date = None
        if task_add_options is not None:
            ocp_date = task_add_options.ocp_date
        url = self.add.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'jobId': self._serialize.url('job_id', job_id, 'str')}
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
        body_content = self._serialize.body(task, 'TaskAddParameter')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [201]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123', 'DataServiceId': 'str'})
            return client_raw_response
    add.metadata = {'url': '/jobs/{jobId}/tasks'}

    def list(self, job_id, task_list_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Lists all of the Tasks that are associated with the specified Job.\n\n        For multi-instance Tasks, information such as affinityId, executionInfo\n        and nodeInfo refer to the primary Task. Use the list subtasks API to\n        retrieve information about subtasks.\n\n        :param job_id: The ID of the Job.\n        :type job_id: str\n        :param task_list_options: Additional parameters for the operation\n        :type task_list_options: ~azure.batch.models.TaskListOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: An iterator like instance of CloudTask\n        :rtype:\n         ~azure.batch.models.CloudTaskPaged[~azure.batch.models.CloudTask]\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        filter = None
        if task_list_options is not None:
            filter = task_list_options.filter
        select = None
        if task_list_options is not None:
            select = task_list_options.select
        expand = None
        if task_list_options is not None:
            expand = task_list_options.expand
        max_results = None
        if task_list_options is not None:
            max_results = task_list_options.max_results
        timeout = None
        if task_list_options is not None:
            timeout = task_list_options.timeout
        client_request_id = None
        if task_list_options is not None:
            client_request_id = task_list_options.client_request_id
        return_client_request_id = None
        if task_list_options is not None:
            return_client_request_id = task_list_options.return_client_request_id
        ocp_date = None
        if task_list_options is not None:
            ocp_date = task_list_options.ocp_date

        def prepare_request(next_link=None):
            if False:
                print('Hello World!')
            if not next_link:
                url = self.list.metadata['url']
                path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'jobId': self._serialize.url('job_id', job_id, 'str')}
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
                print('Hello World!')
            request = prepare_request(next_link)
            response = self._client.send(request, stream=False, **operation_config)
            if response.status_code not in [200]:
                raise models.BatchErrorException(self._deserialize, response)
            return response
        header_dict = None
        if raw:
            header_dict = {}
        deserialized = models.CloudTaskPaged(internal_paging, self._deserialize.dependencies, header_dict)
        return deserialized
    list.metadata = {'url': '/jobs/{jobId}/tasks'}

    def add_collection(self, job_id, value, task_add_collection_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        "Adds a collection of Tasks to the specified Job.\n\n        Note that each Task must have a unique ID. The Batch service may not\n        return the results for each Task in the same order the Tasks were\n        submitted in this request. If the server times out or the connection is\n        closed during the request, the request may have been partially or fully\n        processed, or not at all. In such cases, the user should re-issue the\n        request. Note that it is up to the user to correctly handle failures\n        when re-issuing a request. For example, you should use the same Task\n        IDs during a retry so that if the prior operation succeeded, the retry\n        will not create extra Tasks unexpectedly. If the response contains any\n        Tasks which failed to add, a client can retry the request. In a retry,\n        it is most efficient to resubmit only Tasks that failed to add, and to\n        omit Tasks that were successfully added on the first attempt. The\n        maximum lifetime of a Task from addition to completion is 180 days. If\n        a Task has not completed within 180 days of being added it will be\n        terminated by the Batch service and left in whatever state it was in at\n        that time.\n\n        :param job_id: The ID of the Job to which the Task collection is to be\n         added.\n        :type job_id: str\n        :param value: The total serialized size of this collection must be\n         less than 1MB. If it is greater than 1MB (for example if each Task has\n         100's of resource files or environment variables), the request will\n         fail with code 'RequestBodyTooLarge' and should be retried again with\n         fewer Tasks.\n        :type value: list[~azure.batch.models.TaskAddParameter]\n        :param task_add_collection_options: Additional parameters for the\n         operation\n        :type task_add_collection_options:\n         ~azure.batch.models.TaskAddCollectionOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: TaskAddCollectionResult or ClientRawResponse if raw=true\n        :rtype: ~azure.batch.models.TaskAddCollectionResult or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        "
        timeout = None
        if task_add_collection_options is not None:
            timeout = task_add_collection_options.timeout
        client_request_id = None
        if task_add_collection_options is not None:
            client_request_id = task_add_collection_options.client_request_id
        return_client_request_id = None
        if task_add_collection_options is not None:
            return_client_request_id = task_add_collection_options.return_client_request_id
        ocp_date = None
        if task_add_collection_options is not None:
            ocp_date = task_add_collection_options.ocp_date
        task_collection = models.TaskAddCollectionParameter(value=value)
        url = self.add_collection.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'jobId': self._serialize.url('job_id', job_id, 'str')}
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
        body_content = self._serialize.body(task_collection, 'TaskAddCollectionParameter')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.BatchErrorException(self._deserialize, response)
        header_dict = {}
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('TaskAddCollectionResult', response)
            header_dict = {'client-request-id': 'str', 'request-id': 'str'}
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            client_raw_response.add_headers(header_dict)
            return client_raw_response
        return deserialized
    add_collection.metadata = {'url': '/jobs/{jobId}/addtaskcollection'}

    def delete(self, job_id, task_id, task_delete_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Deletes a Task from the specified Job.\n\n        When a Task is deleted, all of the files in its directory on the\n        Compute Node where it ran are also deleted (regardless of the retention\n        time). For multi-instance Tasks, the delete Task operation applies\n        synchronously to the primary task; subtasks and their files are then\n        deleted asynchronously in the background.\n\n        :param job_id: The ID of the Job from which to delete the Task.\n        :type job_id: str\n        :param task_id: The ID of the Task to delete.\n        :type task_id: str\n        :param task_delete_options: Additional parameters for the operation\n        :type task_delete_options: ~azure.batch.models.TaskDeleteOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if task_delete_options is not None:
            timeout = task_delete_options.timeout
        client_request_id = None
        if task_delete_options is not None:
            client_request_id = task_delete_options.client_request_id
        return_client_request_id = None
        if task_delete_options is not None:
            return_client_request_id = task_delete_options.return_client_request_id
        ocp_date = None
        if task_delete_options is not None:
            ocp_date = task_delete_options.ocp_date
        if_match = None
        if task_delete_options is not None:
            if_match = task_delete_options.if_match
        if_none_match = None
        if task_delete_options is not None:
            if_none_match = task_delete_options.if_none_match
        if_modified_since = None
        if task_delete_options is not None:
            if_modified_since = task_delete_options.if_modified_since
        if_unmodified_since = None
        if task_delete_options is not None:
            if_unmodified_since = task_delete_options.if_unmodified_since
        url = self.delete.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'jobId': self._serialize.url('job_id', job_id, 'str'), 'taskId': self._serialize.url('task_id', task_id, 'str')}
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
        if response.status_code not in [200]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str'})
            return client_raw_response
    delete.metadata = {'url': '/jobs/{jobId}/tasks/{taskId}'}

    def get(self, job_id, task_id, task_get_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Gets information about the specified Task.\n\n        For multi-instance Tasks, information such as affinityId, executionInfo\n        and nodeInfo refer to the primary Task. Use the list subtasks API to\n        retrieve information about subtasks.\n\n        :param job_id: The ID of the Job that contains the Task.\n        :type job_id: str\n        :param task_id: The ID of the Task to get information about.\n        :type task_id: str\n        :param task_get_options: Additional parameters for the operation\n        :type task_get_options: ~azure.batch.models.TaskGetOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: CloudTask or ClientRawResponse if raw=true\n        :rtype: ~azure.batch.models.CloudTask or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        select = None
        if task_get_options is not None:
            select = task_get_options.select
        expand = None
        if task_get_options is not None:
            expand = task_get_options.expand
        timeout = None
        if task_get_options is not None:
            timeout = task_get_options.timeout
        client_request_id = None
        if task_get_options is not None:
            client_request_id = task_get_options.client_request_id
        return_client_request_id = None
        if task_get_options is not None:
            return_client_request_id = task_get_options.return_client_request_id
        ocp_date = None
        if task_get_options is not None:
            ocp_date = task_get_options.ocp_date
        if_match = None
        if task_get_options is not None:
            if_match = task_get_options.if_match
        if_none_match = None
        if task_get_options is not None:
            if_none_match = task_get_options.if_none_match
        if_modified_since = None
        if task_get_options is not None:
            if_modified_since = task_get_options.if_modified_since
        if_unmodified_since = None
        if task_get_options is not None:
            if_unmodified_since = task_get_options.if_unmodified_since
        url = self.get.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'jobId': self._serialize.url('job_id', job_id, 'str'), 'taskId': self._serialize.url('task_id', task_id, 'str')}
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
            deserialized = self._deserialize('CloudTask', response)
            header_dict = {'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123', 'DataServiceId': 'str'}
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            client_raw_response.add_headers(header_dict)
            return client_raw_response
        return deserialized
    get.metadata = {'url': '/jobs/{jobId}/tasks/{taskId}'}

    def update(self, job_id, task_id, constraints=None, task_update_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Updates the properties of the specified Task.\n\n        :param job_id: The ID of the Job containing the Task.\n        :type job_id: str\n        :param task_id: The ID of the Task to update.\n        :type task_id: str\n        :param constraints: Constraints that apply to this Task. If omitted,\n         the Task is given the default constraints. For multi-instance Tasks,\n         updating the retention time applies only to the primary Task and not\n         subtasks.\n        :type constraints: ~azure.batch.models.TaskConstraints\n        :param task_update_options: Additional parameters for the operation\n        :type task_update_options: ~azure.batch.models.TaskUpdateOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if task_update_options is not None:
            timeout = task_update_options.timeout
        client_request_id = None
        if task_update_options is not None:
            client_request_id = task_update_options.client_request_id
        return_client_request_id = None
        if task_update_options is not None:
            return_client_request_id = task_update_options.return_client_request_id
        ocp_date = None
        if task_update_options is not None:
            ocp_date = task_update_options.ocp_date
        if_match = None
        if task_update_options is not None:
            if_match = task_update_options.if_match
        if_none_match = None
        if task_update_options is not None:
            if_none_match = task_update_options.if_none_match
        if_modified_since = None
        if task_update_options is not None:
            if_modified_since = task_update_options.if_modified_since
        if_unmodified_since = None
        if task_update_options is not None:
            if_unmodified_since = task_update_options.if_unmodified_since
        task_update_parameter = models.TaskUpdateParameter(constraints=constraints)
        url = self.update.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'jobId': self._serialize.url('job_id', job_id, 'str'), 'taskId': self._serialize.url('task_id', task_id, 'str')}
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
        body_content = self._serialize.body(task_update_parameter, 'TaskUpdateParameter')
        request = self._client.put(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123', 'DataServiceId': 'str'})
            return client_raw_response
    update.metadata = {'url': '/jobs/{jobId}/tasks/{taskId}'}

    def list_subtasks(self, job_id, task_id, task_list_subtasks_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Lists all of the subtasks that are associated with the specified\n        multi-instance Task.\n\n        If the Task is not a multi-instance Task then this returns an empty\n        collection.\n\n        :param job_id: The ID of the Job.\n        :type job_id: str\n        :param task_id: The ID of the Task.\n        :type task_id: str\n        :param task_list_subtasks_options: Additional parameters for the\n         operation\n        :type task_list_subtasks_options:\n         ~azure.batch.models.TaskListSubtasksOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: CloudTaskListSubtasksResult or ClientRawResponse if raw=true\n        :rtype: ~azure.batch.models.CloudTaskListSubtasksResult or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        select = None
        if task_list_subtasks_options is not None:
            select = task_list_subtasks_options.select
        timeout = None
        if task_list_subtasks_options is not None:
            timeout = task_list_subtasks_options.timeout
        client_request_id = None
        if task_list_subtasks_options is not None:
            client_request_id = task_list_subtasks_options.client_request_id
        return_client_request_id = None
        if task_list_subtasks_options is not None:
            return_client_request_id = task_list_subtasks_options.return_client_request_id
        ocp_date = None
        if task_list_subtasks_options is not None:
            ocp_date = task_list_subtasks_options.ocp_date
        url = self.list_subtasks.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'jobId': self._serialize.url('job_id', job_id, 'str'), 'taskId': self._serialize.url('task_id', task_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
        if select is not None:
            query_parameters['$select'] = self._serialize.query('select', select, 'str')
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
            deserialized = self._deserialize('CloudTaskListSubtasksResult', response)
            header_dict = {'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123'}
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            client_raw_response.add_headers(header_dict)
            return client_raw_response
        return deserialized
    list_subtasks.metadata = {'url': '/jobs/{jobId}/tasks/{taskId}/subtasksinfo'}

    def terminate(self, job_id, task_id, task_terminate_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Terminates the specified Task.\n\n        When the Task has been terminated, it moves to the completed state. For\n        multi-instance Tasks, the terminate Task operation applies\n        synchronously to the primary task; subtasks are then terminated\n        asynchronously in the background.\n\n        :param job_id: The ID of the Job containing the Task.\n        :type job_id: str\n        :param task_id: The ID of the Task to terminate.\n        :type task_id: str\n        :param task_terminate_options: Additional parameters for the operation\n        :type task_terminate_options: ~azure.batch.models.TaskTerminateOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if task_terminate_options is not None:
            timeout = task_terminate_options.timeout
        client_request_id = None
        if task_terminate_options is not None:
            client_request_id = task_terminate_options.client_request_id
        return_client_request_id = None
        if task_terminate_options is not None:
            return_client_request_id = task_terminate_options.return_client_request_id
        ocp_date = None
        if task_terminate_options is not None:
            ocp_date = task_terminate_options.ocp_date
        if_match = None
        if task_terminate_options is not None:
            if_match = task_terminate_options.if_match
        if_none_match = None
        if task_terminate_options is not None:
            if_none_match = task_terminate_options.if_none_match
        if_modified_since = None
        if task_terminate_options is not None:
            if_modified_since = task_terminate_options.if_modified_since
        if_unmodified_since = None
        if task_terminate_options is not None:
            if_unmodified_since = task_terminate_options.if_unmodified_since
        url = self.terminate.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'jobId': self._serialize.url('job_id', job_id, 'str'), 'taskId': self._serialize.url('task_id', task_id, 'str')}
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
        if response.status_code not in [204]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123', 'DataServiceId': 'str'})
            return client_raw_response
    terminate.metadata = {'url': '/jobs/{jobId}/tasks/{taskId}/terminate'}

    def reactivate(self, job_id, task_id, task_reactivate_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        "Reactivates a Task, allowing it to run again even if its retry count\n        has been exhausted.\n\n        Reactivation makes a Task eligible to be retried again up to its\n        maximum retry count. The Task's state is changed to active. As the Task\n        is no longer in the completed state, any previous exit code or failure\n        information is no longer available after reactivation. Each time a Task\n        is reactivated, its retry count is reset to 0. Reactivation will fail\n        for Tasks that are not completed or that previously completed\n        successfully (with an exit code of 0). Additionally, it will fail if\n        the Job has completed (or is terminating or deleting).\n\n        :param job_id: The ID of the Job containing the Task.\n        :type job_id: str\n        :param task_id: The ID of the Task to reactivate.\n        :type task_id: str\n        :param task_reactivate_options: Additional parameters for the\n         operation\n        :type task_reactivate_options:\n         ~azure.batch.models.TaskReactivateOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        "
        timeout = None
        if task_reactivate_options is not None:
            timeout = task_reactivate_options.timeout
        client_request_id = None
        if task_reactivate_options is not None:
            client_request_id = task_reactivate_options.client_request_id
        return_client_request_id = None
        if task_reactivate_options is not None:
            return_client_request_id = task_reactivate_options.return_client_request_id
        ocp_date = None
        if task_reactivate_options is not None:
            ocp_date = task_reactivate_options.ocp_date
        if_match = None
        if task_reactivate_options is not None:
            if_match = task_reactivate_options.if_match
        if_none_match = None
        if task_reactivate_options is not None:
            if_none_match = task_reactivate_options.if_none_match
        if_modified_since = None
        if task_reactivate_options is not None:
            if_modified_since = task_reactivate_options.if_modified_since
        if_unmodified_since = None
        if task_reactivate_options is not None:
            if_unmodified_since = task_reactivate_options.if_unmodified_since
        url = self.reactivate.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'jobId': self._serialize.url('job_id', job_id, 'str'), 'taskId': self._serialize.url('task_id', task_id, 'str')}
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
        if response.status_code not in [204]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123', 'DataServiceId': 'str'})
            return client_raw_response
    reactivate.metadata = {'url': '/jobs/{jobId}/tasks/{taskId}/reactivate'}