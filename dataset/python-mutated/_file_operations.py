import uuid
from msrest.pipeline import ClientRawResponse
from .. import models

class FileOperations(object):
    """FileOperations operations.

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
            for i in range(10):
                print('nop')
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self.api_version = '2023-05-01.17.0'
        self.config = config

    def delete_from_task(self, job_id, task_id, file_path, recursive=None, file_delete_from_task_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Deletes the specified Task file from the Compute Node where the Task\n        ran.\n\n        :param job_id: The ID of the Job that contains the Task.\n        :type job_id: str\n        :param task_id: The ID of the Task whose file you want to delete.\n        :type task_id: str\n        :param file_path: The path to the Task file or directory that you want\n         to delete.\n        :type file_path: str\n        :param recursive: Whether to delete children of a directory. If the\n         filePath parameter represents a directory instead of a file, you can\n         set recursive to true to delete the directory and all of the files and\n         subdirectories in it. If recursive is false then the directory must be\n         empty or deletion will fail.\n        :type recursive: bool\n        :param file_delete_from_task_options: Additional parameters for the\n         operation\n        :type file_delete_from_task_options:\n         ~azure.batch.models.FileDeleteFromTaskOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if file_delete_from_task_options is not None:
            timeout = file_delete_from_task_options.timeout
        client_request_id = None
        if file_delete_from_task_options is not None:
            client_request_id = file_delete_from_task_options.client_request_id
        return_client_request_id = None
        if file_delete_from_task_options is not None:
            return_client_request_id = file_delete_from_task_options.return_client_request_id
        ocp_date = None
        if file_delete_from_task_options is not None:
            ocp_date = file_delete_from_task_options.ocp_date
        url = self.delete_from_task.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'jobId': self._serialize.url('job_id', job_id, 'str'), 'taskId': self._serialize.url('task_id', task_id, 'str'), 'filePath': self._serialize.url('file_path', file_path, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if recursive is not None:
            query_parameters['recursive'] = self._serialize.query('recursive', recursive, 'bool')
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
        request = self._client.delete(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str'})
            return client_raw_response
    delete_from_task.metadata = {'url': '/jobs/{jobId}/tasks/{taskId}/files/{filePath}'}

    def get_from_task(self, job_id, task_id, file_path, file_get_from_task_options=None, custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            while True:
                i = 10
        'Returns the content of the specified Task file.\n\n        :param job_id: The ID of the Job that contains the Task.\n        :type job_id: str\n        :param task_id: The ID of the Task whose file you want to retrieve.\n        :type task_id: str\n        :param file_path: The path to the Task file that you want to get the\n         content of.\n        :type file_path: str\n        :param file_get_from_task_options: Additional parameters for the\n         operation\n        :type file_get_from_task_options:\n         ~azure.batch.models.FileGetFromTaskOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: object or ClientRawResponse if raw=true\n        :rtype: Generator or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if file_get_from_task_options is not None:
            timeout = file_get_from_task_options.timeout
        client_request_id = None
        if file_get_from_task_options is not None:
            client_request_id = file_get_from_task_options.client_request_id
        return_client_request_id = None
        if file_get_from_task_options is not None:
            return_client_request_id = file_get_from_task_options.return_client_request_id
        ocp_date = None
        if file_get_from_task_options is not None:
            ocp_date = file_get_from_task_options.ocp_date
        ocp_range = None
        if file_get_from_task_options is not None:
            ocp_range = file_get_from_task_options.ocp_range
        if_modified_since = None
        if file_get_from_task_options is not None:
            if_modified_since = file_get_from_task_options.if_modified_since
        if_unmodified_since = None
        if file_get_from_task_options is not None:
            if_unmodified_since = file_get_from_task_options.if_unmodified_since
        url = self.get_from_task.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'jobId': self._serialize.url('job_id', job_id, 'str'), 'taskId': self._serialize.url('task_id', task_id, 'str'), 'filePath': self._serialize.url('file_path', file_path, 'str')}
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
        if ocp_range is not None:
            header_parameters['ocp-range'] = self._serialize.header('ocp_range', ocp_range, 'str')
        if if_modified_since is not None:
            header_parameters['If-Modified-Since'] = self._serialize.header('if_modified_since', if_modified_since, 'rfc-1123')
        if if_unmodified_since is not None:
            header_parameters['If-Unmodified-Since'] = self._serialize.header('if_unmodified_since', if_unmodified_since, 'rfc-1123')
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=True, **operation_config)
        if response.status_code not in [200]:
            raise models.BatchErrorException(self._deserialize, response)
        header_dict = {}
        deserialized = self._client.stream_download(response, callback)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            client_raw_response.add_headers(header_dict)
            return client_raw_response
        return deserialized
    get_from_task.metadata = {'url': '/jobs/{jobId}/tasks/{taskId}/files/{filePath}'}

    def get_properties_from_task(self, job_id, task_id, file_path, file_get_properties_from_task_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Gets the properties of the specified Task file.\n\n        :param job_id: The ID of the Job that contains the Task.\n        :type job_id: str\n        :param task_id: The ID of the Task whose file you want to get the\n         properties of.\n        :type task_id: str\n        :param file_path: The path to the Task file that you want to get the\n         properties of.\n        :type file_path: str\n        :param file_get_properties_from_task_options: Additional parameters\n         for the operation\n        :type file_get_properties_from_task_options:\n         ~azure.batch.models.FileGetPropertiesFromTaskOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if file_get_properties_from_task_options is not None:
            timeout = file_get_properties_from_task_options.timeout
        client_request_id = None
        if file_get_properties_from_task_options is not None:
            client_request_id = file_get_properties_from_task_options.client_request_id
        return_client_request_id = None
        if file_get_properties_from_task_options is not None:
            return_client_request_id = file_get_properties_from_task_options.return_client_request_id
        ocp_date = None
        if file_get_properties_from_task_options is not None:
            ocp_date = file_get_properties_from_task_options.ocp_date
        if_modified_since = None
        if file_get_properties_from_task_options is not None:
            if_modified_since = file_get_properties_from_task_options.if_modified_since
        if_unmodified_since = None
        if file_get_properties_from_task_options is not None:
            if_unmodified_since = file_get_properties_from_task_options.if_unmodified_since
        url = self.get_properties_from_task.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'jobId': self._serialize.url('job_id', job_id, 'str'), 'taskId': self._serialize.url('task_id', task_id, 'str'), 'filePath': self._serialize.url('file_path', file_path, 'str')}
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
        if if_modified_since is not None:
            header_parameters['If-Modified-Since'] = self._serialize.header('if_modified_since', if_modified_since, 'rfc-1123')
        if if_unmodified_since is not None:
            header_parameters['If-Unmodified-Since'] = self._serialize.header('if_unmodified_since', if_unmodified_since, 'rfc-1123')
        request = self._client.head(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123', 'ocp-creation-time': 'rfc-1123', 'ocp-batch-file-isdirectory': 'bool', 'ocp-batch-file-url': 'str', 'ocp-batch-file-mode': 'str', 'Content-Type': 'str', 'Content-Length': 'long'})
            return client_raw_response
    get_properties_from_task.metadata = {'url': '/jobs/{jobId}/tasks/{taskId}/files/{filePath}'}

    def delete_from_compute_node(self, pool_id, node_id, file_path, recursive=None, file_delete_from_compute_node_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Deletes the specified file from the Compute Node.\n\n        :param pool_id: The ID of the Pool that contains the Compute Node.\n        :type pool_id: str\n        :param node_id: The ID of the Compute Node from which you want to\n         delete the file.\n        :type node_id: str\n        :param file_path: The path to the file or directory that you want to\n         delete.\n        :type file_path: str\n        :param recursive: Whether to delete children of a directory. If the\n         filePath parameter represents a directory instead of a file, you can\n         set recursive to true to delete the directory and all of the files and\n         subdirectories in it. If recursive is false then the directory must be\n         empty or deletion will fail.\n        :type recursive: bool\n        :param file_delete_from_compute_node_options: Additional parameters\n         for the operation\n        :type file_delete_from_compute_node_options:\n         ~azure.batch.models.FileDeleteFromComputeNodeOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if file_delete_from_compute_node_options is not None:
            timeout = file_delete_from_compute_node_options.timeout
        client_request_id = None
        if file_delete_from_compute_node_options is not None:
            client_request_id = file_delete_from_compute_node_options.client_request_id
        return_client_request_id = None
        if file_delete_from_compute_node_options is not None:
            return_client_request_id = file_delete_from_compute_node_options.return_client_request_id
        ocp_date = None
        if file_delete_from_compute_node_options is not None:
            ocp_date = file_delete_from_compute_node_options.ocp_date
        url = self.delete_from_compute_node.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str'), 'nodeId': self._serialize.url('node_id', node_id, 'str'), 'filePath': self._serialize.url('file_path', file_path, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if recursive is not None:
            query_parameters['recursive'] = self._serialize.query('recursive', recursive, 'bool')
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
        request = self._client.delete(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str'})
            return client_raw_response
    delete_from_compute_node.metadata = {'url': '/pools/{poolId}/nodes/{nodeId}/files/{filePath}'}

    def get_from_compute_node(self, pool_id, node_id, file_path, file_get_from_compute_node_options=None, custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Returns the content of the specified Compute Node file.\n\n        :param pool_id: The ID of the Pool that contains the Compute Node.\n        :type pool_id: str\n        :param node_id: The ID of the Compute Node that contains the file.\n        :type node_id: str\n        :param file_path: The path to the Compute Node file that you want to\n         get the content of.\n        :type file_path: str\n        :param file_get_from_compute_node_options: Additional parameters for\n         the operation\n        :type file_get_from_compute_node_options:\n         ~azure.batch.models.FileGetFromComputeNodeOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: object or ClientRawResponse if raw=true\n        :rtype: Generator or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if file_get_from_compute_node_options is not None:
            timeout = file_get_from_compute_node_options.timeout
        client_request_id = None
        if file_get_from_compute_node_options is not None:
            client_request_id = file_get_from_compute_node_options.client_request_id
        return_client_request_id = None
        if file_get_from_compute_node_options is not None:
            return_client_request_id = file_get_from_compute_node_options.return_client_request_id
        ocp_date = None
        if file_get_from_compute_node_options is not None:
            ocp_date = file_get_from_compute_node_options.ocp_date
        ocp_range = None
        if file_get_from_compute_node_options is not None:
            ocp_range = file_get_from_compute_node_options.ocp_range
        if_modified_since = None
        if file_get_from_compute_node_options is not None:
            if_modified_since = file_get_from_compute_node_options.if_modified_since
        if_unmodified_since = None
        if file_get_from_compute_node_options is not None:
            if_unmodified_since = file_get_from_compute_node_options.if_unmodified_since
        url = self.get_from_compute_node.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str'), 'nodeId': self._serialize.url('node_id', node_id, 'str'), 'filePath': self._serialize.url('file_path', file_path, 'str')}
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
        if ocp_range is not None:
            header_parameters['ocp-range'] = self._serialize.header('ocp_range', ocp_range, 'str')
        if if_modified_since is not None:
            header_parameters['If-Modified-Since'] = self._serialize.header('if_modified_since', if_modified_since, 'rfc-1123')
        if if_unmodified_since is not None:
            header_parameters['If-Unmodified-Since'] = self._serialize.header('if_unmodified_since', if_unmodified_since, 'rfc-1123')
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=True, **operation_config)
        if response.status_code not in [200]:
            raise models.BatchErrorException(self._deserialize, response)
        header_dict = {}
        deserialized = self._client.stream_download(response, callback)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            client_raw_response.add_headers(header_dict)
            return client_raw_response
        return deserialized
    get_from_compute_node.metadata = {'url': '/pools/{poolId}/nodes/{nodeId}/files/{filePath}'}

    def get_properties_from_compute_node(self, pool_id, node_id, file_path, file_get_properties_from_compute_node_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Gets the properties of the specified Compute Node file.\n\n        :param pool_id: The ID of the Pool that contains the Compute Node.\n        :type pool_id: str\n        :param node_id: The ID of the Compute Node that contains the file.\n        :type node_id: str\n        :param file_path: The path to the Compute Node file that you want to\n         get the properties of.\n        :type file_path: str\n        :param file_get_properties_from_compute_node_options: Additional\n         parameters for the operation\n        :type file_get_properties_from_compute_node_options:\n         ~azure.batch.models.FileGetPropertiesFromComputeNodeOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if file_get_properties_from_compute_node_options is not None:
            timeout = file_get_properties_from_compute_node_options.timeout
        client_request_id = None
        if file_get_properties_from_compute_node_options is not None:
            client_request_id = file_get_properties_from_compute_node_options.client_request_id
        return_client_request_id = None
        if file_get_properties_from_compute_node_options is not None:
            return_client_request_id = file_get_properties_from_compute_node_options.return_client_request_id
        ocp_date = None
        if file_get_properties_from_compute_node_options is not None:
            ocp_date = file_get_properties_from_compute_node_options.ocp_date
        if_modified_since = None
        if file_get_properties_from_compute_node_options is not None:
            if_modified_since = file_get_properties_from_compute_node_options.if_modified_since
        if_unmodified_since = None
        if file_get_properties_from_compute_node_options is not None:
            if_unmodified_since = file_get_properties_from_compute_node_options.if_unmodified_since
        url = self.get_properties_from_compute_node.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str'), 'nodeId': self._serialize.url('node_id', node_id, 'str'), 'filePath': self._serialize.url('file_path', file_path, 'str')}
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
        if if_modified_since is not None:
            header_parameters['If-Modified-Since'] = self._serialize.header('if_modified_since', if_modified_since, 'rfc-1123')
        if if_unmodified_since is not None:
            header_parameters['If-Unmodified-Since'] = self._serialize.header('if_unmodified_since', if_unmodified_since, 'rfc-1123')
        request = self._client.head(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123', 'ocp-creation-time': 'rfc-1123', 'ocp-batch-file-isdirectory': 'bool', 'ocp-batch-file-url': 'str', 'ocp-batch-file-mode': 'str', 'Content-Type': 'str', 'Content-Length': 'long'})
            return client_raw_response
    get_properties_from_compute_node.metadata = {'url': '/pools/{poolId}/nodes/{nodeId}/files/{filePath}'}

    def list_from_task(self, job_id, task_id, recursive=None, file_list_from_task_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        "Lists the files in a Task's directory on its Compute Node.\n\n        :param job_id: The ID of the Job that contains the Task.\n        :type job_id: str\n        :param task_id: The ID of the Task whose files you want to list.\n        :type task_id: str\n        :param recursive: Whether to list children of the Task directory. This\n         parameter can be used in combination with the filter parameter to list\n         specific type of files.\n        :type recursive: bool\n        :param file_list_from_task_options: Additional parameters for the\n         operation\n        :type file_list_from_task_options:\n         ~azure.batch.models.FileListFromTaskOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: An iterator like instance of NodeFile\n        :rtype:\n         ~azure.batch.models.NodeFilePaged[~azure.batch.models.NodeFile]\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        "
        filter = None
        if file_list_from_task_options is not None:
            filter = file_list_from_task_options.filter
        max_results = None
        if file_list_from_task_options is not None:
            max_results = file_list_from_task_options.max_results
        timeout = None
        if file_list_from_task_options is not None:
            timeout = file_list_from_task_options.timeout
        client_request_id = None
        if file_list_from_task_options is not None:
            client_request_id = file_list_from_task_options.client_request_id
        return_client_request_id = None
        if file_list_from_task_options is not None:
            return_client_request_id = file_list_from_task_options.return_client_request_id
        ocp_date = None
        if file_list_from_task_options is not None:
            ocp_date = file_list_from_task_options.ocp_date

        def prepare_request(next_link=None):
            if False:
                i = 10
                return i + 15
            if not next_link:
                url = self.list_from_task.metadata['url']
                path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'jobId': self._serialize.url('job_id', job_id, 'str'), 'taskId': self._serialize.url('task_id', task_id, 'str')}
                url = self._client.format_url(url, **path_format_arguments)
                query_parameters = {}
                if recursive is not None:
                    query_parameters['recursive'] = self._serialize.query('recursive', recursive, 'bool')
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
        deserialized = models.NodeFilePaged(internal_paging, self._deserialize.dependencies, header_dict)
        return deserialized
    list_from_task.metadata = {'url': '/jobs/{jobId}/tasks/{taskId}/files'}

    def list_from_compute_node(self, pool_id, node_id, recursive=None, file_list_from_compute_node_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Lists all of the files in Task directories on the specified Compute\n        Node.\n\n        :param pool_id: The ID of the Pool that contains the Compute Node.\n        :type pool_id: str\n        :param node_id: The ID of the Compute Node whose files you want to\n         list.\n        :type node_id: str\n        :param recursive: Whether to list children of a directory.\n        :type recursive: bool\n        :param file_list_from_compute_node_options: Additional parameters for\n         the operation\n        :type file_list_from_compute_node_options:\n         ~azure.batch.models.FileListFromComputeNodeOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: An iterator like instance of NodeFile\n        :rtype:\n         ~azure.batch.models.NodeFilePaged[~azure.batch.models.NodeFile]\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        filter = None
        if file_list_from_compute_node_options is not None:
            filter = file_list_from_compute_node_options.filter
        max_results = None
        if file_list_from_compute_node_options is not None:
            max_results = file_list_from_compute_node_options.max_results
        timeout = None
        if file_list_from_compute_node_options is not None:
            timeout = file_list_from_compute_node_options.timeout
        client_request_id = None
        if file_list_from_compute_node_options is not None:
            client_request_id = file_list_from_compute_node_options.client_request_id
        return_client_request_id = None
        if file_list_from_compute_node_options is not None:
            return_client_request_id = file_list_from_compute_node_options.return_client_request_id
        ocp_date = None
        if file_list_from_compute_node_options is not None:
            ocp_date = file_list_from_compute_node_options.ocp_date

        def prepare_request(next_link=None):
            if False:
                i = 10
                return i + 15
            if not next_link:
                url = self.list_from_compute_node.metadata['url']
                path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str'), 'nodeId': self._serialize.url('node_id', node_id, 'str')}
                url = self._client.format_url(url, **path_format_arguments)
                query_parameters = {}
                if recursive is not None:
                    query_parameters['recursive'] = self._serialize.query('recursive', recursive, 'bool')
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
        deserialized = models.NodeFilePaged(internal_paging, self._deserialize.dependencies, header_dict)
        return deserialized
    list_from_compute_node.metadata = {'url': '/pools/{poolId}/nodes/{nodeId}/files'}