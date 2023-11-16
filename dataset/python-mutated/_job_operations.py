import uuid
from msrest.pipeline import ClientRawResponse
from .. import models

class JobOperations(object):
    """JobOperations operations.

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
            return 10
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self.api_version = '2023-05-01.17.0'
        self.config = config

    def delete(self, job_id, job_delete_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        "Deletes a Job.\n\n        Deleting a Job also deletes all Tasks that are part of that Job, and\n        all Job statistics. This also overrides the retention period for Task\n        data; that is, if the Job contains Tasks which are still retained on\n        Compute Nodes, the Batch services deletes those Tasks' working\n        directories and all their contents.  When a Delete Job request is\n        received, the Batch service sets the Job to the deleting state. All\n        update operations on a Job that is in deleting state will fail with\n        status code 409 (Conflict), with additional information indicating that\n        the Job is being deleted.\n\n        :param job_id: The ID of the Job to delete.\n        :type job_id: str\n        :param job_delete_options: Additional parameters for the operation\n        :type job_delete_options: ~azure.batch.models.JobDeleteOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        "
        timeout = None
        if job_delete_options is not None:
            timeout = job_delete_options.timeout
        client_request_id = None
        if job_delete_options is not None:
            client_request_id = job_delete_options.client_request_id
        return_client_request_id = None
        if job_delete_options is not None:
            return_client_request_id = job_delete_options.return_client_request_id
        ocp_date = None
        if job_delete_options is not None:
            ocp_date = job_delete_options.ocp_date
        if_match = None
        if job_delete_options is not None:
            if_match = job_delete_options.if_match
        if_none_match = None
        if job_delete_options is not None:
            if_none_match = job_delete_options.if_none_match
        if_modified_since = None
        if job_delete_options is not None:
            if_modified_since = job_delete_options.if_modified_since
        if_unmodified_since = None
        if job_delete_options is not None:
            if_unmodified_since = job_delete_options.if_unmodified_since
        url = self.delete.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'jobId': self._serialize.url('job_id', job_id, 'str')}
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
    delete.metadata = {'url': '/jobs/{jobId}'}

    def get(self, job_id, job_get_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Gets information about the specified Job.\n\n        :param job_id: The ID of the Job.\n        :type job_id: str\n        :param job_get_options: Additional parameters for the operation\n        :type job_get_options: ~azure.batch.models.JobGetOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: CloudJob or ClientRawResponse if raw=true\n        :rtype: ~azure.batch.models.CloudJob or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        select = None
        if job_get_options is not None:
            select = job_get_options.select
        expand = None
        if job_get_options is not None:
            expand = job_get_options.expand
        timeout = None
        if job_get_options is not None:
            timeout = job_get_options.timeout
        client_request_id = None
        if job_get_options is not None:
            client_request_id = job_get_options.client_request_id
        return_client_request_id = None
        if job_get_options is not None:
            return_client_request_id = job_get_options.return_client_request_id
        ocp_date = None
        if job_get_options is not None:
            ocp_date = job_get_options.ocp_date
        if_match = None
        if job_get_options is not None:
            if_match = job_get_options.if_match
        if_none_match = None
        if job_get_options is not None:
            if_none_match = job_get_options.if_none_match
        if_modified_since = None
        if job_get_options is not None:
            if_modified_since = job_get_options.if_modified_since
        if_unmodified_since = None
        if job_get_options is not None:
            if_unmodified_since = job_get_options.if_unmodified_since
        url = self.get.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'jobId': self._serialize.url('job_id', job_id, 'str')}
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
            deserialized = self._deserialize('CloudJob', response)
            header_dict = {'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123'}
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            client_raw_response.add_headers(header_dict)
            return client_raw_response
        return deserialized
    get.metadata = {'url': '/jobs/{jobId}'}

    def patch(self, job_id, job_patch_parameter, job_patch_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Updates the properties of the specified Job.\n\n        This replaces only the Job properties specified in the request. For\n        example, if the Job has constraints, and a request does not specify the\n        constraints element, then the Job keeps the existing constraints.\n\n        :param job_id: The ID of the Job whose properties you want to update.\n        :type job_id: str\n        :param job_patch_parameter: The parameters for the request.\n        :type job_patch_parameter: ~azure.batch.models.JobPatchParameter\n        :param job_patch_options: Additional parameters for the operation\n        :type job_patch_options: ~azure.batch.models.JobPatchOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if job_patch_options is not None:
            timeout = job_patch_options.timeout
        client_request_id = None
        if job_patch_options is not None:
            client_request_id = job_patch_options.client_request_id
        return_client_request_id = None
        if job_patch_options is not None:
            return_client_request_id = job_patch_options.return_client_request_id
        ocp_date = None
        if job_patch_options is not None:
            ocp_date = job_patch_options.ocp_date
        if_match = None
        if job_patch_options is not None:
            if_match = job_patch_options.if_match
        if_none_match = None
        if job_patch_options is not None:
            if_none_match = job_patch_options.if_none_match
        if_modified_since = None
        if job_patch_options is not None:
            if_modified_since = job_patch_options.if_modified_since
        if_unmodified_since = None
        if job_patch_options is not None:
            if_unmodified_since = job_patch_options.if_unmodified_since
        url = self.patch.metadata['url']
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
        if if_match is not None:
            header_parameters['If-Match'] = self._serialize.header('if_match', if_match, 'str')
        if if_none_match is not None:
            header_parameters['If-None-Match'] = self._serialize.header('if_none_match', if_none_match, 'str')
        if if_modified_since is not None:
            header_parameters['If-Modified-Since'] = self._serialize.header('if_modified_since', if_modified_since, 'rfc-1123')
        if if_unmodified_since is not None:
            header_parameters['If-Unmodified-Since'] = self._serialize.header('if_unmodified_since', if_unmodified_since, 'rfc-1123')
        body_content = self._serialize.body(job_patch_parameter, 'JobPatchParameter')
        request = self._client.patch(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123', 'DataServiceId': 'str'})
            return client_raw_response
    patch.metadata = {'url': '/jobs/{jobId}'}

    def update(self, job_id, job_update_parameter, job_update_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Updates the properties of the specified Job.\n\n        This fully replaces all the updatable properties of the Job. For\n        example, if the Job has constraints associated with it and if\n        constraints is not specified with this request, then the Batch service\n        will remove the existing constraints.\n\n        :param job_id: The ID of the Job whose properties you want to update.\n        :type job_id: str\n        :param job_update_parameter: The parameters for the request.\n        :type job_update_parameter: ~azure.batch.models.JobUpdateParameter\n        :param job_update_options: Additional parameters for the operation\n        :type job_update_options: ~azure.batch.models.JobUpdateOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if job_update_options is not None:
            timeout = job_update_options.timeout
        client_request_id = None
        if job_update_options is not None:
            client_request_id = job_update_options.client_request_id
        return_client_request_id = None
        if job_update_options is not None:
            return_client_request_id = job_update_options.return_client_request_id
        ocp_date = None
        if job_update_options is not None:
            ocp_date = job_update_options.ocp_date
        if_match = None
        if job_update_options is not None:
            if_match = job_update_options.if_match
        if_none_match = None
        if job_update_options is not None:
            if_none_match = job_update_options.if_none_match
        if_modified_since = None
        if job_update_options is not None:
            if_modified_since = job_update_options.if_modified_since
        if_unmodified_since = None
        if job_update_options is not None:
            if_unmodified_since = job_update_options.if_unmodified_since
        url = self.update.metadata['url']
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
        if if_match is not None:
            header_parameters['If-Match'] = self._serialize.header('if_match', if_match, 'str')
        if if_none_match is not None:
            header_parameters['If-None-Match'] = self._serialize.header('if_none_match', if_none_match, 'str')
        if if_modified_since is not None:
            header_parameters['If-Modified-Since'] = self._serialize.header('if_modified_since', if_modified_since, 'rfc-1123')
        if if_unmodified_since is not None:
            header_parameters['If-Unmodified-Since'] = self._serialize.header('if_unmodified_since', if_unmodified_since, 'rfc-1123')
        body_content = self._serialize.body(job_update_parameter, 'JobUpdateParameter')
        request = self._client.put(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123', 'DataServiceId': 'str'})
            return client_raw_response
    update.metadata = {'url': '/jobs/{jobId}'}

    def disable(self, job_id, disable_tasks, job_disable_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        "Disables the specified Job, preventing new Tasks from running.\n\n        The Batch Service immediately moves the Job to the disabling state.\n        Batch then uses the disableTasks parameter to determine what to do with\n        the currently running Tasks of the Job. The Job remains in the\n        disabling state until the disable operation is completed and all Tasks\n        have been dealt with according to the disableTasks option; the Job then\n        moves to the disabled state. No new Tasks are started under the Job\n        until it moves back to active state. If you try to disable a Job that\n        is in any state other than active, disabling, or disabled, the request\n        fails with status code 409.\n\n        :param job_id: The ID of the Job to disable.\n        :type job_id: str\n        :param disable_tasks: Possible values include: 'requeue', 'terminate',\n         'wait'\n        :type disable_tasks: str or ~azure.batch.models.DisableJobOption\n        :param job_disable_options: Additional parameters for the operation\n        :type job_disable_options: ~azure.batch.models.JobDisableOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        "
        timeout = None
        if job_disable_options is not None:
            timeout = job_disable_options.timeout
        client_request_id = None
        if job_disable_options is not None:
            client_request_id = job_disable_options.client_request_id
        return_client_request_id = None
        if job_disable_options is not None:
            return_client_request_id = job_disable_options.return_client_request_id
        ocp_date = None
        if job_disable_options is not None:
            ocp_date = job_disable_options.ocp_date
        if_match = None
        if job_disable_options is not None:
            if_match = job_disable_options.if_match
        if_none_match = None
        if job_disable_options is not None:
            if_none_match = job_disable_options.if_none_match
        if_modified_since = None
        if job_disable_options is not None:
            if_modified_since = job_disable_options.if_modified_since
        if_unmodified_since = None
        if job_disable_options is not None:
            if_unmodified_since = job_disable_options.if_unmodified_since
        job_disable_parameter = models.JobDisableParameter(disable_tasks=disable_tasks)
        url = self.disable.metadata['url']
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
        if if_match is not None:
            header_parameters['If-Match'] = self._serialize.header('if_match', if_match, 'str')
        if if_none_match is not None:
            header_parameters['If-None-Match'] = self._serialize.header('if_none_match', if_none_match, 'str')
        if if_modified_since is not None:
            header_parameters['If-Modified-Since'] = self._serialize.header('if_modified_since', if_modified_since, 'rfc-1123')
        if if_unmodified_since is not None:
            header_parameters['If-Unmodified-Since'] = self._serialize.header('if_unmodified_since', if_unmodified_since, 'rfc-1123')
        body_content = self._serialize.body(job_disable_parameter, 'JobDisableParameter')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [202]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123', 'DataServiceId': 'str'})
            return client_raw_response
    disable.metadata = {'url': '/jobs/{jobId}/disable'}

    def enable(self, job_id, job_enable_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Enables the specified Job, allowing new Tasks to run.\n\n        When you call this API, the Batch service sets a disabled Job to the\n        enabling state. After the this operation is completed, the Job moves to\n        the active state, and scheduling of new Tasks under the Job resumes.\n        The Batch service does not allow a Task to remain in the active state\n        for more than 180 days. Therefore, if you enable a Job containing\n        active Tasks which were added more than 180 days ago, those Tasks will\n        not run.\n\n        :param job_id: The ID of the Job to enable.\n        :type job_id: str\n        :param job_enable_options: Additional parameters for the operation\n        :type job_enable_options: ~azure.batch.models.JobEnableOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if job_enable_options is not None:
            timeout = job_enable_options.timeout
        client_request_id = None
        if job_enable_options is not None:
            client_request_id = job_enable_options.client_request_id
        return_client_request_id = None
        if job_enable_options is not None:
            return_client_request_id = job_enable_options.return_client_request_id
        ocp_date = None
        if job_enable_options is not None:
            ocp_date = job_enable_options.ocp_date
        if_match = None
        if job_enable_options is not None:
            if_match = job_enable_options.if_match
        if_none_match = None
        if job_enable_options is not None:
            if_none_match = job_enable_options.if_none_match
        if_modified_since = None
        if job_enable_options is not None:
            if_modified_since = job_enable_options.if_modified_since
        if_unmodified_since = None
        if job_enable_options is not None:
            if_unmodified_since = job_enable_options.if_unmodified_since
        url = self.enable.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'jobId': self._serialize.url('job_id', job_id, 'str')}
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
    enable.metadata = {'url': '/jobs/{jobId}/enable'}

    def terminate(self, job_id, terminate_reason=None, job_terminate_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Terminates the specified Job, marking it as completed.\n\n        When a Terminate Job request is received, the Batch service sets the\n        Job to the terminating state. The Batch service then terminates any\n        running Tasks associated with the Job and runs any required Job release\n        Tasks. Then the Job moves into the completed state. If there are any\n        Tasks in the Job in the active state, they will remain in the active\n        state. Once a Job is terminated, new Tasks cannot be added and any\n        remaining active Tasks will not be scheduled.\n\n        :param job_id: The ID of the Job to terminate.\n        :type job_id: str\n        :param terminate_reason:\n        :type terminate_reason: str\n        :param job_terminate_options: Additional parameters for the operation\n        :type job_terminate_options: ~azure.batch.models.JobTerminateOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if job_terminate_options is not None:
            timeout = job_terminate_options.timeout
        client_request_id = None
        if job_terminate_options is not None:
            client_request_id = job_terminate_options.client_request_id
        return_client_request_id = None
        if job_terminate_options is not None:
            return_client_request_id = job_terminate_options.return_client_request_id
        ocp_date = None
        if job_terminate_options is not None:
            ocp_date = job_terminate_options.ocp_date
        if_match = None
        if job_terminate_options is not None:
            if_match = job_terminate_options.if_match
        if_none_match = None
        if job_terminate_options is not None:
            if_none_match = job_terminate_options.if_none_match
        if_modified_since = None
        if job_terminate_options is not None:
            if_modified_since = job_terminate_options.if_modified_since
        if_unmodified_since = None
        if job_terminate_options is not None:
            if_unmodified_since = job_terminate_options.if_unmodified_since
        job_terminate_parameter = None
        if terminate_reason is not None:
            job_terminate_parameter = models.JobTerminateParameter(terminate_reason=terminate_reason)
        url = self.terminate.metadata['url']
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
        if if_match is not None:
            header_parameters['If-Match'] = self._serialize.header('if_match', if_match, 'str')
        if if_none_match is not None:
            header_parameters['If-None-Match'] = self._serialize.header('if_none_match', if_none_match, 'str')
        if if_modified_since is not None:
            header_parameters['If-Modified-Since'] = self._serialize.header('if_modified_since', if_modified_since, 'rfc-1123')
        if if_unmodified_since is not None:
            header_parameters['If-Unmodified-Since'] = self._serialize.header('if_unmodified_since', if_unmodified_since, 'rfc-1123')
        if job_terminate_parameter is not None:
            body_content = self._serialize.body(job_terminate_parameter, 'JobTerminateParameter')
        else:
            body_content = None
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [202]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123', 'DataServiceId': 'str'})
            return client_raw_response
    terminate.metadata = {'url': '/jobs/{jobId}/terminate'}

    def add(self, job, job_add_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Adds a Job to the specified Account.\n\n        The Batch service supports two ways to control the work done as part of\n        a Job. In the first approach, the user specifies a Job Manager Task.\n        The Batch service launches this Task when it is ready to start the Job.\n        The Job Manager Task controls all other Tasks that run under this Job,\n        by using the Task APIs. In the second approach, the user directly\n        controls the execution of Tasks under an active Job, by using the Task\n        APIs. Also note: when naming Jobs, avoid including sensitive\n        information such as user names or secret project names. This\n        information may appear in telemetry logs accessible to Microsoft\n        Support engineers.\n\n        :param job: The Job to be added.\n        :type job: ~azure.batch.models.JobAddParameter\n        :param job_add_options: Additional parameters for the operation\n        :type job_add_options: ~azure.batch.models.JobAddOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if job_add_options is not None:
            timeout = job_add_options.timeout
        client_request_id = None
        if job_add_options is not None:
            client_request_id = job_add_options.client_request_id
        return_client_request_id = None
        if job_add_options is not None:
            return_client_request_id = job_add_options.return_client_request_id
        ocp_date = None
        if job_add_options is not None:
            ocp_date = job_add_options.ocp_date
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
        body_content = self._serialize.body(job, 'JobAddParameter')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [201]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123', 'DataServiceId': 'str'})
            return client_raw_response
    add.metadata = {'url': '/jobs'}

    def list(self, job_list_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Lists all of the Jobs in the specified Account.\n\n        :param job_list_options: Additional parameters for the operation\n        :type job_list_options: ~azure.batch.models.JobListOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: An iterator like instance of CloudJob\n        :rtype:\n         ~azure.batch.models.CloudJobPaged[~azure.batch.models.CloudJob]\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        filter = None
        if job_list_options is not None:
            filter = job_list_options.filter
        select = None
        if job_list_options is not None:
            select = job_list_options.select
        expand = None
        if job_list_options is not None:
            expand = job_list_options.expand
        max_results = None
        if job_list_options is not None:
            max_results = job_list_options.max_results
        timeout = None
        if job_list_options is not None:
            timeout = job_list_options.timeout
        client_request_id = None
        if job_list_options is not None:
            client_request_id = job_list_options.client_request_id
        return_client_request_id = None
        if job_list_options is not None:
            return_client_request_id = job_list_options.return_client_request_id
        ocp_date = None
        if job_list_options is not None:
            ocp_date = job_list_options.ocp_date

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
                print('Hello World!')
            request = prepare_request(next_link)
            response = self._client.send(request, stream=False, **operation_config)
            if response.status_code not in [200]:
                raise models.BatchErrorException(self._deserialize, response)
            return response
        header_dict = None
        if raw:
            header_dict = {}
        deserialized = models.CloudJobPaged(internal_paging, self._deserialize.dependencies, header_dict)
        return deserialized
    list.metadata = {'url': '/jobs'}

    def list_from_job_schedule(self, job_schedule_id, job_list_from_job_schedule_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Lists the Jobs that have been created under the specified Job Schedule.\n\n        :param job_schedule_id: The ID of the Job Schedule from which you want\n         to get a list of Jobs.\n        :type job_schedule_id: str\n        :param job_list_from_job_schedule_options: Additional parameters for\n         the operation\n        :type job_list_from_job_schedule_options:\n         ~azure.batch.models.JobListFromJobScheduleOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: An iterator like instance of CloudJob\n        :rtype:\n         ~azure.batch.models.CloudJobPaged[~azure.batch.models.CloudJob]\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        filter = None
        if job_list_from_job_schedule_options is not None:
            filter = job_list_from_job_schedule_options.filter
        select = None
        if job_list_from_job_schedule_options is not None:
            select = job_list_from_job_schedule_options.select
        expand = None
        if job_list_from_job_schedule_options is not None:
            expand = job_list_from_job_schedule_options.expand
        max_results = None
        if job_list_from_job_schedule_options is not None:
            max_results = job_list_from_job_schedule_options.max_results
        timeout = None
        if job_list_from_job_schedule_options is not None:
            timeout = job_list_from_job_schedule_options.timeout
        client_request_id = None
        if job_list_from_job_schedule_options is not None:
            client_request_id = job_list_from_job_schedule_options.client_request_id
        return_client_request_id = None
        if job_list_from_job_schedule_options is not None:
            return_client_request_id = job_list_from_job_schedule_options.return_client_request_id
        ocp_date = None
        if job_list_from_job_schedule_options is not None:
            ocp_date = job_list_from_job_schedule_options.ocp_date

        def prepare_request(next_link=None):
            if False:
                return 10
            if not next_link:
                url = self.list_from_job_schedule.metadata['url']
                path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'jobScheduleId': self._serialize.url('job_schedule_id', job_schedule_id, 'str')}
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
        deserialized = models.CloudJobPaged(internal_paging, self._deserialize.dependencies, header_dict)
        return deserialized
    list_from_job_schedule.metadata = {'url': '/jobschedules/{jobScheduleId}/jobs'}

    def list_preparation_and_release_task_status(self, job_id, job_list_preparation_and_release_task_status_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Lists the execution status of the Job Preparation and Job Release Task\n        for the specified Job across the Compute Nodes where the Job has run.\n\n        This API returns the Job Preparation and Job Release Task status on all\n        Compute Nodes that have run the Job Preparation or Job Release Task.\n        This includes Compute Nodes which have since been removed from the\n        Pool. If this API is invoked on a Job which has no Job Preparation or\n        Job Release Task, the Batch service returns HTTP status code 409\n        (Conflict) with an error code of JobPreparationTaskNotSpecified.\n\n        :param job_id: The ID of the Job.\n        :type job_id: str\n        :param job_list_preparation_and_release_task_status_options:\n         Additional parameters for the operation\n        :type job_list_preparation_and_release_task_status_options:\n         ~azure.batch.models.JobListPreparationAndReleaseTaskStatusOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: An iterator like instance of\n         JobPreparationAndReleaseTaskExecutionInformation\n        :rtype:\n         ~azure.batch.models.JobPreparationAndReleaseTaskExecutionInformationPaged[~azure.batch.models.JobPreparationAndReleaseTaskExecutionInformation]\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        filter = None
        if job_list_preparation_and_release_task_status_options is not None:
            filter = job_list_preparation_and_release_task_status_options.filter
        select = None
        if job_list_preparation_and_release_task_status_options is not None:
            select = job_list_preparation_and_release_task_status_options.select
        max_results = None
        if job_list_preparation_and_release_task_status_options is not None:
            max_results = job_list_preparation_and_release_task_status_options.max_results
        timeout = None
        if job_list_preparation_and_release_task_status_options is not None:
            timeout = job_list_preparation_and_release_task_status_options.timeout
        client_request_id = None
        if job_list_preparation_and_release_task_status_options is not None:
            client_request_id = job_list_preparation_and_release_task_status_options.client_request_id
        return_client_request_id = None
        if job_list_preparation_and_release_task_status_options is not None:
            return_client_request_id = job_list_preparation_and_release_task_status_options.return_client_request_id
        ocp_date = None
        if job_list_preparation_and_release_task_status_options is not None:
            ocp_date = job_list_preparation_and_release_task_status_options.ocp_date

        def prepare_request(next_link=None):
            if False:
                print('Hello World!')
            if not next_link:
                url = self.list_preparation_and_release_task_status.metadata['url']
                path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'jobId': self._serialize.url('job_id', job_id, 'str')}
                url = self._client.format_url(url, **path_format_arguments)
                query_parameters = {}
                query_parameters['api-version'] = self._serialize.query('self.api_version', self.api_version, 'str')
                if filter is not None:
                    query_parameters['$filter'] = self._serialize.query('filter', filter, 'str')
                if select is not None:
                    query_parameters['$select'] = self._serialize.query('select', select, 'str')
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
                while True:
                    i = 10
            request = prepare_request(next_link)
            response = self._client.send(request, stream=False, **operation_config)
            if response.status_code not in [200]:
                raise models.BatchErrorException(self._deserialize, response)
            return response
        header_dict = None
        if raw:
            header_dict = {}
        deserialized = models.JobPreparationAndReleaseTaskExecutionInformationPaged(internal_paging, self._deserialize.dependencies, header_dict)
        return deserialized
    list_preparation_and_release_task_status.metadata = {'url': '/jobs/{jobId}/jobpreparationandreleasetaskstatus'}

    def get_task_counts(self, job_id, job_get_task_counts_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Gets the Task counts for the specified Job.\n\n        Task counts provide a count of the Tasks by active, running or\n        completed Task state, and a count of Tasks which succeeded or failed.\n        Tasks in the preparing state are counted as running. Note that the\n        numbers returned may not always be up to date. If you need exact task\n        counts, use a list query.\n\n        :param job_id: The ID of the Job.\n        :type job_id: str\n        :param job_get_task_counts_options: Additional parameters for the\n         operation\n        :type job_get_task_counts_options:\n         ~azure.batch.models.JobGetTaskCountsOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: TaskCountsResult or ClientRawResponse if raw=true\n        :rtype: ~azure.batch.models.TaskCountsResult or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if job_get_task_counts_options is not None:
            timeout = job_get_task_counts_options.timeout
        client_request_id = None
        if job_get_task_counts_options is not None:
            client_request_id = job_get_task_counts_options.client_request_id
        return_client_request_id = None
        if job_get_task_counts_options is not None:
            return_client_request_id = job_get_task_counts_options.return_client_request_id
        ocp_date = None
        if job_get_task_counts_options is not None:
            ocp_date = job_get_task_counts_options.ocp_date
        url = self.get_task_counts.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'jobId': self._serialize.url('job_id', job_id, 'str')}
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
            deserialized = self._deserialize('TaskCountsResult', response)
            header_dict = {'client-request-id': 'str', 'request-id': 'str'}
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            client_raw_response.add_headers(header_dict)
            return client_raw_response
        return deserialized
    get_task_counts.metadata = {'url': '/jobs/{jobId}/taskcounts'}