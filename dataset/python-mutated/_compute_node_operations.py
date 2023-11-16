import uuid
from msrest.pipeline import ClientRawResponse
from .. import models

class ComputeNodeOperations(object):
    """ComputeNodeOperations operations.

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

    def add_user(self, pool_id, node_id, user, compute_node_add_user_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Adds a user Account to the specified Compute Node.\n\n        You can add a user Account to a Compute Node only when it is in the\n        idle or running state.\n\n        :param pool_id: The ID of the Pool that contains the Compute Node.\n        :type pool_id: str\n        :param node_id: The ID of the machine on which you want to create a\n         user Account.\n        :type node_id: str\n        :param user: The user Account to be created.\n        :type user: ~azure.batch.models.ComputeNodeUser\n        :param compute_node_add_user_options: Additional parameters for the\n         operation\n        :type compute_node_add_user_options:\n         ~azure.batch.models.ComputeNodeAddUserOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if compute_node_add_user_options is not None:
            timeout = compute_node_add_user_options.timeout
        client_request_id = None
        if compute_node_add_user_options is not None:
            client_request_id = compute_node_add_user_options.client_request_id
        return_client_request_id = None
        if compute_node_add_user_options is not None:
            return_client_request_id = compute_node_add_user_options.return_client_request_id
        ocp_date = None
        if compute_node_add_user_options is not None:
            ocp_date = compute_node_add_user_options.ocp_date
        url = self.add_user.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str'), 'nodeId': self._serialize.url('node_id', node_id, 'str')}
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
        body_content = self._serialize.body(user, 'ComputeNodeUser')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [201]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123', 'DataServiceId': 'str'})
            return client_raw_response
    add_user.metadata = {'url': '/pools/{poolId}/nodes/{nodeId}/users'}

    def delete_user(self, pool_id, node_id, user_name, compute_node_delete_user_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Deletes a user Account from the specified Compute Node.\n\n        You can delete a user Account to a Compute Node only when it is in the\n        idle or running state.\n\n        :param pool_id: The ID of the Pool that contains the Compute Node.\n        :type pool_id: str\n        :param node_id: The ID of the machine on which you want to delete a\n         user Account.\n        :type node_id: str\n        :param user_name: The name of the user Account to delete.\n        :type user_name: str\n        :param compute_node_delete_user_options: Additional parameters for the\n         operation\n        :type compute_node_delete_user_options:\n         ~azure.batch.models.ComputeNodeDeleteUserOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if compute_node_delete_user_options is not None:
            timeout = compute_node_delete_user_options.timeout
        client_request_id = None
        if compute_node_delete_user_options is not None:
            client_request_id = compute_node_delete_user_options.client_request_id
        return_client_request_id = None
        if compute_node_delete_user_options is not None:
            return_client_request_id = compute_node_delete_user_options.return_client_request_id
        ocp_date = None
        if compute_node_delete_user_options is not None:
            ocp_date = compute_node_delete_user_options.ocp_date
        url = self.delete_user.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str'), 'nodeId': self._serialize.url('node_id', node_id, 'str'), 'userName': self._serialize.url('user_name', user_name, 'str')}
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
        request = self._client.delete(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str'})
            return client_raw_response
    delete_user.metadata = {'url': '/pools/{poolId}/nodes/{nodeId}/users/{userName}'}

    def update_user(self, pool_id, node_id, user_name, node_update_user_parameter, compute_node_update_user_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Updates the password and expiration time of a user Account on the\n        specified Compute Node.\n\n        This operation replaces of all the updatable properties of the Account.\n        For example, if the expiryTime element is not specified, the current\n        value is replaced with the default value, not left unmodified. You can\n        update a user Account on a Compute Node only when it is in the idle or\n        running state.\n\n        :param pool_id: The ID of the Pool that contains the Compute Node.\n        :type pool_id: str\n        :param node_id: The ID of the machine on which you want to update a\n         user Account.\n        :type node_id: str\n        :param user_name: The name of the user Account to update.\n        :type user_name: str\n        :param node_update_user_parameter: The parameters for the request.\n        :type node_update_user_parameter:\n         ~azure.batch.models.NodeUpdateUserParameter\n        :param compute_node_update_user_options: Additional parameters for the\n         operation\n        :type compute_node_update_user_options:\n         ~azure.batch.models.ComputeNodeUpdateUserOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if compute_node_update_user_options is not None:
            timeout = compute_node_update_user_options.timeout
        client_request_id = None
        if compute_node_update_user_options is not None:
            client_request_id = compute_node_update_user_options.client_request_id
        return_client_request_id = None
        if compute_node_update_user_options is not None:
            return_client_request_id = compute_node_update_user_options.return_client_request_id
        ocp_date = None
        if compute_node_update_user_options is not None:
            ocp_date = compute_node_update_user_options.ocp_date
        url = self.update_user.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str'), 'nodeId': self._serialize.url('node_id', node_id, 'str'), 'userName': self._serialize.url('user_name', user_name, 'str')}
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
        body_content = self._serialize.body(node_update_user_parameter, 'NodeUpdateUserParameter')
        request = self._client.put(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123', 'DataServiceId': 'str'})
            return client_raw_response
    update_user.metadata = {'url': '/pools/{poolId}/nodes/{nodeId}/users/{userName}'}

    def get(self, pool_id, node_id, compute_node_get_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Gets information about the specified Compute Node.\n\n        :param pool_id: The ID of the Pool that contains the Compute Node.\n        :type pool_id: str\n        :param node_id: The ID of the Compute Node that you want to get\n         information about.\n        :type node_id: str\n        :param compute_node_get_options: Additional parameters for the\n         operation\n        :type compute_node_get_options:\n         ~azure.batch.models.ComputeNodeGetOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ComputeNode or ClientRawResponse if raw=true\n        :rtype: ~azure.batch.models.ComputeNode or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        select = None
        if compute_node_get_options is not None:
            select = compute_node_get_options.select
        timeout = None
        if compute_node_get_options is not None:
            timeout = compute_node_get_options.timeout
        client_request_id = None
        if compute_node_get_options is not None:
            client_request_id = compute_node_get_options.client_request_id
        return_client_request_id = None
        if compute_node_get_options is not None:
            return_client_request_id = compute_node_get_options.return_client_request_id
        ocp_date = None
        if compute_node_get_options is not None:
            ocp_date = compute_node_get_options.ocp_date
        url = self.get.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str'), 'nodeId': self._serialize.url('node_id', node_id, 'str')}
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
            deserialized = self._deserialize('ComputeNode', response)
            header_dict = {'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123'}
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            client_raw_response.add_headers(header_dict)
            return client_raw_response
        return deserialized
    get.metadata = {'url': '/pools/{poolId}/nodes/{nodeId}'}

    def reboot(self, pool_id, node_id, node_reboot_option=None, compute_node_reboot_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        "Restarts the specified Compute Node.\n\n        You can restart a Compute Node only if it is in an idle or running\n        state.\n\n        :param pool_id: The ID of the Pool that contains the Compute Node.\n        :type pool_id: str\n        :param node_id: The ID of the Compute Node that you want to restart.\n        :type node_id: str\n        :param node_reboot_option: The default value is requeue. Possible\n         values include: 'requeue', 'terminate', 'taskCompletion',\n         'retainedData'\n        :type node_reboot_option: str or\n         ~azure.batch.models.ComputeNodeRebootOption\n        :param compute_node_reboot_options: Additional parameters for the\n         operation\n        :type compute_node_reboot_options:\n         ~azure.batch.models.ComputeNodeRebootOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        "
        timeout = None
        if compute_node_reboot_options is not None:
            timeout = compute_node_reboot_options.timeout
        client_request_id = None
        if compute_node_reboot_options is not None:
            client_request_id = compute_node_reboot_options.client_request_id
        return_client_request_id = None
        if compute_node_reboot_options is not None:
            return_client_request_id = compute_node_reboot_options.return_client_request_id
        ocp_date = None
        if compute_node_reboot_options is not None:
            ocp_date = compute_node_reboot_options.ocp_date
        node_reboot_parameter = None
        if node_reboot_option is not None:
            node_reboot_parameter = models.NodeRebootParameter(node_reboot_option=node_reboot_option)
        url = self.reboot.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str'), 'nodeId': self._serialize.url('node_id', node_id, 'str')}
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
        if node_reboot_parameter is not None:
            body_content = self._serialize.body(node_reboot_parameter, 'NodeRebootParameter')
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
    reboot.metadata = {'url': '/pools/{poolId}/nodes/{nodeId}/reboot'}

    def reimage(self, pool_id, node_id, node_reimage_option=None, compute_node_reimage_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        "Reinstalls the operating system on the specified Compute Node.\n\n        You can reinstall the operating system on a Compute Node only if it is\n        in an idle or running state. This API can be invoked only on Pools\n        created with the cloud service configuration property.\n\n        :param pool_id: The ID of the Pool that contains the Compute Node.\n        :type pool_id: str\n        :param node_id: The ID of the Compute Node that you want to restart.\n        :type node_id: str\n        :param node_reimage_option: The default value is requeue. Possible\n         values include: 'requeue', 'terminate', 'taskCompletion',\n         'retainedData'\n        :type node_reimage_option: str or\n         ~azure.batch.models.ComputeNodeReimageOption\n        :param compute_node_reimage_options: Additional parameters for the\n         operation\n        :type compute_node_reimage_options:\n         ~azure.batch.models.ComputeNodeReimageOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        "
        timeout = None
        if compute_node_reimage_options is not None:
            timeout = compute_node_reimage_options.timeout
        client_request_id = None
        if compute_node_reimage_options is not None:
            client_request_id = compute_node_reimage_options.client_request_id
        return_client_request_id = None
        if compute_node_reimage_options is not None:
            return_client_request_id = compute_node_reimage_options.return_client_request_id
        ocp_date = None
        if compute_node_reimage_options is not None:
            ocp_date = compute_node_reimage_options.ocp_date
        node_reimage_parameter = None
        if node_reimage_option is not None:
            node_reimage_parameter = models.NodeReimageParameter(node_reimage_option=node_reimage_option)
        url = self.reimage.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str'), 'nodeId': self._serialize.url('node_id', node_id, 'str')}
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
        if node_reimage_parameter is not None:
            body_content = self._serialize.body(node_reimage_parameter, 'NodeReimageParameter')
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
    reimage.metadata = {'url': '/pools/{poolId}/nodes/{nodeId}/reimage'}

    def disable_scheduling(self, pool_id, node_id, node_disable_scheduling_option=None, compute_node_disable_scheduling_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        "Disables Task scheduling on the specified Compute Node.\n\n        You can disable Task scheduling on a Compute Node only if its current\n        scheduling state is enabled.\n\n        :param pool_id: The ID of the Pool that contains the Compute Node.\n        :type pool_id: str\n        :param node_id: The ID of the Compute Node on which you want to\n         disable Task scheduling.\n        :type node_id: str\n        :param node_disable_scheduling_option: The default value is requeue.\n         Possible values include: 'requeue', 'terminate', 'taskCompletion'\n        :type node_disable_scheduling_option: str or\n         ~azure.batch.models.DisableComputeNodeSchedulingOption\n        :param compute_node_disable_scheduling_options: Additional parameters\n         for the operation\n        :type compute_node_disable_scheduling_options:\n         ~azure.batch.models.ComputeNodeDisableSchedulingOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        "
        timeout = None
        if compute_node_disable_scheduling_options is not None:
            timeout = compute_node_disable_scheduling_options.timeout
        client_request_id = None
        if compute_node_disable_scheduling_options is not None:
            client_request_id = compute_node_disable_scheduling_options.client_request_id
        return_client_request_id = None
        if compute_node_disable_scheduling_options is not None:
            return_client_request_id = compute_node_disable_scheduling_options.return_client_request_id
        ocp_date = None
        if compute_node_disable_scheduling_options is not None:
            ocp_date = compute_node_disable_scheduling_options.ocp_date
        node_disable_scheduling_parameter = None
        if node_disable_scheduling_option is not None:
            node_disable_scheduling_parameter = models.NodeDisableSchedulingParameter(node_disable_scheduling_option=node_disable_scheduling_option)
        url = self.disable_scheduling.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str'), 'nodeId': self._serialize.url('node_id', node_id, 'str')}
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
        if node_disable_scheduling_parameter is not None:
            body_content = self._serialize.body(node_disable_scheduling_parameter, 'NodeDisableSchedulingParameter')
        else:
            body_content = None
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.BatchErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123', 'DataServiceId': 'str'})
            return client_raw_response
    disable_scheduling.metadata = {'url': '/pools/{poolId}/nodes/{nodeId}/disablescheduling'}

    def enable_scheduling(self, pool_id, node_id, compute_node_enable_scheduling_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Enables Task scheduling on the specified Compute Node.\n\n        You can enable Task scheduling on a Compute Node only if its current\n        scheduling state is disabled.\n\n        :param pool_id: The ID of the Pool that contains the Compute Node.\n        :type pool_id: str\n        :param node_id: The ID of the Compute Node on which you want to enable\n         Task scheduling.\n        :type node_id: str\n        :param compute_node_enable_scheduling_options: Additional parameters\n         for the operation\n        :type compute_node_enable_scheduling_options:\n         ~azure.batch.models.ComputeNodeEnableSchedulingOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if compute_node_enable_scheduling_options is not None:
            timeout = compute_node_enable_scheduling_options.timeout
        client_request_id = None
        if compute_node_enable_scheduling_options is not None:
            client_request_id = compute_node_enable_scheduling_options.client_request_id
        return_client_request_id = None
        if compute_node_enable_scheduling_options is not None:
            return_client_request_id = compute_node_enable_scheduling_options.return_client_request_id
        ocp_date = None
        if compute_node_enable_scheduling_options is not None:
            ocp_date = compute_node_enable_scheduling_options.ocp_date
        url = self.enable_scheduling.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str'), 'nodeId': self._serialize.url('node_id', node_id, 'str')}
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
    enable_scheduling.metadata = {'url': '/pools/{poolId}/nodes/{nodeId}/enablescheduling'}

    def get_remote_login_settings(self, pool_id, node_id, compute_node_get_remote_login_settings_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Gets the settings required for remote login to a Compute Node.\n\n        Before you can remotely login to a Compute Node using the remote login\n        settings, you must create a user Account on the Compute Node. This API\n        can be invoked only on Pools created with the virtual machine\n        configuration property. For Pools created with a cloud service\n        configuration, see the GetRemoteDesktop API.\n\n        :param pool_id: The ID of the Pool that contains the Compute Node.\n        :type pool_id: str\n        :param node_id: The ID of the Compute Node for which to obtain the\n         remote login settings.\n        :type node_id: str\n        :param compute_node_get_remote_login_settings_options: Additional\n         parameters for the operation\n        :type compute_node_get_remote_login_settings_options:\n         ~azure.batch.models.ComputeNodeGetRemoteLoginSettingsOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ComputeNodeGetRemoteLoginSettingsResult or ClientRawResponse\n         if raw=true\n        :rtype: ~azure.batch.models.ComputeNodeGetRemoteLoginSettingsResult or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if compute_node_get_remote_login_settings_options is not None:
            timeout = compute_node_get_remote_login_settings_options.timeout
        client_request_id = None
        if compute_node_get_remote_login_settings_options is not None:
            client_request_id = compute_node_get_remote_login_settings_options.client_request_id
        return_client_request_id = None
        if compute_node_get_remote_login_settings_options is not None:
            return_client_request_id = compute_node_get_remote_login_settings_options.return_client_request_id
        ocp_date = None
        if compute_node_get_remote_login_settings_options is not None:
            ocp_date = compute_node_get_remote_login_settings_options.ocp_date
        url = self.get_remote_login_settings.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str'), 'nodeId': self._serialize.url('node_id', node_id, 'str')}
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
            deserialized = self._deserialize('ComputeNodeGetRemoteLoginSettingsResult', response)
            header_dict = {'client-request-id': 'str', 'request-id': 'str', 'ETag': 'str', 'Last-Modified': 'rfc-1123'}
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            client_raw_response.add_headers(header_dict)
            return client_raw_response
        return deserialized
    get_remote_login_settings.metadata = {'url': '/pools/{poolId}/nodes/{nodeId}/remoteloginsettings'}

    def get_remote_desktop(self, pool_id, node_id, compute_node_get_remote_desktop_options=None, custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Gets the Remote Desktop Protocol file for the specified Compute Node.\n\n        Before you can access a Compute Node by using the RDP file, you must\n        create a user Account on the Compute Node. This API can only be invoked\n        on Pools created with a cloud service configuration. For Pools created\n        with a virtual machine configuration, see the GetRemoteLoginSettings\n        API.\n\n        :param pool_id: The ID of the Pool that contains the Compute Node.\n        :type pool_id: str\n        :param node_id: The ID of the Compute Node for which you want to get\n         the Remote Desktop Protocol file.\n        :type node_id: str\n        :param compute_node_get_remote_desktop_options: Additional parameters\n         for the operation\n        :type compute_node_get_remote_desktop_options:\n         ~azure.batch.models.ComputeNodeGetRemoteDesktopOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: object or ClientRawResponse if raw=true\n        :rtype: Generator or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if compute_node_get_remote_desktop_options is not None:
            timeout = compute_node_get_remote_desktop_options.timeout
        client_request_id = None
        if compute_node_get_remote_desktop_options is not None:
            client_request_id = compute_node_get_remote_desktop_options.client_request_id
        return_client_request_id = None
        if compute_node_get_remote_desktop_options is not None:
            return_client_request_id = compute_node_get_remote_desktop_options.return_client_request_id
        ocp_date = None
        if compute_node_get_remote_desktop_options is not None:
            ocp_date = compute_node_get_remote_desktop_options.ocp_date
        url = self.get_remote_desktop.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str'), 'nodeId': self._serialize.url('node_id', node_id, 'str')}
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
    get_remote_desktop.metadata = {'url': '/pools/{poolId}/nodes/{nodeId}/rdp'}

    def upload_batch_service_logs(self, pool_id, node_id, upload_batch_service_logs_configuration, compute_node_upload_batch_service_logs_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Upload Azure Batch service log files from the specified Compute Node to\n        Azure Blob Storage.\n\n        This is for gathering Azure Batch service log files in an automated\n        fashion from Compute Nodes if you are experiencing an error and wish to\n        escalate to Azure support. The Azure Batch service log files should be\n        shared with Azure support to aid in debugging issues with the Batch\n        service.\n\n        :param pool_id: The ID of the Pool that contains the Compute Node.\n        :type pool_id: str\n        :param node_id: The ID of the Compute Node from which you want to\n         upload the Azure Batch service log files.\n        :type node_id: str\n        :param upload_batch_service_logs_configuration: The Azure Batch\n         service log files upload configuration.\n        :type upload_batch_service_logs_configuration:\n         ~azure.batch.models.UploadBatchServiceLogsConfiguration\n        :param compute_node_upload_batch_service_logs_options: Additional\n         parameters for the operation\n        :type compute_node_upload_batch_service_logs_options:\n         ~azure.batch.models.ComputeNodeUploadBatchServiceLogsOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: UploadBatchServiceLogsResult or ClientRawResponse if raw=true\n        :rtype: ~azure.batch.models.UploadBatchServiceLogsResult or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        timeout = None
        if compute_node_upload_batch_service_logs_options is not None:
            timeout = compute_node_upload_batch_service_logs_options.timeout
        client_request_id = None
        if compute_node_upload_batch_service_logs_options is not None:
            client_request_id = compute_node_upload_batch_service_logs_options.client_request_id
        return_client_request_id = None
        if compute_node_upload_batch_service_logs_options is not None:
            return_client_request_id = compute_node_upload_batch_service_logs_options.return_client_request_id
        ocp_date = None
        if compute_node_upload_batch_service_logs_options is not None:
            ocp_date = compute_node_upload_batch_service_logs_options.ocp_date
        url = self.upload_batch_service_logs.metadata['url']
        path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str'), 'nodeId': self._serialize.url('node_id', node_id, 'str')}
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
        body_content = self._serialize.body(upload_batch_service_logs_configuration, 'UploadBatchServiceLogsConfiguration')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.BatchErrorException(self._deserialize, response)
        header_dict = {}
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('UploadBatchServiceLogsResult', response)
            header_dict = {'client-request-id': 'str', 'request-id': 'str'}
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            client_raw_response.add_headers(header_dict)
            return client_raw_response
        return deserialized
    upload_batch_service_logs.metadata = {'url': '/pools/{poolId}/nodes/{nodeId}/uploadbatchservicelogs'}

    def list(self, pool_id, compute_node_list_options=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Lists the Compute Nodes in the specified Pool.\n\n        :param pool_id: The ID of the Pool from which you want to list Compute\n         Nodes.\n        :type pool_id: str\n        :param compute_node_list_options: Additional parameters for the\n         operation\n        :type compute_node_list_options:\n         ~azure.batch.models.ComputeNodeListOptions\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: An iterator like instance of ComputeNode\n        :rtype:\n         ~azure.batch.models.ComputeNodePaged[~azure.batch.models.ComputeNode]\n        :raises:\n         :class:`BatchErrorException<azure.batch.models.BatchErrorException>`\n        '
        filter = None
        if compute_node_list_options is not None:
            filter = compute_node_list_options.filter
        select = None
        if compute_node_list_options is not None:
            select = compute_node_list_options.select
        max_results = None
        if compute_node_list_options is not None:
            max_results = compute_node_list_options.max_results
        timeout = None
        if compute_node_list_options is not None:
            timeout = compute_node_list_options.timeout
        client_request_id = None
        if compute_node_list_options is not None:
            client_request_id = compute_node_list_options.client_request_id
        return_client_request_id = None
        if compute_node_list_options is not None:
            return_client_request_id = compute_node_list_options.return_client_request_id
        ocp_date = None
        if compute_node_list_options is not None:
            ocp_date = compute_node_list_options.ocp_date

        def prepare_request(next_link=None):
            if False:
                return 10
            if not next_link:
                url = self.list.metadata['url']
                path_format_arguments = {'batchUrl': self._serialize.url('self.config.batch_url', self.config.batch_url, 'str', skip_quote=True), 'poolId': self._serialize.url('pool_id', pool_id, 'str')}
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
        deserialized = models.ComputeNodePaged(internal_paging, self._deserialize.dependencies, header_dict)
        return deserialized
    list.metadata = {'url': '/pools/{poolId}/nodes'}