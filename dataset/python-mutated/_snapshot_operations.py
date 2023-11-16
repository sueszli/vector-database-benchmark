from msrest.pipeline import ClientRawResponse
from .. import models

class SnapshotOperations(object):
    """SnapshotOperations operations.

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

    def take(self, type, object_id, apply_scope, user_data=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Submit an operation to take a snapshot of face list, large face list,\n        person group or large person group, with user-specified snapshot type,\n        source object id, apply scope and an optional user data.<br />\n        The snapshot interfaces are for users to backup and restore their face\n        data from one face subscription to another, inside same region or\n        across regions. The workflow contains two phases, user first calls\n        Snapshot - Take to create a copy of the source object and store it as a\n        snapshot, then calls Snapshot - Apply to paste the snapshot to target\n        subscription. The snapshots are stored in a centralized location (per\n        Azure instance), so that they can be applied cross accounts and\n        regions.<br />\n        Taking snapshot is an asynchronous operation. An operation id can be\n        obtained from the "Operation-Location" field in response header, to be\n        used in OperationStatus - Get for tracking the progress of creating the\n        snapshot. The snapshot id will be included in the "resourceLocation"\n        field in OperationStatus - Get response when the operation status is\n        "succeeded".<br />\n        Snapshot taking time depends on the number of person and face entries\n        in the source object. It could be in seconds, or up to several hours\n        for 1,000,000 persons with multiple faces.<br />\n        Snapshots will be automatically expired and cleaned in 48 hours after\n        it is created by Snapshot - Take. User can delete the snapshot using\n        Snapshot - Delete by themselves any time before expiration.<br />\n        Taking snapshot for a certain object will not block any other\n        operations against the object. All read-only operations (Get/List and\n        Identify/FindSimilar/Verify) can be conducted as usual. For all\n        writable operations, including Add/Update/Delete the source object or\n        its persons/faces and Train, they are not blocked but not recommended\n        because writable updates may not be reflected on the snapshot during\n        its taking. After snapshot taking is completed, all readable and\n        writable operations can work as normal. Snapshot will also include the\n        training results of the source object, which means target subscription\n        the snapshot applied to does not need re-train the target object before\n        calling Identify/FindSimilar.<br />\n        * Free-tier subscription quota: 100 take operations per month.\n        * S0-tier subscription quota: 100 take operations per day.\n\n        :param type: User specified type for the source object to take\n         snapshot from. Currently FaceList, PersonGroup, LargeFaceList and\n         LargePersonGroup are supported. Possible values include: \'FaceList\',\n         \'LargeFaceList\', \'LargePersonGroup\', \'PersonGroup\'\n        :type type: str or\n         ~azure.cognitiveservices.vision.face.models.SnapshotObjectType\n        :param object_id: User specified source object id to take snapshot\n         from.\n        :type object_id: str\n        :param apply_scope: User specified array of target Face subscription\n         ids for the snapshot. For each snapshot, only subscriptions included\n         in the applyScope of Snapshot - Take can apply it.\n        :type apply_scope: list[str]\n        :param user_data: User specified data about the snapshot for any\n         purpose. Length should not exceed 16KB.\n        :type user_data: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        body = models.TakeSnapshotRequest(type=type, object_id=object_id, apply_scope=apply_scope, user_data=user_data)
        url = self.take.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(body, 'TakeSnapshotRequest')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [202]:
            raise models.APIErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'Operation-Location': 'str'})
            return client_raw_response
    take.metadata = {'url': '/snapshots'}

    def list(self, type=None, apply_scope=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        "List all accessible snapshots with related information, including\n        snapshots that were taken by the user, or snapshots to be applied to\n        the user (subscription id was included in the applyScope in Snapshot -\n        Take).\n\n        :param type: User specified object type as a search filter. Possible\n         values include: 'FaceList', 'LargeFaceList', 'LargePersonGroup',\n         'PersonGroup'\n        :type type: str or\n         ~azure.cognitiveservices.vision.face.models.SnapshotObjectType\n        :param apply_scope: User specified snapshot apply scopes as a search\n         filter. ApplyScope is an array of the target Azure subscription ids\n         for the snapshot, specified by the user who created the snapshot by\n         Snapshot - Take.\n        :type apply_scope: list[str]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype: list[~azure.cognitiveservices.vision.face.models.Snapshot] or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        "
        url = self.list.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if type is not None:
            query_parameters['type'] = self._serialize.query('type', type, 'SnapshotObjectType')
        if apply_scope is not None:
            query_parameters['applyScope'] = self._serialize.query('apply_scope', apply_scope, '[str]', div=',')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('[Snapshot]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    list.metadata = {'url': '/snapshots'}

    def get(self, snapshot_id, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Retrieve information about a snapshot. Snapshot is only accessible to\n        the source subscription who took it, and target subscriptions included\n        in the applyScope in Snapshot - Take.\n\n        :param snapshot_id: Id referencing a particular snapshot.\n        :type snapshot_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Snapshot or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.face.models.Snapshot or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        url = self.get.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'snapshotId': self._serialize.url('snapshot_id', snapshot_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('Snapshot', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get.metadata = {'url': '/snapshots/{snapshotId}'}

    def update(self, snapshot_id, apply_scope=None, user_data=None, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Update the information of a snapshot. Only the source subscription who\n        took the snapshot can update the snapshot.\n\n        :param snapshot_id: Id referencing a particular snapshot.\n        :type snapshot_id: str\n        :param apply_scope: Array of the target Face subscription ids for the\n         snapshot, specified by the user who created the snapshot when calling\n         Snapshot - Take. For each snapshot, only subscriptions included in the\n         applyScope of Snapshot - Take can apply it.\n        :type apply_scope: list[str]\n        :param user_data: User specified data about the snapshot for any\n         purpose. Length should not exceed 16KB.\n        :type user_data: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        body = models.UpdateSnapshotRequest(apply_scope=apply_scope, user_data=user_data)
        url = self.update.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'snapshotId': self._serialize.url('snapshot_id', snapshot_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(body, 'UpdateSnapshotRequest')
        request = self._client.patch(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    update.metadata = {'url': '/snapshots/{snapshotId}'}

    def delete(self, snapshot_id, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Delete an existing snapshot according to the snapshotId. All object\n        data and information in the snapshot will also be deleted. Only the\n        source subscription who took the snapshot can delete the snapshot. If\n        the user does not delete a snapshot with this API, the snapshot will\n        still be automatically deleted in 48 hours after creation.\n\n        :param snapshot_id: Id referencing a particular snapshot.\n        :type snapshot_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        url = self.delete.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'snapshotId': self._serialize.url('snapshot_id', snapshot_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.delete(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    delete.metadata = {'url': '/snapshots/{snapshotId}'}

    def apply(self, snapshot_id, object_id, mode='CreateNew', custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Submit an operation to apply a snapshot to current subscription. For\n        each snapshot, only subscriptions included in the applyScope of\n        Snapshot - Take can apply it.<br />\n        The snapshot interfaces are for users to backup and restore their face\n        data from one face subscription to another, inside same region or\n        across regions. The workflow contains two phases, user first calls\n        Snapshot - Take to create a copy of the source object and store it as a\n        snapshot, then calls Snapshot - Apply to paste the snapshot to target\n        subscription. The snapshots are stored in a centralized location (per\n        Azure instance), so that they can be applied cross accounts and\n        regions.<br />\n        Applying snapshot is an asynchronous operation. An operation id can be\n        obtained from the "Operation-Location" field in response header, to be\n        used in OperationStatus - Get for tracking the progress of applying the\n        snapshot. The target object id will be included in the\n        "resourceLocation" field in OperationStatus - Get response when the\n        operation status is "succeeded".<br />\n        Snapshot applying time depends on the number of person and face entries\n        in the snapshot object. It could be in seconds, or up to 1 hour for\n        1,000,000 persons with multiple faces.<br />\n        Snapshots will be automatically expired and cleaned in 48 hours after\n        it is created by Snapshot - Take. So the target subscription is\n        required to apply the snapshot in 48 hours since its creation.<br />\n        Applying a snapshot will not block any other operations against the\n        target object, however it is not recommended because the correctness\n        cannot be guaranteed during snapshot applying. After snapshot applying\n        is completed, all operations towards the target object can work as\n        normal. Snapshot also includes the training results of the source\n        object, which means target subscription the snapshot applied to does\n        not need re-train the target object before calling\n        Identify/FindSimilar.<br />\n        One snapshot can be applied multiple times in parallel, while currently\n        only CreateNew apply mode is supported, which means the apply operation\n        will fail if target subscription already contains an object of same\n        type and using the same objectId. Users can specify the "objectId" in\n        request body to avoid such conflicts.<br />\n        * Free-tier subscription quota: 100 apply operations per month.\n        * S0-tier subscription quota: 100 apply operations per day.\n\n        :param snapshot_id: Id referencing a particular snapshot.\n        :type snapshot_id: str\n        :param object_id: User specified target object id to be created from\n         the snapshot.\n        :type object_id: str\n        :param mode: Snapshot applying mode. Currently only CreateNew is\n         supported, which means the apply operation will fail if target\n         subscription already contains an object of same type and using the\n         same objectId. Users can specify the "objectId" in request body to\n         avoid such conflicts. Possible values include: \'CreateNew\'\n        :type mode: str or\n         ~azure.cognitiveservices.vision.face.models.SnapshotApplyMode\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        body = models.ApplySnapshotRequest(object_id=object_id, mode=mode)
        url = self.apply.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'snapshotId': self._serialize.url('snapshot_id', snapshot_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(body, 'ApplySnapshotRequest')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [202]:
            raise models.APIErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'Operation-Location': 'str'})
            return client_raw_response
    apply.metadata = {'url': '/snapshots/{snapshotId}/apply'}

    def get_operation_status(self, operation_id, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Retrieve the status of a take/apply snapshot operation.\n\n        :param operation_id: Id referencing a particular take/apply snapshot\n         operation.\n        :type operation_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: OperationStatus or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.face.models.OperationStatus or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        url = self.get_operation_status.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'operationId': self._serialize.url('operation_id', operation_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('OperationStatus', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_operation_status.metadata = {'url': '/operations/{operationId}'}