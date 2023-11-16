from msrest.pipeline import ClientRawResponse
from .. import models

class LargePersonGroupOperations(object):
    """LargePersonGroupOperations operations.

    You should not instantiate directly this class, but create a Client instance that will create it for you and attach it as attribute.

    :param client: Client for service requests.
    :param config: Configuration of service client.
    :param serializer: An object model serializer.
    :param deserializer: An object model deserializer.
    """
    models = models

    def __init__(self, client, config, serializer, deserializer):
        if False:
            return 10
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self.config = config

    def create(self, large_person_group_id, name, user_data=None, recognition_model='recognition_01', custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        "Create a new large person group with user-specified largePersonGroupId,\n        name, an optional userData and recognitionModel.\n        <br /> A large person group is the container of the uploaded person\n        data, including face recognition feature, and up to 1,000,000\n        people.\n        <br /> After creation, use [LargePersonGroup Person -\n        Create](https://docs.microsoft.com/rest/api/faceapi/largepersongroupperson/create)\n        to add person into the group, and call [LargePersonGroup -\n        Train](https://docs.microsoft.com/rest/api/faceapi/largepersongroup/train)\n        to get this group ready for [Face -\n        Identify](https://docs.microsoft.com/rest/api/faceapi/face/identify).\n        <br /> No image will be stored. Only the person's extracted face\n        features and userData will be stored on server until [LargePersonGroup\n        Person -\n        Delete](https://docs.microsoft.com/rest/api/faceapi/largepersongroupperson/delete)\n        or [LargePersonGroup -\n        Delete](https://docs.microsoft.com/rest/api/faceapi/largepersongroup/delete)\n        is called.\n        <br/>'recognitionModel' should be specified to associate with this\n        large person group. The default value for 'recognitionModel' is\n        'recognition_01', if the latest model needed, please explicitly specify\n        the model you need in this parameter. New faces that are added to an\n        existing large person group will use the recognition model that's\n        already associated with the collection. Existing face features in a\n        large person group can't be updated to features extracted by another\n        version of recognition model. Please refer to [Specify a face\n        recognition\n        model](https://docs.microsoft.com/azure/cognitive-services/face/face-api-how-to-topics/specify-recognition-model).\n        Large person group quota:\n        * Free-tier subscription quota: 1,000 large person groups.\n        * S0-tier subscription quota: 1,000,000 large person groups.\n\n        :param large_person_group_id: Id referencing a particular large person\n         group.\n        :type large_person_group_id: str\n        :param name: User defined name, maximum length is 128.\n        :type name: str\n        :param user_data: User specified data. Length should not exceed 16KB.\n        :type user_data: str\n        :param recognition_model: Possible values include: 'recognition_01',\n         'recognition_02', 'recognition_03', 'recognition_04'\n        :type recognition_model: str or\n         ~azure.cognitiveservices.vision.face.models.RecognitionModel\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        "
        body = models.MetaDataContract(name=name, user_data=user_data, recognition_model=recognition_model)
        url = self.create.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largePersonGroupId': self._serialize.url('large_person_group_id', large_person_group_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(body, 'MetaDataContract')
        request = self._client.put(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    create.metadata = {'url': '/largepersongroups/{largePersonGroupId}'}

    def delete(self, large_person_group_id, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Delete an existing large person group. Persisted face features of all\n        people in the large person group will also be deleted.\n\n        :param large_person_group_id: Id referencing a particular large person\n         group.\n        :type large_person_group_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        url = self.delete.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largePersonGroupId': self._serialize.url('large_person_group_id', large_person_group_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
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
    delete.metadata = {'url': '/largepersongroups/{largePersonGroupId}'}

    def get(self, large_person_group_id, return_recognition_model=False, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        "Retrieve the information of a large person group, including its name,\n        userData and recognitionModel. This API returns large person group\n        information only, use [LargePersonGroup Person -\n        List](https://docs.microsoft.com/rest/api/faceapi/largepersongroupperson/list)\n        instead to retrieve person information under the large person group.\n        .\n\n        :param large_person_group_id: Id referencing a particular large person\n         group.\n        :type large_person_group_id: str\n        :param return_recognition_model: A value indicating whether the\n         operation should return 'recognitionModel' in response.\n        :type return_recognition_model: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: LargePersonGroup or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.face.models.LargePersonGroup\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        "
        url = self.get.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largePersonGroupId': self._serialize.url('large_person_group_id', large_person_group_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if return_recognition_model is not None:
            query_parameters['returnRecognitionModel'] = self._serialize.query('return_recognition_model', return_recognition_model, 'bool')
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
            deserialized = self._deserialize('LargePersonGroup', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get.metadata = {'url': '/largepersongroups/{largePersonGroupId}'}

    def update(self, large_person_group_id, name=None, user_data=None, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        "Update an existing large person group's display name and userData. The\n        properties which does not appear in request body will not be updated.\n\n        :param large_person_group_id: Id referencing a particular large person\n         group.\n        :type large_person_group_id: str\n        :param name: User defined name, maximum length is 128.\n        :type name: str\n        :param user_data: User specified data. Length should not exceed 16KB.\n        :type user_data: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        "
        body = models.NameAndUserDataContract(name=name, user_data=user_data)
        url = self.update.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largePersonGroupId': self._serialize.url('large_person_group_id', large_person_group_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(body, 'NameAndUserDataContract')
        request = self._client.patch(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    update.metadata = {'url': '/largepersongroups/{largePersonGroupId}'}

    def get_training_status(self, large_person_group_id, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Retrieve the training status of a large person group (completed or\n        ongoing).\n\n        :param large_person_group_id: Id referencing a particular large person\n         group.\n        :type large_person_group_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: TrainingStatus or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.face.models.TrainingStatus or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        url = self.get_training_status.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largePersonGroupId': self._serialize.url('large_person_group_id', large_person_group_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
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
            deserialized = self._deserialize('TrainingStatus', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_training_status.metadata = {'url': '/largepersongroups/{largePersonGroupId}/training'}

    def list(self, start=None, top=1000, return_recognition_model=False, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'List all existing large person groups’ largePersonGroupId, name,\n        userData and recognitionModel.<br />\n        * Large person groups are stored in alphabetical order of\n        largePersonGroupId.\n        * "start" parameter (string, optional) is a user-provided\n        largePersonGroupId value that returned entries have larger ids by\n        string comparison. "start" set to empty to indicate return from the\n        first item.\n        * "top" parameter (int, optional) specifies the number of entries to\n        return. A maximal of 1000 entries can be returned in one call. To fetch\n        more, you can specify "start" with the last returned entry’s Id of the\n        current call.\n        <br />\n        For example, total 5 large person groups: "group1", ..., "group5".\n        <br /> "start=&top=" will return all 5 groups.\n        <br /> "start=&top=2" will return "group1", "group2".\n        <br /> "start=group2&top=3" will return "group3", "group4", "group5".\n        .\n\n        :param start: List large person groups from the least\n         largePersonGroupId greater than the "start".\n        :type start: str\n        :param top: The number of large person groups to list.\n        :type top: int\n        :param return_recognition_model: A value indicating whether the\n         operation should return \'recognitionModel\' in response.\n        :type return_recognition_model: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.vision.face.models.LargePersonGroup] or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        url = self.list.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if start is not None:
            query_parameters['start'] = self._serialize.query('start', start, 'str', max_length=64)
        if top is not None:
            query_parameters['top'] = self._serialize.query('top', top, 'int', maximum=1000, minimum=1)
        if return_recognition_model is not None:
            query_parameters['returnRecognitionModel'] = self._serialize.query('return_recognition_model', return_recognition_model, 'bool')
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
            deserialized = self._deserialize('[LargePersonGroup]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    list.metadata = {'url': '/largepersongroups'}

    def train(self, large_person_group_id, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Queue a large person group training task, the training task may not be\n        started immediately.\n\n        :param large_person_group_id: Id referencing a particular large person\n         group.\n        :type large_person_group_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        url = self.train.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largePersonGroupId': self._serialize.url('large_person_group_id', large_person_group_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.post(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [202]:
            raise models.APIErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    train.metadata = {'url': '/largepersongroups/{largePersonGroupId}/train'}