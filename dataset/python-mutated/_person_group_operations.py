from msrest.pipeline import ClientRawResponse
from .. import models

class PersonGroupOperations(object):
    """PersonGroupOperations operations.

    You should not instantiate directly this class, but create a Client instance that will create it for you and attach it as attribute.

    :param client: Client for service requests.
    :param config: Configuration of service client.
    :param serializer: An object model serializer.
    :param deserializer: An object model deserializer.
    """
    models = models

    def __init__(self, client, config, serializer, deserializer):
        if False:
            for i in range(10):
                print('nop')
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self.config = config

    def create(self, person_group_id, name, user_data=None, recognition_model='recognition_01', custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        "Create a new person group with specified personGroupId, name,\n        user-provided userData and recognitionModel.\n        <br /> A person group is the container of the uploaded person data,\n        including face recognition features.\n        <br /> After creation, use [PersonGroup Person -\n        Create](https://docs.microsoft.com/rest/api/faceapi/persongroupperson/create)\n        to add persons into the group, and then call [PersonGroup -\n        Train](https://docs.microsoft.com/rest/api/faceapi/persongroup/train)\n        to get this group ready for [Face -\n        Identify](https://docs.microsoft.com/rest/api/faceapi/face/identify).\n        <br /> No image will be stored. Only the person's extracted face\n        features and userData will be stored on server until [PersonGroup\n        Person -\n        Delete](https://docs.microsoft.com/rest/api/faceapi/persongroupperson/delete)\n        or [PersonGroup -\n        Delete](https://docs.microsoft.com/rest/api/faceapi/persongroup/delete)\n        is called.\n        <br/>'recognitionModel' should be specified to associate with this\n        person group. The default value for 'recognitionModel' is\n        'recognition_01', if the latest model needed, please explicitly specify\n        the model you need in this parameter. New faces that are added to an\n        existing person group will use the recognition model that's already\n        associated with the collection. Existing face features in a person\n        group can't be updated to features extracted by another version of\n        recognition model.\n        Person group quota:\n        * Free-tier subscription quota: 1,000 person groups. Each holds up to\n        1,000 persons.\n        * S0-tier subscription quota: 1,000,000 person groups. Each holds up to\n        10,000 persons.\n        * to handle larger scale face identification problem, please consider\n        using\n        [LargePersonGroup](https://docs.microsoft.com/rest/api/faceapi/largepersongroup).\n\n        :param person_group_id: Id referencing a particular person group.\n        :type person_group_id: str\n        :param name: User defined name, maximum length is 128.\n        :type name: str\n        :param user_data: User specified data. Length should not exceed 16KB.\n        :type user_data: str\n        :param recognition_model: Possible values include: 'recognition_01',\n         'recognition_02', 'recognition_03', 'recognition_04'\n        :type recognition_model: str or\n         ~azure.cognitiveservices.vision.face.models.RecognitionModel\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        "
        body = models.MetaDataContract(name=name, user_data=user_data, recognition_model=recognition_model)
        url = self.create.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'personGroupId': self._serialize.url('person_group_id', person_group_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
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
    create.metadata = {'url': '/persongroups/{personGroupId}'}

    def delete(self, person_group_id, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Delete an existing person group. Persisted face features of all people\n        in the person group will also be deleted.\n\n        :param person_group_id: Id referencing a particular person group.\n        :type person_group_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        url = self.delete.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'personGroupId': self._serialize.url('person_group_id', person_group_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
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
    delete.metadata = {'url': '/persongroups/{personGroupId}'}

    def get(self, person_group_id, return_recognition_model=False, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        "Retrieve person group name, userData and recognitionModel. To get\n        person information under this personGroup, use [PersonGroup Person -\n        List](https://docs.microsoft.com/rest/api/faceapi/persongroupperson/list).\n\n        :param person_group_id: Id referencing a particular person group.\n        :type person_group_id: str\n        :param return_recognition_model: A value indicating whether the\n         operation should return 'recognitionModel' in response.\n        :type return_recognition_model: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: PersonGroup or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.face.models.PersonGroup or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        "
        url = self.get.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'personGroupId': self._serialize.url('person_group_id', person_group_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
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
            deserialized = self._deserialize('PersonGroup', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get.metadata = {'url': '/persongroups/{personGroupId}'}

    def update(self, person_group_id, name=None, user_data=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        "Update an existing person group's display name and userData. The\n        properties which does not appear in request body will not be updated.\n\n        :param person_group_id: Id referencing a particular person group.\n        :type person_group_id: str\n        :param name: User defined name, maximum length is 128.\n        :type name: str\n        :param user_data: User specified data. Length should not exceed 16KB.\n        :type user_data: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        "
        body = models.NameAndUserDataContract(name=name, user_data=user_data)
        url = self.update.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'personGroupId': self._serialize.url('person_group_id', person_group_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
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
    update.metadata = {'url': '/persongroups/{personGroupId}'}

    def get_training_status(self, person_group_id, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Retrieve the training status of a person group (completed or ongoing).\n\n        :param person_group_id: Id referencing a particular person group.\n        :type person_group_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: TrainingStatus or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.face.models.TrainingStatus or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        url = self.get_training_status.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'personGroupId': self._serialize.url('person_group_id', person_group_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
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
    get_training_status.metadata = {'url': '/persongroups/{personGroupId}/training'}

    def list(self, start=None, top=1000, return_recognition_model=False, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'List person groups’ personGroupId, name, userData and\n        recognitionModel.<br />\n        * Person groups are stored in alphabetical order of personGroupId.\n        * "start" parameter (string, optional) is a user-provided personGroupId\n        value that returned entries have larger ids by string comparison.\n        "start" set to empty to indicate return from the first item.\n        * "top" parameter (int, optional) specifies the number of entries to\n        return. A maximal of 1000 entries can be returned in one call. To fetch\n        more, you can specify "start" with the last returned entry’s Id of the\n        current call.\n        <br />\n        For example, total 5 person groups: "group1", ..., "group5".\n        <br /> "start=&top=" will return all 5 groups.\n        <br /> "start=&top=2" will return "group1", "group2".\n        <br /> "start=group2&top=3" will return "group3", "group4", "group5".\n        .\n\n        :param start: List person groups from the least personGroupId greater\n         than the "start".\n        :type start: str\n        :param top: The number of person groups to list.\n        :type top: int\n        :param return_recognition_model: A value indicating whether the\n         operation should return \'recognitionModel\' in response.\n        :type return_recognition_model: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype: list[~azure.cognitiveservices.vision.face.models.PersonGroup]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
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
            deserialized = self._deserialize('[PersonGroup]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    list.metadata = {'url': '/persongroups'}

    def train(self, person_group_id, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Queue a person group training task, the training task may not be\n        started immediately.\n\n        :param person_group_id: Id referencing a particular person group.\n        :type person_group_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        url = self.train.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'personGroupId': self._serialize.url('person_group_id', person_group_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
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
    train.metadata = {'url': '/persongroups/{personGroupId}/train'}