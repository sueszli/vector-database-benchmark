from msrest.pipeline import ClientRawResponse
from .. import models

class LargeFaceListOperations(object):
    """LargeFaceListOperations operations.

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

    def create(self, large_face_list_id, name, user_data=None, recognition_model='recognition_01', custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        "Create an empty large face list with user-specified largeFaceListId,\n        name, an optional userData and recognitionModel.\n        <br /> Large face list is a list of faces, up to 1,000,000 faces, and\n        used by [Face - Find\n        Similar](https://docs.microsoft.com/rest/api/faceapi/face/findsimilar).\n        <br /> After creation, user should use [LargeFaceList Face -\n        Add](https://docs.microsoft.com/rest/api/faceapi/largefacelist/addfacefromurl)\n        to import the faces and [LargeFaceList -\n        Train](https://docs.microsoft.com/rest/api/faceapi/largefacelist/train)\n        to make it ready for [Face - Find\n        Similar](https://docs.microsoft.com/rest/api/faceapi/face/findsimilar).\n        No image will be stored. Only the extracted face features are stored on\n        server until [LargeFaceList -\n        Delete](https://docs.microsoft.com/rest/api/faceapi/largefacelist/delete)\n        is called.\n        <br /> Find Similar is used for scenario like finding celebrity-like\n        faces, similar face filtering, or as a light way face identification.\n        But if the actual use is to identify person, please use\n        [PersonGroup](https://docs.microsoft.com/rest/api/faceapi/persongroup)\n        /\n        [LargePersonGroup](https://docs.microsoft.com/rest/api/faceapi/largepersongroup)\n        and [Face -\n        Identify](https://docs.microsoft.com/rest/api/faceapi/face/identify).\n        <br/>'recognitionModel' should be specified to associate with this\n        large face list. The default value for 'recognitionModel' is\n        'recognition_01', if the latest model needed, please explicitly specify\n        the model you need in this parameter. New faces that are added to an\n        existing large face list will use the recognition model that's already\n        associated with the collection. Existing face features in a large face\n        list can't be updated to features extracted by another version of\n        recognition model. Please refer to [Specify a recognition\n        model](https://docs.microsoft.com/azure/cognitive-services/face/face-api-how-to-topics/specify-recognition-model).\n        Large face list quota:\n        * Free-tier subscription quota: 64 large face lists.\n        * S0-tier subscription quota: 1,000,000 large face lists.\n\n        :param large_face_list_id: Id referencing a particular large face\n         list.\n        :type large_face_list_id: str\n        :param name: User defined name, maximum length is 128.\n        :type name: str\n        :param user_data: User specified data. Length should not exceed 16KB.\n        :type user_data: str\n        :param recognition_model: Possible values include: 'recognition_01',\n         'recognition_02', 'recognition_03', 'recognition_04'\n        :type recognition_model: str or\n         ~azure.cognitiveservices.vision.face.models.RecognitionModel\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        "
        body = models.MetaDataContract(name=name, user_data=user_data, recognition_model=recognition_model)
        url = self.create.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largeFaceListId': self._serialize.url('large_face_list_id', large_face_list_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
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
    create.metadata = {'url': '/largefacelists/{largeFaceListId}'}

    def get(self, large_face_list_id, return_recognition_model=False, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        "Retrieve a large face list’s largeFaceListId, name, userData and\n        recognitionModel.\n\n        :param large_face_list_id: Id referencing a particular large face\n         list.\n        :type large_face_list_id: str\n        :param return_recognition_model: A value indicating whether the\n         operation should return 'recognitionModel' in response.\n        :type return_recognition_model: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: LargeFaceList or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.face.models.LargeFaceList or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        "
        url = self.get.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largeFaceListId': self._serialize.url('large_face_list_id', large_face_list_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
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
            deserialized = self._deserialize('LargeFaceList', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get.metadata = {'url': '/largefacelists/{largeFaceListId}'}

    def update(self, large_face_list_id, name=None, user_data=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Update information of a large face list.\n\n        :param large_face_list_id: Id referencing a particular large face\n         list.\n        :type large_face_list_id: str\n        :param name: User defined name, maximum length is 128.\n        :type name: str\n        :param user_data: User specified data. Length should not exceed 16KB.\n        :type user_data: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        body = models.NameAndUserDataContract(name=name, user_data=user_data)
        url = self.update.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largeFaceListId': self._serialize.url('large_face_list_id', large_face_list_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
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
    update.metadata = {'url': '/largefacelists/{largeFaceListId}'}

    def delete(self, large_face_list_id, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Delete a specified large face list.\n\n        :param large_face_list_id: Id referencing a particular large face\n         list.\n        :type large_face_list_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        url = self.delete.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largeFaceListId': self._serialize.url('large_face_list_id', large_face_list_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
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
    delete.metadata = {'url': '/largefacelists/{largeFaceListId}'}

    def get_training_status(self, large_face_list_id, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Retrieve the training status of a large face list (completed or\n        ongoing).\n\n        :param large_face_list_id: Id referencing a particular large face\n         list.\n        :type large_face_list_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: TrainingStatus or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.face.models.TrainingStatus or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        url = self.get_training_status.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largeFaceListId': self._serialize.url('large_face_list_id', large_face_list_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
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
    get_training_status.metadata = {'url': '/largefacelists/{largeFaceListId}/training'}

    def list(self, return_recognition_model=False, start=None, top=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'List large face lists’ information of largeFaceListId, name, userData\n        and recognitionModel. <br />\n        To get face information inside largeFaceList use [LargeFaceList Face -\n        Get](https://docs.microsoft.com/rest/api/faceapi/largefacelist/getface)<br\n        />\n        * Large face lists are stored in alphabetical order of largeFaceListId.\n        * "start" parameter (string, optional) is a user-provided\n        largeFaceListId value that returned entries have larger ids by string\n        comparison. "start" set to empty to indicate return from the first\n        item.\n        * "top" parameter (int, optional) specifies the number of entries to\n        return. A maximal of 1000 entries can be returned in one call. To fetch\n        more, you can specify "start" with the last returned entry’s Id of the\n        current call.\n        <br />\n        For example, total 5 large person lists: "list1", ..., "list5".\n        <br /> "start=&top=" will return all 5 lists.\n        <br /> "start=&top=2" will return "list1", "list2".\n        <br /> "start=list2&top=3" will return "list3", "list4", "list5".\n        .\n\n        :param return_recognition_model: A value indicating whether the\n         operation should return \'recognitionModel\' in response.\n        :type return_recognition_model: bool\n        :param start: Starting large face list id to return (used to list a\n         range of large face lists).\n        :type start: str\n        :param top: Number of large face lists to return starting with the\n         large face list id indicated by the \'start\' parameter.\n        :type top: int\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.vision.face.models.LargeFaceList] or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        url = self.list.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if return_recognition_model is not None:
            query_parameters['returnRecognitionModel'] = self._serialize.query('return_recognition_model', return_recognition_model, 'bool')
        if start is not None:
            query_parameters['start'] = self._serialize.query('start', start, 'str')
        if top is not None:
            query_parameters['top'] = self._serialize.query('top', top, 'int', maximum=1000, minimum=1)
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
            deserialized = self._deserialize('[LargeFaceList]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    list.metadata = {'url': '/largefacelists'}

    def train(self, large_face_list_id, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Queue a large face list training task, the training task may not be\n        started immediately.\n\n        :param large_face_list_id: Id referencing a particular large face\n         list.\n        :type large_face_list_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        url = self.train.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largeFaceListId': self._serialize.url('large_face_list_id', large_face_list_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
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
    train.metadata = {'url': '/largefacelists/{largeFaceListId}/train'}

    def delete_face(self, large_face_list_id, persisted_face_id, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Delete a face from a large face list by specified largeFaceListId and\n        persistedFaceId.\n        <br /> Adding/deleting faces to/from a same large face list are\n        processed sequentially and to/from different large face lists are in\n        parallel.\n\n        :param large_face_list_id: Id referencing a particular large face\n         list.\n        :type large_face_list_id: str\n        :param persisted_face_id: Id referencing a particular persistedFaceId\n         of an existing face.\n        :type persisted_face_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        url = self.delete_face.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largeFaceListId': self._serialize.url('large_face_list_id', large_face_list_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$'), 'persistedFaceId': self._serialize.url('persisted_face_id', persisted_face_id, 'str')}
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
    delete_face.metadata = {'url': '/largefacelists/{largeFaceListId}/persistedfaces/{persistedFaceId}'}

    def get_face(self, large_face_list_id, persisted_face_id, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Retrieve information about a persisted face (specified by\n        persistedFaceId and its belonging largeFaceListId).\n\n        :param large_face_list_id: Id referencing a particular large face\n         list.\n        :type large_face_list_id: str\n        :param persisted_face_id: Id referencing a particular persistedFaceId\n         of an existing face.\n        :type persisted_face_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: PersistedFace or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.face.models.PersistedFace or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        url = self.get_face.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largeFaceListId': self._serialize.url('large_face_list_id', large_face_list_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$'), 'persistedFaceId': self._serialize.url('persisted_face_id', persisted_face_id, 'str')}
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
            deserialized = self._deserialize('PersistedFace', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_face.metadata = {'url': '/largefacelists/{largeFaceListId}/persistedfaces/{persistedFaceId}'}

    def update_face(self, large_face_list_id, persisted_face_id, user_data=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        "Update a persisted face's userData field.\n\n        :param large_face_list_id: Id referencing a particular large face\n         list.\n        :type large_face_list_id: str\n        :param persisted_face_id: Id referencing a particular persistedFaceId\n         of an existing face.\n        :type persisted_face_id: str\n        :param user_data: User-provided data attached to the face. The size\n         limit is 1KB.\n        :type user_data: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        "
        body = models.UpdateFaceRequest(user_data=user_data)
        url = self.update_face.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largeFaceListId': self._serialize.url('large_face_list_id', large_face_list_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$'), 'persistedFaceId': self._serialize.url('persisted_face_id', persisted_face_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(body, 'UpdateFaceRequest')
        request = self._client.patch(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    update_face.metadata = {'url': '/largefacelists/{largeFaceListId}/persistedfaces/{persistedFaceId}'}

    def add_face_from_url(self, large_face_list_id, url, user_data=None, target_face=None, detection_model='detection_01', custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Add a face to a specified large face list, up to 1,000,000 faces.\n        <br /> To deal with an image contains multiple faces, input face can be\n        specified as an image with a targetFace rectangle. It returns a\n        persistedFaceId representing the added face. No image will be stored.\n        Only the extracted face feature will be stored on server until\n        [LargeFaceList Face -\n        Delete](https://docs.microsoft.com/rest/api/faceapi/largefacelist/deleteface)\n        or [LargeFaceList -\n        Delete](https://docs.microsoft.com/rest/api/faceapi/largefacelist/delete)\n        is called.\n        <br /> Note persistedFaceId is different from faceId generated by [Face\n        -\n        Detect](https://docs.microsoft.com/rest/api/faceapi/face/detectwithurl).\n        * Higher face image quality means better recognition precision. Please\n        consider high-quality faces: frontal, clear, and face size is 200x200\n        pixels (100 pixels between eyes) or bigger.\n        * JPEG, PNG, GIF (the first frame), and BMP format are supported. The\n        allowed image file size is from 1KB to 6MB.\n        * "targetFace" rectangle should contain one face. Zero or multiple\n        faces will be regarded as an error. If the provided "targetFace"\n        rectangle is not returned from [Face -\n        Detect](https://docs.microsoft.com/rest/api/faceapi/face/detectwithurl),\n        there’s no guarantee to detect and add the face successfully.\n        * Out of detectable face size (36x36 - 4096x4096 pixels), large\n        head-pose, or large occlusions will cause failures.\n        * Adding/deleting faces to/from a same face list are processed\n        sequentially and to/from different face lists are in parallel.\n        * The minimum detectable face size is 36x36 pixels in an image no\n        larger than 1920x1080 pixels. Images with dimensions higher than\n        1920x1080 pixels will need a proportionally larger minimum face size.\n        * Different \'detectionModel\' values can be provided. To use and compare\n        different detection models, please refer to [How to specify a detection\n        model](https://docs.microsoft.com/azure/cognitive-services/face/face-api-how-to-topics/specify-detection-model)\n        Quota:\n        * Free-tier subscription quota: 1,000 faces per large face list.\n        * S0-tier subscription quota: 1,000,000 faces per large face list.\n\n        :param large_face_list_id: Id referencing a particular large face\n         list.\n        :type large_face_list_id: str\n        :param url: Publicly reachable URL of an image\n        :type url: str\n        :param user_data: User-specified data about the face for any purpose.\n         The maximum length is 1KB.\n        :type user_data: str\n        :param target_face: A face rectangle to specify the target face to be\n         added to a person in the format of "targetFace=left,top,width,height".\n         E.g. "targetFace=10,10,100,100". If there is more than one face in the\n         image, targetFace is required to specify which face to add. No\n         targetFace means there is only one face detected in the entire image.\n        :type target_face: list[int]\n        :param detection_model: Name of detection model. Detection model is\n         used to detect faces in the submitted image. A detection model name\n         can be provided when performing Face - Detect or (Large)FaceList - Add\n         Face or (Large)PersonGroup - Add Face. The default value is\n         \'detection_01\', if another model is needed, please explicitly specify\n         it. Possible values include: \'detection_01\', \'detection_02\',\n         \'detection_03\'\n        :type detection_model: str or\n         ~azure.cognitiveservices.vision.face.models.DetectionModel\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: PersistedFace or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.face.models.PersistedFace or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        image_url = models.ImageUrl(url=url)
        url = self.add_face_from_url.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largeFaceListId': self._serialize.url('large_face_list_id', large_face_list_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if user_data is not None:
            query_parameters['userData'] = self._serialize.query('user_data', user_data, 'str', max_length=1024)
        if target_face is not None:
            query_parameters['targetFace'] = self._serialize.query('target_face', target_face, '[int]', div=',')
        if detection_model is not None:
            query_parameters['detectionModel'] = self._serialize.query('detection_model', detection_model, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(image_url, 'ImageUrl')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('PersistedFace', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    add_face_from_url.metadata = {'url': '/largefacelists/{largeFaceListId}/persistedfaces'}

    def list_faces(self, large_face_list_id, start=None, top=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        "List all faces in a large face list, and retrieve face information\n        (including userData and persistedFaceIds of registered faces of the\n        face).\n\n        :param large_face_list_id: Id referencing a particular large face\n         list.\n        :type large_face_list_id: str\n        :param start: Starting face id to return (used to list a range of\n         faces).\n        :type start: str\n        :param top: Number of faces to return starting with the face id\n         indicated by the 'start' parameter.\n        :type top: int\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.vision.face.models.PersistedFace] or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        "
        url = self.list_faces.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largeFaceListId': self._serialize.url('large_face_list_id', large_face_list_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if start is not None:
            query_parameters['start'] = self._serialize.query('start', start, 'str')
        if top is not None:
            query_parameters['top'] = self._serialize.query('top', top, 'int', maximum=1000, minimum=1)
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
            deserialized = self._deserialize('[PersistedFace]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    list_faces.metadata = {'url': '/largefacelists/{largeFaceListId}/persistedfaces'}

    def add_face_from_stream(self, large_face_list_id, image, user_data=None, target_face=None, detection_model='detection_01', custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            while True:
                i = 10
        'Add a face to a specified large face list, up to 1,000,000 faces.\n        <br /> To deal with an image contains multiple faces, input face can be\n        specified as an image with a targetFace rectangle. It returns a\n        persistedFaceId representing the added face. No image will be stored.\n        Only the extracted face feature will be stored on server until\n        [LargeFaceList Face -\n        Delete](https://docs.microsoft.com/rest/api/faceapi/largefacelist/deleteface)\n        or [LargeFaceList -\n        Delete](https://docs.microsoft.com/rest/api/faceapi/largefacelist/delete)\n        is called.\n        <br /> Note persistedFaceId is different from faceId generated by [Face\n        -\n        Detect](https://docs.microsoft.com/rest/api/faceapi/face/detectwithurl).\n        * Higher face image quality means better recognition precision. Please\n        consider high-quality faces: frontal, clear, and face size is 200x200\n        pixels (100 pixels between eyes) or bigger.\n        * JPEG, PNG, GIF (the first frame), and BMP format are supported. The\n        allowed image file size is from 1KB to 6MB.\n        * "targetFace" rectangle should contain one face. Zero or multiple\n        faces will be regarded as an error. If the provided "targetFace"\n        rectangle is not returned from [Face -\n        Detect](https://docs.microsoft.com/rest/api/faceapi/face/detectwithurl),\n        there’s no guarantee to detect and add the face successfully.\n        * Out of detectable face size (36x36 - 4096x4096 pixels), large\n        head-pose, or large occlusions will cause failures.\n        * Adding/deleting faces to/from a same face list are processed\n        sequentially and to/from different face lists are in parallel.\n        * The minimum detectable face size is 36x36 pixels in an image no\n        larger than 1920x1080 pixels. Images with dimensions higher than\n        1920x1080 pixels will need a proportionally larger minimum face size.\n        * Different \'detectionModel\' values can be provided. To use and compare\n        different detection models, please refer to [How to specify a detection\n        model](https://docs.microsoft.com/azure/cognitive-services/face/face-api-how-to-topics/specify-detection-model).\n        Quota:\n        * Free-tier subscription quota: 1,000 faces per large face list.\n        * S0-tier subscription quota: 1,000,000 faces per large face list.\n\n        :param large_face_list_id: Id referencing a particular large face\n         list.\n        :type large_face_list_id: str\n        :param image: An image stream.\n        :type image: Generator\n        :param user_data: User-specified data about the face for any purpose.\n         The maximum length is 1KB.\n        :type user_data: str\n        :param target_face: A face rectangle to specify the target face to be\n         added to a person in the format of "targetFace=left,top,width,height".\n         E.g. "targetFace=10,10,100,100". If there is more than one face in the\n         image, targetFace is required to specify which face to add. No\n         targetFace means there is only one face detected in the entire image.\n        :type target_face: list[int]\n        :param detection_model: Name of detection model. Detection model is\n         used to detect faces in the submitted image. A detection model name\n         can be provided when performing Face - Detect or (Large)FaceList - Add\n         Face or (Large)PersonGroup - Add Face. The default value is\n         \'detection_01\', if another model is needed, please explicitly specify\n         it. Possible values include: \'detection_01\', \'detection_02\',\n         \'detection_03\'\n        :type detection_model: str or\n         ~azure.cognitiveservices.vision.face.models.DetectionModel\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: PersistedFace or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.face.models.PersistedFace or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        url = self.add_face_from_stream.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largeFaceListId': self._serialize.url('large_face_list_id', large_face_list_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if user_data is not None:
            query_parameters['userData'] = self._serialize.query('user_data', user_data, 'str', max_length=1024)
        if target_face is not None:
            query_parameters['targetFace'] = self._serialize.query('target_face', target_face, '[int]', div=',')
        if detection_model is not None:
            query_parameters['detectionModel'] = self._serialize.query('detection_model', detection_model, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/octet-stream'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._client.stream_upload(image, callback)
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('PersistedFace', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    add_face_from_stream.metadata = {'url': '/largefacelists/{largeFaceListId}/persistedfaces'}