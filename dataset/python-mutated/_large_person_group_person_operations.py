from msrest.pipeline import ClientRawResponse
from .. import models

class LargePersonGroupPersonOperations(object):
    """LargePersonGroupPersonOperations operations.

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

    def create(self, large_person_group_id, name=None, user_data=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Create a new person in a specified large person group.\n\n        :param large_person_group_id: Id referencing a particular large person\n         group.\n        :type large_person_group_id: str\n        :param name: User defined name, maximum length is 128.\n        :type name: str\n        :param user_data: User specified data. Length should not exceed 16KB.\n        :type user_data: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Person or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.face.models.Person or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        body = models.NameAndUserDataContract(name=name, user_data=user_data)
        url = self.create.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largePersonGroupId': self._serialize.url('large_person_group_id', large_person_group_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(body, 'NameAndUserDataContract')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('Person', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    create.metadata = {'url': '/largepersongroups/{largePersonGroupId}/persons'}

    def list(self, large_person_group_id, start=None, top=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        "List all persons in a large person group, and retrieve person\n        information (including personId, name, userData and persistedFaceIds of\n        registered faces of the person).\n\n        :param large_person_group_id: Id referencing a particular large person\n         group.\n        :type large_person_group_id: str\n        :param start: Starting person id to return (used to list a range of\n         persons).\n        :type start: str\n        :param top: Number of persons to return starting with the person id\n         indicated by the 'start' parameter.\n        :type top: int\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype: list[~azure.cognitiveservices.vision.face.models.Person] or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        "
        url = self.list.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largePersonGroupId': self._serialize.url('large_person_group_id', large_person_group_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
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
            deserialized = self._deserialize('[Person]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    list.metadata = {'url': '/largepersongroups/{largePersonGroupId}/persons'}

    def delete(self, large_person_group_id, person_id, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Delete an existing person from a large person group. The\n        persistedFaceId, userData, person name and face feature in the person\n        entry will all be deleted.\n\n        :param large_person_group_id: Id referencing a particular large person\n         group.\n        :type large_person_group_id: str\n        :param person_id: Id referencing a particular person.\n        :type person_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        url = self.delete.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largePersonGroupId': self._serialize.url('large_person_group_id', large_person_group_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$'), 'personId': self._serialize.url('person_id', person_id, 'str')}
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
    delete.metadata = {'url': '/largepersongroups/{largePersonGroupId}/persons/{personId}'}

    def get(self, large_person_group_id, person_id, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        "Retrieve a person's name and userData, and the persisted faceIds\n        representing the registered person face feature.\n\n        :param large_person_group_id: Id referencing a particular large person\n         group.\n        :type large_person_group_id: str\n        :param person_id: Id referencing a particular person.\n        :type person_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Person or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.face.models.Person or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        "
        url = self.get.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largePersonGroupId': self._serialize.url('large_person_group_id', large_person_group_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$'), 'personId': self._serialize.url('person_id', person_id, 'str')}
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
            deserialized = self._deserialize('Person', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get.metadata = {'url': '/largepersongroups/{largePersonGroupId}/persons/{personId}'}

    def update(self, large_person_group_id, person_id, name=None, user_data=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Update name or userData of a person.\n\n        :param large_person_group_id: Id referencing a particular large person\n         group.\n        :type large_person_group_id: str\n        :param person_id: Id referencing a particular person.\n        :type person_id: str\n        :param name: User defined name, maximum length is 128.\n        :type name: str\n        :param user_data: User specified data. Length should not exceed 16KB.\n        :type user_data: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        body = models.NameAndUserDataContract(name=name, user_data=user_data)
        url = self.update.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largePersonGroupId': self._serialize.url('large_person_group_id', large_person_group_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$'), 'personId': self._serialize.url('person_id', person_id, 'str')}
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
    update.metadata = {'url': '/largepersongroups/{largePersonGroupId}/persons/{personId}'}

    def delete_face(self, large_person_group_id, person_id, persisted_face_id, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Delete a face from a person in a large person group by specified\n        largePersonGroupId, personId and persistedFaceId.\n        <br /> Adding/deleting faces to/from a same person will be processed\n        sequentially. Adding/deleting faces to/from different persons are\n        processed in parallel.\n\n        :param large_person_group_id: Id referencing a particular large person\n         group.\n        :type large_person_group_id: str\n        :param person_id: Id referencing a particular person.\n        :type person_id: str\n        :param persisted_face_id: Id referencing a particular persistedFaceId\n         of an existing face.\n        :type persisted_face_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        url = self.delete_face.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largePersonGroupId': self._serialize.url('large_person_group_id', large_person_group_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$'), 'personId': self._serialize.url('person_id', person_id, 'str'), 'persistedFaceId': self._serialize.url('persisted_face_id', persisted_face_id, 'str')}
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
    delete_face.metadata = {'url': '/largepersongroups/{largePersonGroupId}/persons/{personId}/persistedfaces/{persistedFaceId}'}

    def get_face(self, large_person_group_id, person_id, persisted_face_id, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Retrieve information about a persisted face (specified by\n        persistedFaceId, personId and its belonging largePersonGroupId).\n\n        :param large_person_group_id: Id referencing a particular large person\n         group.\n        :type large_person_group_id: str\n        :param person_id: Id referencing a particular person.\n        :type person_id: str\n        :param persisted_face_id: Id referencing a particular persistedFaceId\n         of an existing face.\n        :type persisted_face_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: PersistedFace or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.face.models.PersistedFace or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        url = self.get_face.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largePersonGroupId': self._serialize.url('large_person_group_id', large_person_group_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$'), 'personId': self._serialize.url('person_id', person_id, 'str'), 'persistedFaceId': self._serialize.url('persisted_face_id', persisted_face_id, 'str')}
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
    get_face.metadata = {'url': '/largepersongroups/{largePersonGroupId}/persons/{personId}/persistedfaces/{persistedFaceId}'}

    def update_face(self, large_person_group_id, person_id, persisted_face_id, user_data=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        "Update a person persisted face's userData field.\n\n        :param large_person_group_id: Id referencing a particular large person\n         group.\n        :type large_person_group_id: str\n        :param person_id: Id referencing a particular person.\n        :type person_id: str\n        :param persisted_face_id: Id referencing a particular persistedFaceId\n         of an existing face.\n        :type persisted_face_id: str\n        :param user_data: User-provided data attached to the face. The size\n         limit is 1KB.\n        :type user_data: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        "
        body = models.UpdateFaceRequest(user_data=user_data)
        url = self.update_face.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largePersonGroupId': self._serialize.url('large_person_group_id', large_person_group_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$'), 'personId': self._serialize.url('person_id', person_id, 'str'), 'persistedFaceId': self._serialize.url('persisted_face_id', persisted_face_id, 'str')}
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
    update_face.metadata = {'url': '/largepersongroups/{largePersonGroupId}/persons/{personId}/persistedfaces/{persistedFaceId}'}

    def add_face_from_url(self, large_person_group_id, person_id, url, user_data=None, target_face=None, detection_model='detection_01', custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Add a face to a person into a large person group for face\n        identification or verification. To deal with an image contains multiple\n        faces, input face can be specified as an image with a targetFace\n        rectangle. It returns a persistedFaceId representing the added face. No\n        image will be stored. Only the extracted face feature will be stored on\n        server until [LargePersonGroup PersonFace -\n        Delete](https://docs.microsoft.com/rest/api/faceapi/largepersongroupperson/deleteface),\n        [LargePersonGroup Person -\n        Delete](https://docs.microsoft.com/rest/api/faceapi/largepersongroupperson/delete)\n        or [LargePersonGroup -\n        Delete](https://docs.microsoft.com/rest/api/faceapi/largepersongroup/delete)\n        is called.\n        <br /> Note persistedFaceId is different from faceId generated by [Face\n        -\n        Detect](https://docs.microsoft.com/rest/api/faceapi/face/detectwithurl).\n        * Higher face image quality means better recognition precision. Please\n        consider high-quality faces: frontal, clear, and face size is 200x200\n        pixels (100 pixels between eyes) or bigger.\n        * Each person entry can hold up to 248 faces.\n        * JPEG, PNG, GIF (the first frame), and BMP format are supported. The\n        allowed image file size is from 1KB to 6MB.\n        * "targetFace" rectangle should contain one face. Zero or multiple\n        faces will be regarded as an error. If the provided "targetFace"\n        rectangle is not returned from [Face -\n        Detect](https://docs.microsoft.com/rest/api/faceapi/face/detectwithurl),\n        there’s no guarantee to detect and add the face successfully.\n        * Out of detectable face size (36x36 - 4096x4096 pixels), large\n        head-pose, or large occlusions will cause failures.\n        * Adding/deleting faces to/from a same person will be processed\n        sequentially. Adding/deleting faces to/from different persons are\n        processed in parallel.\n        * The minimum detectable face size is 36x36 pixels in an image no\n        larger than 1920x1080 pixels. Images with dimensions higher than\n        1920x1080 pixels will need a proportionally larger minimum face size.\n        * Different \'detectionModel\' values can be provided. To use and compare\n        different detection models, please refer to [How to specify a detection\n        model](https://docs.microsoft.com/azure/cognitive-services/face/face-api-how-to-topics/specify-detection-model).\n\n        :param large_person_group_id: Id referencing a particular large person\n         group.\n        :type large_person_group_id: str\n        :param person_id: Id referencing a particular person.\n        :type person_id: str\n        :param url: Publicly reachable URL of an image\n        :type url: str\n        :param user_data: User-specified data about the face for any purpose.\n         The maximum length is 1KB.\n        :type user_data: str\n        :param target_face: A face rectangle to specify the target face to be\n         added to a person in the format of "targetFace=left,top,width,height".\n         E.g. "targetFace=10,10,100,100". If there is more than one face in the\n         image, targetFace is required to specify which face to add. No\n         targetFace means there is only one face detected in the entire image.\n        :type target_face: list[int]\n        :param detection_model: Name of detection model. Detection model is\n         used to detect faces in the submitted image. A detection model name\n         can be provided when performing Face - Detect or (Large)FaceList - Add\n         Face or (Large)PersonGroup - Add Face. The default value is\n         \'detection_01\', if another model is needed, please explicitly specify\n         it. Possible values include: \'detection_01\', \'detection_02\',\n         \'detection_03\'\n        :type detection_model: str or\n         ~azure.cognitiveservices.vision.face.models.DetectionModel\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: PersistedFace or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.face.models.PersistedFace or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        image_url = models.ImageUrl(url=url)
        url = self.add_face_from_url.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largePersonGroupId': self._serialize.url('large_person_group_id', large_person_group_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$'), 'personId': self._serialize.url('person_id', person_id, 'str')}
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
    add_face_from_url.metadata = {'url': '/largepersongroups/{largePersonGroupId}/persons/{personId}/persistedfaces'}

    def add_face_from_stream(self, large_person_group_id, person_id, image, user_data=None, target_face=None, detection_model='detection_01', custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            print('Hello World!')
        'Add a face to a person into a large person group for face\n        identification or verification. To deal with an image contains multiple\n        faces, input face can be specified as an image with a targetFace\n        rectangle. It returns a persistedFaceId representing the added face. No\n        image will be stored. Only the extracted face feature will be stored on\n        server until [LargePersonGroup PersonFace -\n        Delete](https://docs.microsoft.com/rest/api/faceapi/largepersongroupperson/deleteface),\n        [LargePersonGroup Person -\n        Delete](https://docs.microsoft.com/rest/api/faceapi/largepersongroupperson/delete)\n        or [LargePersonGroup -\n        Delete](https://docs.microsoft.com/rest/api/faceapi/largepersongroup/delete)\n        is called.\n        <br /> Note persistedFaceId is different from faceId generated by [Face\n        -\n        Detect](https://docs.microsoft.com/rest/api/faceapi/face/detectwithurl).\n        * Higher face image quality means better recognition precision. Please\n        consider high-quality faces: frontal, clear, and face size is 200x200\n        pixels (100 pixels between eyes) or bigger.\n        * Each person entry can hold up to 248 faces.\n        * JPEG, PNG, GIF (the first frame), and BMP format are supported. The\n        allowed image file size is from 1KB to 6MB.\n        * "targetFace" rectangle should contain one face. Zero or multiple\n        faces will be regarded as an error. If the provided "targetFace"\n        rectangle is not returned from [Face -\n        Detect](https://docs.microsoft.com/rest/api/faceapi/face/detectwithurl),\n        there’s no guarantee to detect and add the face successfully.\n        * Out of detectable face size (36x36 - 4096x4096 pixels), large\n        head-pose, or large occlusions will cause failures.\n        * Adding/deleting faces to/from a same person will be processed\n        sequentially. Adding/deleting faces to/from different persons are\n        processed in parallel.\n        * The minimum detectable face size is 36x36 pixels in an image no\n        larger than 1920x1080 pixels. Images with dimensions higher than\n        1920x1080 pixels will need a proportionally larger minimum face size.\n        * Different \'detectionModel\' values can be provided. To use and compare\n        different detection models, please refer to [How to specify a detection\n        model](https://docs.microsoft.com/azure/cognitive-services/face/face-api-how-to-topics/specify-detection-model).\n\n        :param large_person_group_id: Id referencing a particular large person\n         group.\n        :type large_person_group_id: str\n        :param person_id: Id referencing a particular person.\n        :type person_id: str\n        :param image: An image stream.\n        :type image: Generator\n        :param user_data: User-specified data about the face for any purpose.\n         The maximum length is 1KB.\n        :type user_data: str\n        :param target_face: A face rectangle to specify the target face to be\n         added to a person in the format of "targetFace=left,top,width,height".\n         E.g. "targetFace=10,10,100,100". If there is more than one face in the\n         image, targetFace is required to specify which face to add. No\n         targetFace means there is only one face detected in the entire image.\n        :type target_face: list[int]\n        :param detection_model: Name of detection model. Detection model is\n         used to detect faces in the submitted image. A detection model name\n         can be provided when performing Face - Detect or (Large)FaceList - Add\n         Face or (Large)PersonGroup - Add Face. The default value is\n         \'detection_01\', if another model is needed, please explicitly specify\n         it. Possible values include: \'detection_01\', \'detection_02\',\n         \'detection_03\'\n        :type detection_model: str or\n         ~azure.cognitiveservices.vision.face.models.DetectionModel\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: PersistedFace or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.face.models.PersistedFace or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        url = self.add_face_from_stream.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'largePersonGroupId': self._serialize.url('large_person_group_id', large_person_group_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$'), 'personId': self._serialize.url('person_id', person_id, 'str')}
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
    add_face_from_stream.metadata = {'url': '/largepersongroups/{largePersonGroupId}/persons/{personId}/persistedfaces'}