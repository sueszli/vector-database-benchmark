from msrest.pipeline import ClientRawResponse
from .. import models

class FaceListOperations(object):
    """FaceListOperations operations.

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

    def create(self, face_list_id, name, user_data=None, recognition_model='recognition_01', custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        "Create an empty face list with user-specified faceListId, name, an\n        optional userData and recognitionModel. Up to 64 face lists are allowed\n        in one subscription.\n        <br /> Face list is a list of faces, up to 1,000 faces, and used by\n        [Face - Find\n        Similar](https://docs.microsoft.com/rest/api/faceapi/face/findsimilar).\n        <br /> After creation, user should use [FaceList - Add\n        Face](https://docs.microsoft.com/rest/api/faceapi/facelist/addfacefromurl)\n        to import the faces. No image will be stored. Only the extracted face\n        features are stored on server until [FaceList -\n        Delete](https://docs.microsoft.com/rest/api/faceapi/facelist/delete) is\n        called.\n        <br /> Find Similar is used for scenario like finding celebrity-like\n        faces, similar face filtering, or as a light way face identification.\n        But if the actual use is to identify person, please use\n        [PersonGroup](https://docs.microsoft.com/rest/api/faceapi/persongroup)\n        /\n        [LargePersonGroup](https://docs.microsoft.com/rest/api/faceapi/largepersongroup)\n        and [Face -\n        Identify](https://docs.microsoft.com/rest/api/faceapi/face/identify).\n        <br /> Please consider\n        [LargeFaceList](https://docs.microsoft.com/rest/api/faceapi/largefacelist)\n        when the face number is large. It can support up to 1,000,000 faces.\n        <br />'recognitionModel' should be specified to associate with this\n        face list. The default value for 'recognitionModel' is\n        'recognition_01', if the latest model needed, please explicitly specify\n        the model you need in this parameter. New faces that are added to an\n        existing face list will use the recognition model that's already\n        associated with the collection. Existing face features in a face list\n        can't be updated to features extracted by another version of\n        recognition model.\n        Please Refer to [Specify a face recognition\n        model](https://docs.microsoft.com/azure/cognitive-services/face/face-api-how-to-topics/specify-recognition-model).\n\n        :param face_list_id: Id referencing a particular face list.\n        :type face_list_id: str\n        :param name: User defined name, maximum length is 128.\n        :type name: str\n        :param user_data: User specified data. Length should not exceed 16KB.\n        :type user_data: str\n        :param recognition_model: Possible values include: 'recognition_01',\n         'recognition_02', 'recognition_03', 'recognition_04'\n        :type recognition_model: str or\n         ~azure.cognitiveservices.vision.face.models.RecognitionModel\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        "
        body = models.MetaDataContract(name=name, user_data=user_data, recognition_model=recognition_model)
        url = self.create.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'faceListId': self._serialize.url('face_list_id', face_list_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
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
    create.metadata = {'url': '/facelists/{faceListId}'}

    def get(self, face_list_id, return_recognition_model=False, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        "Retrieve a face list’s faceListId, name, userData, recognitionModel and\n        faces in the face list.\n        .\n\n        :param face_list_id: Id referencing a particular face list.\n        :type face_list_id: str\n        :param return_recognition_model: A value indicating whether the\n         operation should return 'recognitionModel' in response.\n        :type return_recognition_model: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: FaceList or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.face.models.FaceList or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        "
        url = self.get.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'faceListId': self._serialize.url('face_list_id', face_list_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
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
            deserialized = self._deserialize('FaceList', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get.metadata = {'url': '/facelists/{faceListId}'}

    def update(self, face_list_id, name=None, user_data=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Update information of a face list.\n\n        :param face_list_id: Id referencing a particular face list.\n        :type face_list_id: str\n        :param name: User defined name, maximum length is 128.\n        :type name: str\n        :param user_data: User specified data. Length should not exceed 16KB.\n        :type user_data: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        body = models.NameAndUserDataContract(name=name, user_data=user_data)
        url = self.update.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'faceListId': self._serialize.url('face_list_id', face_list_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
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
    update.metadata = {'url': '/facelists/{faceListId}'}

    def delete(self, face_list_id, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Delete a specified face list.\n\n        :param face_list_id: Id referencing a particular face list.\n        :type face_list_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        url = self.delete.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'faceListId': self._serialize.url('face_list_id', face_list_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
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
    delete.metadata = {'url': '/facelists/{faceListId}'}

    def list(self, return_recognition_model=False, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        "List face lists’ faceListId, name, userData and recognitionModel. <br\n        />\n        To get face information inside faceList use [FaceList -\n        Get](https://docs.microsoft.com/rest/api/faceapi/facelist/get)\n        .\n\n        :param return_recognition_model: A value indicating whether the\n         operation should return 'recognitionModel' in response.\n        :type return_recognition_model: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype: list[~azure.cognitiveservices.vision.face.models.FaceList] or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        "
        url = self.list.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
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
            deserialized = self._deserialize('[FaceList]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    list.metadata = {'url': '/facelists'}

    def delete_face(self, face_list_id, persisted_face_id, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Delete a face from a face list by specified faceListId and\n        persistedFaceId.\n        <br /> Adding/deleting faces to/from a same face list are processed\n        sequentially and to/from different face lists are in parallel.\n\n        :param face_list_id: Id referencing a particular face list.\n        :type face_list_id: str\n        :param persisted_face_id: Id referencing a particular persistedFaceId\n         of an existing face.\n        :type persisted_face_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        url = self.delete_face.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'faceListId': self._serialize.url('face_list_id', face_list_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$'), 'persistedFaceId': self._serialize.url('persisted_face_id', persisted_face_id, 'str')}
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
    delete_face.metadata = {'url': '/facelists/{faceListId}/persistedfaces/{persistedFaceId}'}

    def add_face_from_url(self, face_list_id, url, user_data=None, target_face=None, detection_model='detection_01', custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Add a face to a specified face list, up to 1,000 faces.\n        <br /> To deal with an image contains multiple faces, input face can be\n        specified as an image with a targetFace rectangle. It returns a\n        persistedFaceId representing the added face. No image will be stored.\n        Only the extracted face feature will be stored on server until\n        [FaceList - Delete\n        Face](https://docs.microsoft.com/rest/api/faceapi/facelist/deleteface)\n        or [FaceList -\n        Delete](https://docs.microsoft.com/rest/api/faceapi/facelist/delete) is\n        called.\n        <br /> Note persistedFaceId is different from faceId generated by [Face\n        -\n        Detect](https://docs.microsoft.com/rest/api/faceapi/face/detectwithurl).\n        * Higher face image quality means better detection and recognition\n        precision. Please consider high-quality faces: frontal, clear, and face\n        size is 200x200 pixels (100 pixels between eyes) or bigger.\n        * JPEG, PNG, GIF (the first frame), and BMP format are supported. The\n        allowed image file size is from 1KB to 6MB.\n        * "targetFace" rectangle should contain one face. Zero or multiple\n        faces will be regarded as an error. If the provided "targetFace"\n        rectangle is not returned from [Face -\n        Detect](https://docs.microsoft.com/rest/api/faceapi/face/detectwithurl),\n        there’s no guarantee to detect and add the face successfully.\n        * Out of detectable face size (36x36 - 4096x4096 pixels), large\n        head-pose, or large occlusions will cause failures.\n        * Adding/deleting faces to/from a same face list are processed\n        sequentially and to/from different face lists are in parallel.\n        * The minimum detectable face size is 36x36 pixels in an image no\n        larger than 1920x1080 pixels. Images with dimensions higher than\n        1920x1080 pixels will need a proportionally larger minimum face size.\n        * Different \'detectionModel\' values can be provided. To use and compare\n        different detection models, please refer to [How to specify a detection\n        model](https://docs.microsoft.com/azure/cognitive-services/face/face-api-how-to-topics/specify-detection-model).\n\n        :param face_list_id: Id referencing a particular face list.\n        :type face_list_id: str\n        :param url: Publicly reachable URL of an image\n        :type url: str\n        :param user_data: User-specified data about the face for any purpose.\n         The maximum length is 1KB.\n        :type user_data: str\n        :param target_face: A face rectangle to specify the target face to be\n         added to a person in the format of "targetFace=left,top,width,height".\n         E.g. "targetFace=10,10,100,100". If there is more than one face in the\n         image, targetFace is required to specify which face to add. No\n         targetFace means there is only one face detected in the entire image.\n        :type target_face: list[int]\n        :param detection_model: Name of detection model. Detection model is\n         used to detect faces in the submitted image. A detection model name\n         can be provided when performing Face - Detect or (Large)FaceList - Add\n         Face or (Large)PersonGroup - Add Face. The default value is\n         \'detection_01\', if another model is needed, please explicitly specify\n         it. Possible values include: \'detection_01\', \'detection_02\',\n         \'detection_03\'\n        :type detection_model: str or\n         ~azure.cognitiveservices.vision.face.models.DetectionModel\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: PersistedFace or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.face.models.PersistedFace or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        image_url = models.ImageUrl(url=url)
        url = self.add_face_from_url.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'faceListId': self._serialize.url('face_list_id', face_list_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
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
    add_face_from_url.metadata = {'url': '/facelists/{faceListId}/persistedfaces'}

    def add_face_from_stream(self, face_list_id, image, user_data=None, target_face=None, detection_model='detection_01', custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            i = 10
            return i + 15
        'Add a face to a specified face list, up to 1,000 faces.\n        <br /> To deal with an image contains multiple faces, input face can be\n        specified as an image with a targetFace rectangle. It returns a\n        persistedFaceId representing the added face. No image will be stored.\n        Only the extracted face feature will be stored on server until\n        [FaceList - Delete\n        Face](https://docs.microsoft.com/rest/api/faceapi/facelist/deleteface)\n        or [FaceList -\n        Delete](https://docs.microsoft.com/rest/api/faceapi/facelist/delete) is\n        called.\n        <br /> Note persistedFaceId is different from faceId generated by [Face\n        -\n        Detect](https://docs.microsoft.com/rest/api/faceapi/face/detectwithurl).\n        * Higher face image quality means better detection and recognition\n        precision. Please consider high-quality faces: frontal, clear, and face\n        size is 200x200 pixels (100 pixels between eyes) or bigger.\n        * JPEG, PNG, GIF (the first frame), and BMP format are supported. The\n        allowed image file size is from 1KB to 6MB.\n        * "targetFace" rectangle should contain one face. Zero or multiple\n        faces will be regarded as an error. If the provided "targetFace"\n        rectangle is not returned from [Face -\n        Detect](https://docs.microsoft.com/rest/api/faceapi/face/detectwithurl),\n        there’s no guarantee to detect and add the face successfully.\n        * Out of detectable face size (36x36 - 4096x4096 pixels), large\n        head-pose, or large occlusions will cause failures.\n        * Adding/deleting faces to/from a same face list are processed\n        sequentially and to/from different face lists are in parallel.\n        * The minimum detectable face size is 36x36 pixels in an image no\n        larger than 1920x1080 pixels. Images with dimensions higher than\n        1920x1080 pixels will need a proportionally larger minimum face size.\n        * Different \'detectionModel\' values can be provided. To use and compare\n        different detection models, please refer to [How to specify a detection\n        model](https://docs.microsoft.com/azure/cognitive-services/face/face-api-how-to-topics/specify-detection-model).\n\n        :param face_list_id: Id referencing a particular face list.\n        :type face_list_id: str\n        :param image: An image stream.\n        :type image: Generator\n        :param user_data: User-specified data about the face for any purpose.\n         The maximum length is 1KB.\n        :type user_data: str\n        :param target_face: A face rectangle to specify the target face to be\n         added to a person in the format of "targetFace=left,top,width,height".\n         E.g. "targetFace=10,10,100,100". If there is more than one face in the\n         image, targetFace is required to specify which face to add. No\n         targetFace means there is only one face detected in the entire image.\n        :type target_face: list[int]\n        :param detection_model: Name of detection model. Detection model is\n         used to detect faces in the submitted image. A detection model name\n         can be provided when performing Face - Detect or (Large)FaceList - Add\n         Face or (Large)PersonGroup - Add Face. The default value is\n         \'detection_01\', if another model is needed, please explicitly specify\n         it. Possible values include: \'detection_01\', \'detection_02\',\n         \'detection_03\'\n        :type detection_model: str or\n         ~azure.cognitiveservices.vision.face.models.DetectionModel\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: PersistedFace or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.face.models.PersistedFace or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        url = self.add_face_from_stream.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'faceListId': self._serialize.url('face_list_id', face_list_id, 'str', max_length=64, pattern='^[a-z0-9-_]+$')}
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
    add_face_from_stream.metadata = {'url': '/facelists/{faceListId}/persistedfaces'}