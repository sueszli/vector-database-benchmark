from msrest.pipeline import ClientRawResponse
from .. import models

class FaceOperations(object):
    """FaceOperations operations.

    You should not instantiate directly this class, but create a Client instance that will create it for you and attach it as attribute.

    :param client: Client for service requests.
    :param config: Configuration of service client.
    :param serializer: An object model serializer.
    :param deserializer: An object model deserializer.
    """
    models = models

    def __init__(self, client, config, serializer, deserializer):
        if False:
            i = 10
            return i + 15
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self.config = config

    def find_similar(self, face_id, face_list_id=None, large_face_list_id=None, face_ids=None, max_num_of_candidates_returned=20, mode='matchPerson', custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Given query face\'s faceId, to search the similar-looking faces from a\n        faceId array, a face list or a large face list. faceId array contains\n        the faces created by [Face - Detect With\n        Url](https://docs.microsoft.com/rest/api/faceapi/face/detectwithurl) or\n        [Face - Detect With\n        Stream](https://docs.microsoft.com/rest/api/faceapi/face/detectwithstream),\n        which will expire at the time specified by faceIdTimeToLive after\n        creation. A "faceListId" is created by [FaceList -\n        Create](https://docs.microsoft.com/rest/api/faceapi/facelist/create)\n        containing persistedFaceIds that will not expire. And a\n        "largeFaceListId" is created by [LargeFaceList -\n        Create](https://docs.microsoft.com/rest/api/faceapi/largefacelist/create)\n        containing persistedFaceIds that will also not expire. Depending on the\n        input the returned similar faces list contains faceIds or\n        persistedFaceIds ranked by similarity.\n        <br/>Find similar has two working modes, "matchPerson" and "matchFace".\n        "matchPerson" is the default mode that it tries to find faces of the\n        same person as possible by using internal same-person thresholds. It is\n        useful to find a known person\'s other photos. Note that an empty list\n        will be returned if no faces pass the internal thresholds. "matchFace"\n        mode ignores same-person thresholds and returns ranked similar faces\n        anyway, even the similarity is low. It can be used in the cases like\n        searching celebrity-looking faces.\n        <br/>The \'recognitionModel\' associated with the query face\'s faceId\n        should be the same as the \'recognitionModel\' used by the target faceId\n        array, face list or large face list.\n        .\n\n        :param face_id: FaceId of the query face. User needs to call Face -\n         Detect first to get a valid faceId. Note that this faceId is not\n         persisted and will expire at the time specified by faceIdTimeToLive\n         after the detection call\n        :type face_id: str\n        :param face_list_id: An existing user-specified unique candidate face\n         list, created in Face List - Create a Face List. Face list contains a\n         set of persistedFaceIds which are persisted and will never expire.\n         Parameter faceListId, largeFaceListId and faceIds should not be\n         provided at the same time.\n        :type face_list_id: str\n        :param large_face_list_id: An existing user-specified unique candidate\n         large face list, created in LargeFaceList - Create. Large face list\n         contains a set of persistedFaceIds which are persisted and will never\n         expire. Parameter faceListId, largeFaceListId and faceIds should not\n         be provided at the same time.\n        :type large_face_list_id: str\n        :param face_ids: An array of candidate faceIds. All of them are\n         created by Face - Detect and the faceIds will expire at the time\n         specified by faceIdTimeToLive after the detection call. The number of\n         faceIds is limited to 1000. Parameter faceListId, largeFaceListId and\n         faceIds should not be provided at the same time.\n        :type face_ids: list[str]\n        :param max_num_of_candidates_returned: The number of top similar faces\n         returned. The valid range is [1, 1000].\n        :type max_num_of_candidates_returned: int\n        :param mode: Similar face searching mode. It can be "matchPerson" or\n         "matchFace". Possible values include: \'matchPerson\', \'matchFace\'\n        :type mode: str or\n         ~azure.cognitiveservices.vision.face.models.FindSimilarMatchMode\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype: list[~azure.cognitiveservices.vision.face.models.SimilarFace]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        body = models.FindSimilarRequest(face_id=face_id, face_list_id=face_list_id, large_face_list_id=large_face_list_id, face_ids=face_ids, max_num_of_candidates_returned=max_num_of_candidates_returned, mode=mode)
        url = self.find_similar.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(body, 'FindSimilarRequest')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('[SimilarFace]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    find_similar.metadata = {'url': '/findsimilars'}

    def group(self, face_ids, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        "Divide candidate faces into groups based on face similarity.<br />\n        * The output is one or more disjointed face groups and a messyGroup. A\n        face group contains faces that have similar looking, often of the same\n        person. Face groups are ranked by group size, i.e. number of faces.\n        Notice that faces belonging to a same person might be split into\n        several groups in the result.\n        * MessyGroup is a special face group containing faces that cannot find\n        any similar counterpart face from original faces. The messyGroup will\n        not appear in the result if all faces found their counterparts.\n        * Group API needs at least 2 candidate faces and 1000 at most. We\n        suggest to try [Face -\n        Verify](https://docs.microsoft.com/rest/api/faceapi/face/verifyfacetoface)\n        when you only have 2 candidate faces.\n        * The 'recognitionModel' associated with the query faces' faceIds\n        should be the same.\n        .\n\n        :param face_ids: Array of candidate faceId created by Face - Detect.\n         The maximum is 1000 faces\n        :type face_ids: list[str]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: GroupResult or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.face.models.GroupResult or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        "
        body = models.GroupRequest(face_ids=face_ids)
        url = self.group.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(body, 'GroupRequest')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('GroupResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    group.metadata = {'url': '/group'}

    def identify(self, face_ids, person_group_id=None, large_person_group_id=None, max_num_of_candidates_returned=1, confidence_threshold=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        "1-to-many identification to find the closest matches of the specific\n        query person face from a person group or large person group.\n        <br/> For each face in the faceIds array, Face Identify will compute\n        similarities between the query face and all the faces in the person\n        group (given by personGroupId) or large person group (given by\n        largePersonGroupId), and return candidate person(s) for that face\n        ranked by similarity confidence. The person group/large person group\n        should be trained to make it ready for identification. See more in\n        [PersonGroup -\n        Train](https://docs.microsoft.com/rest/api/faceapi/persongroup/train)\n        and [LargePersonGroup -\n        Train](https://docs.microsoft.com/rest/api/faceapi/largepersongroup/train).\n        <br/>\n        Remarks:<br />\n        * The algorithm allows more than one face to be identified\n        independently at the same request, but no more than 10 faces.\n        * Each person in the person group/large person group could have more\n        than one face, but no more than 248 faces.\n        * Higher face image quality means better identification precision.\n        Please consider high-quality faces: frontal, clear, and face size is\n        200x200 pixels (100 pixels between eyes) or bigger.\n        * Number of candidates returned is restricted by\n        maxNumOfCandidatesReturned and confidenceThreshold. If no person is\n        identified, the returned candidates will be an empty array.\n        * Try [Face - Find\n        Similar](https://docs.microsoft.com/rest/api/faceapi/face/findsimilar)\n        when you need to find similar faces from a face list/large face list\n        instead of a person group/large person group.\n        * The 'recognitionModel' associated with the query faces' faceIds\n        should be the same as the 'recognitionModel' used by the target person\n        group or large person group.\n        .\n\n        :param face_ids: Array of query faces faceIds, created by the Face -\n         Detect. Each of the faces are identified independently. The valid\n         number of faceIds is between [1, 10].\n        :type face_ids: list[str]\n        :param person_group_id: PersonGroupId of the target person group,\n         created by PersonGroup - Create. Parameter personGroupId and\n         largePersonGroupId should not be provided at the same time.\n        :type person_group_id: str\n        :param large_person_group_id: LargePersonGroupId of the target large\n         person group, created by LargePersonGroup - Create. Parameter\n         personGroupId and largePersonGroupId should not be provided at the\n         same time.\n        :type large_person_group_id: str\n        :param max_num_of_candidates_returned: The range of\n         maxNumOfCandidatesReturned is between 1 and 100 (default is 1).\n        :type max_num_of_candidates_returned: int\n        :param confidence_threshold: Confidence threshold of identification,\n         used to judge whether one face belong to one person. The range of\n         confidenceThreshold is [0, 1] (default specified by algorithm).\n        :type confidence_threshold: float\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.vision.face.models.IdentifyResult] or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        "
        body = models.IdentifyRequest(face_ids=face_ids, person_group_id=person_group_id, large_person_group_id=large_person_group_id, max_num_of_candidates_returned=max_num_of_candidates_returned, confidence_threshold=confidence_threshold)
        url = self.identify.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(body, 'IdentifyRequest')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('[IdentifyResult]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    identify.metadata = {'url': '/identify'}

    def verify_face_to_face(self, face_id1, face_id2, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        "Verify whether two faces belong to a same person or whether one face\n        belongs to a person.\n        <br/>\n        Remarks:<br />\n        * Higher face image quality means better identification precision.\n        Please consider high-quality faces: frontal, clear, and face size is\n        200x200 pixels (100 pixels between eyes) or bigger.\n        * For the scenarios that are sensitive to accuracy please make your own\n        judgment.\n        * The 'recognitionModel' associated with the query faces' faceIds\n        should be the same as the 'recognitionModel' used by the target face,\n        person group or large person group.\n        .\n\n        :param face_id1: FaceId of the first face, comes from Face - Detect\n        :type face_id1: str\n        :param face_id2: FaceId of the second face, comes from Face - Detect\n        :type face_id2: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: VerifyResult or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.face.models.VerifyResult or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        "
        body = models.VerifyFaceToFaceRequest(face_id1=face_id1, face_id2=face_id2)
        url = self.verify_face_to_face.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(body, 'VerifyFaceToFaceRequest')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('VerifyResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    verify_face_to_face.metadata = {'url': '/verify'}

    def detect_with_url(self, url, return_face_id=True, return_face_landmarks=False, return_face_attributes=None, recognition_model='recognition_01', return_recognition_model=False, detection_model='detection_01', face_id_time_to_live=86400, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Detect human faces in an image, return face rectangles, and optionally\n        with faceIds, landmarks, and attributes.<br />\n        * No image will be stored. Only the extracted face feature will be\n        stored on server. The faceId is an identifier of the face feature and\n        will be used in [Face -\n        Identify](https://docs.microsoft.com/rest/api/faceapi/face/identify),\n        [Face -\n        Verify](https://docs.microsoft.com/rest/api/faceapi/face/verifyfacetoface),\n        and [Face - Find\n        Similar](https://docs.microsoft.com/rest/api/faceapi/face/findsimilar).\n        The stored face feature(s) will expire and be deleted at the time\n        specified by faceIdTimeToLive after the original detection call.\n        * Optional parameters include faceId, landmarks, and attributes.\n        Attributes include age, gender, headPose, smile, facialHair, glasses,\n        emotion, hair, makeup, occlusion, accessories, blur, exposure, noise,\n        mask, and qualityForRecognition. Some of the results returned for\n        specific attributes may not be highly accurate.\n        * JPEG, PNG, GIF (the first frame), and BMP format are supported. The\n        allowed image file size is from 1KB to 6MB.\n        * Up to 100 faces can be returned for an image. Faces are ranked by\n        face rectangle size from large to small.\n        * For optimal results when querying [Face -\n        Identify](https://docs.microsoft.com/rest/api/faceapi/face/identify),\n        [Face -\n        Verify](https://docs.microsoft.com/rest/api/faceapi/face/verifyfacetoface),\n        and [Face - Find\n        Similar](https://docs.microsoft.com/rest/api/faceapi/face/findsimilar)\n        (\'returnFaceId\' is true), please use faces that are: frontal, clear,\n        and with a minimum size of 200x200 pixels (100 pixels between eyes).\n        * The minimum detectable face size is 36x36 pixels in an image no\n        larger than 1920x1080 pixels. Images with dimensions higher than\n        1920x1080 pixels will need a proportionally larger minimum face size.\n        * Different \'detectionModel\' values can be provided. To use and compare\n        different detection models, please refer to [How to specify a detection\n        model](https://docs.microsoft.com/azure/cognitive-services/face/face-api-how-to-topics/specify-detection-model).\n        * Different \'recognitionModel\' values are provided. If follow-up\n        operations like Verify, Identify, Find Similar are needed, please\n        specify the recognition model with \'recognitionModel\' parameter. The\n        default value for \'recognitionModel\' is \'recognition_01\', if latest\n        model needed, please explicitly specify the model you need in this\n        parameter. Once specified, the detected faceIds will be associated with\n        the specified recognition model. More details, please refer to [Specify\n        a recognition\n        model](https://docs.microsoft.com/azure/cognitive-services/face/face-api-how-to-topics/specify-recognition-model).\n\n        :param url: Publicly reachable URL of an image\n        :type url: str\n        :param return_face_id: A value indicating whether the operation should\n         return faceIds of detected faces.\n        :type return_face_id: bool\n        :param return_face_landmarks: A value indicating whether the operation\n         should return landmarks of the detected faces.\n        :type return_face_landmarks: bool\n        :param return_face_attributes: Analyze and return the one or more\n         specified face attributes in the comma-separated string like\n         "returnFaceAttributes=age,gender". The available attributes depends on\n         the \'detectionModel\' specified. \'detection_01\' supports age, gender,\n         headPose, smile, facialHair, glasses, emotion, hair, makeup,\n         occlusion, accessories, blur, exposure, noise, and\n         qualityForRecognition. While \'detection_02\' does not support any\n         attributes and \'detection_03\' only supports mask and\n         qualityForRecognition. Additionally, qualityForRecognition is only\n         supported when the \'recognitionModel\' is specified as \'recognition_03\'\n         or \'recognition_04\'. Note that each face attribute analysis has\n         additional computational and time cost.\n        :type return_face_attributes: list[str or\n         ~azure.cognitiveservices.vision.face.models.FaceAttributeType]\n        :param recognition_model: Name of recognition model. Recognition model\n         is used when the face features are extracted and associated with\n         detected faceIds, (Large)FaceList or (Large)PersonGroup. A recognition\n         model name can be provided when performing Face - Detect or\n         (Large)FaceList - Create or (Large)PersonGroup - Create. The default\n         value is \'recognition_01\', if latest model needed, please explicitly\n         specify the model you need. Possible values include: \'recognition_01\',\n         \'recognition_02\', \'recognition_03\', \'recognition_04\'\n        :type recognition_model: str or\n         ~azure.cognitiveservices.vision.face.models.RecognitionModel\n        :param return_recognition_model: A value indicating whether the\n         operation should return \'recognitionModel\' in response.\n        :type return_recognition_model: bool\n        :param detection_model: Name of detection model. Detection model is\n         used to detect faces in the submitted image. A detection model name\n         can be provided when performing Face - Detect or (Large)FaceList - Add\n         Face or (Large)PersonGroup - Add Face. The default value is\n         \'detection_01\', if another model is needed, please explicitly specify\n         it. Possible values include: \'detection_01\', \'detection_02\',\n         \'detection_03\'\n        :type detection_model: str or\n         ~azure.cognitiveservices.vision.face.models.DetectionModel\n        :param face_id_time_to_live: The number of seconds for the faceId\n         being cached. Supported range from 60 seconds up to 86400 seconds. The\n         default value is 86400 (24 hours).\n        :type face_id_time_to_live: int\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype: list[~azure.cognitiveservices.vision.face.models.DetectedFace]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        image_url = models.ImageUrl(url=url)
        url = self.detect_with_url.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if return_face_id is not None:
            query_parameters['returnFaceId'] = self._serialize.query('return_face_id', return_face_id, 'bool')
        if return_face_landmarks is not None:
            query_parameters['returnFaceLandmarks'] = self._serialize.query('return_face_landmarks', return_face_landmarks, 'bool')
        if return_face_attributes is not None:
            query_parameters['returnFaceAttributes'] = self._serialize.query('return_face_attributes', return_face_attributes, '[FaceAttributeType]', div=',')
        if recognition_model is not None:
            query_parameters['recognitionModel'] = self._serialize.query('recognition_model', recognition_model, 'str')
        if return_recognition_model is not None:
            query_parameters['returnRecognitionModel'] = self._serialize.query('return_recognition_model', return_recognition_model, 'bool')
        if detection_model is not None:
            query_parameters['detectionModel'] = self._serialize.query('detection_model', detection_model, 'str')
        if face_id_time_to_live is not None:
            query_parameters['faceIdTimeToLive'] = self._serialize.query('face_id_time_to_live', face_id_time_to_live, 'int', maximum=86400, minimum=60)
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
            deserialized = self._deserialize('[DetectedFace]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    detect_with_url.metadata = {'url': '/detect'}

    def verify_face_to_person(self, face_id, person_id, person_group_id=None, large_person_group_id=None, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Verify whether two faces belong to a same person. Compares a face Id\n        with a Person Id.\n\n        :param face_id: FaceId of the face, comes from Face - Detect\n        :type face_id: str\n        :param person_id: Specify a certain person in a person group or a\n         large person group. personId is created in PersonGroup Person - Create\n         or LargePersonGroup Person - Create.\n        :type person_id: str\n        :param person_group_id: Using existing personGroupId and personId for\n         fast loading a specified person. personGroupId is created in\n         PersonGroup - Create. Parameter personGroupId and largePersonGroupId\n         should not be provided at the same time.\n        :type person_group_id: str\n        :param large_person_group_id: Using existing largePersonGroupId and\n         personId for fast loading a specified person. largePersonGroupId is\n         created in LargePersonGroup - Create. Parameter personGroupId and\n         largePersonGroupId should not be provided at the same time.\n        :type large_person_group_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: VerifyResult or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.face.models.VerifyResult or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        body = models.VerifyFaceToPersonRequest(face_id=face_id, person_group_id=person_group_id, large_person_group_id=large_person_group_id, person_id=person_id)
        url = self.verify_face_to_person.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(body, 'VerifyFaceToPersonRequest')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('VerifyResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    verify_face_to_person.metadata = {'url': '/verify'}

    def detect_with_stream(self, image, return_face_id=True, return_face_landmarks=False, return_face_attributes=None, recognition_model='recognition_01', return_recognition_model=False, detection_model='detection_01', face_id_time_to_live=86400, custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            while True:
                i = 10
        'Detect human faces in an image, return face rectangles, and optionally\n        with faceIds, landmarks, and attributes.<br />\n        * No image will be stored. Only the extracted face feature will be\n        stored on server. The faceId is an identifier of the face feature and\n        will be used in [Face -\n        Identify](https://docs.microsoft.com/rest/api/faceapi/face/identify),\n        [Face -\n        Verify](https://docs.microsoft.com/rest/api/faceapi/face/verifyfacetoface),\n        and [Face - Find\n        Similar](https://docs.microsoft.com/rest/api/faceapi/face/findsimilar).\n        The stored face feature(s) will expire and be deleted at the time\n        specified by faceIdTimeToLive after the original detection call.\n        * Optional parameters include faceId, landmarks, and attributes.\n        Attributes include age, gender, headPose, smile, facialHair, glasses,\n        emotion, hair, makeup, occlusion, accessories, blur, exposure, noise,\n        mask, and qualityForRecognition. Some of the results returned for\n        specific attributes may not be highly accurate.\n        * JPEG, PNG, GIF (the first frame), and BMP format are supported. The\n        allowed image file size is from 1KB to 6MB.\n        * Up to 100 faces can be returned for an image. Faces are ranked by\n        face rectangle size from large to small.\n        * For optimal results when querying [Face -\n        Identify](https://docs.microsoft.com/rest/api/faceapi/face/identify),\n        [Face -\n        Verify](https://docs.microsoft.com/rest/api/faceapi/face/verifyfacetoface),\n        and [Face - Find\n        Similar](https://docs.microsoft.com/rest/api/faceapi/face/findsimilar)\n        (\'returnFaceId\' is true), please use faces that are: frontal, clear,\n        and with a minimum size of 200x200 pixels (100 pixels between eyes).\n        * The minimum detectable face size is 36x36 pixels in an image no\n        larger than 1920x1080 pixels. Images with dimensions higher than\n        1920x1080 pixels will need a proportionally larger minimum face size.\n        * Different \'detectionModel\' values can be provided. To use and compare\n        different detection models, please refer to [How to specify a detection\n        model](https://docs.microsoft.com/azure/cognitive-services/face/face-api-how-to-topics/specify-detection-model)\n        * Different \'recognitionModel\' values are provided. If follow-up\n        operations like Verify, Identify, Find Similar are needed, please\n        specify the recognition model with \'recognitionModel\' parameter. The\n        default value for \'recognitionModel\' is \'recognition_01\', if latest\n        model needed, please explicitly specify the model you need in this\n        parameter. Once specified, the detected faceIds will be associated with\n        the specified recognition model. More details, please refer to [Specify\n        a recognition\n        model](https://docs.microsoft.com/azure/cognitive-services/face/face-api-how-to-topics/specify-recognition-model).\n\n        :param image: An image stream.\n        :type image: Generator\n        :param return_face_id: A value indicating whether the operation should\n         return faceIds of detected faces.\n        :type return_face_id: bool\n        :param return_face_landmarks: A value indicating whether the operation\n         should return landmarks of the detected faces.\n        :type return_face_landmarks: bool\n        :param return_face_attributes: Analyze and return the one or more\n         specified face attributes in the comma-separated string like\n         "returnFaceAttributes=age,gender". The available attributes depends on\n         the \'detectionModel\' specified. \'detection_01\' supports age, gender,\n         headPose, smile, facialHair, glasses, emotion, hair, makeup,\n         occlusion, accessories, blur, exposure, noise, and\n         qualityForRecognition. While \'detection_02\' does not support any\n         attributes and \'detection_03\' only supports mask and\n         qualityForRecognition. Additionally, qualityForRecognition is only\n         supported when the \'recognitionModel\' is specified as \'recognition_03\'\n         or \'recognition_04\'. Note that each face attribute analysis has\n         additional computational and time cost.\n        :type return_face_attributes: list[str or\n         ~azure.cognitiveservices.vision.face.models.FaceAttributeType]\n        :param recognition_model: Name of recognition model. Recognition model\n         is used when the face features are extracted and associated with\n         detected faceIds, (Large)FaceList or (Large)PersonGroup. A recognition\n         model name can be provided when performing Face - Detect or\n         (Large)FaceList - Create or (Large)PersonGroup - Create. The default\n         value is \'recognition_01\', if latest model needed, please explicitly\n         specify the model you need. Possible values include: \'recognition_01\',\n         \'recognition_02\', \'recognition_03\', \'recognition_04\'\n        :type recognition_model: str or\n         ~azure.cognitiveservices.vision.face.models.RecognitionModel\n        :param return_recognition_model: A value indicating whether the\n         operation should return \'recognitionModel\' in response.\n        :type return_recognition_model: bool\n        :param detection_model: Name of detection model. Detection model is\n         used to detect faces in the submitted image. A detection model name\n         can be provided when performing Face - Detect or (Large)FaceList - Add\n         Face or (Large)PersonGroup - Add Face. The default value is\n         \'detection_01\', if another model is needed, please explicitly specify\n         it. Possible values include: \'detection_01\', \'detection_02\',\n         \'detection_03\'\n        :type detection_model: str or\n         ~azure.cognitiveservices.vision.face.models.DetectionModel\n        :param face_id_time_to_live: The number of seconds for the faceId\n         being cached. Supported range from 60 seconds up to 86400 seconds. The\n         default value is 86400 (24 hours).\n        :type face_id_time_to_live: int\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype: list[~azure.cognitiveservices.vision.face.models.DetectedFace]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.face.models.APIErrorException>`\n        '
        url = self.detect_with_stream.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if return_face_id is not None:
            query_parameters['returnFaceId'] = self._serialize.query('return_face_id', return_face_id, 'bool')
        if return_face_landmarks is not None:
            query_parameters['returnFaceLandmarks'] = self._serialize.query('return_face_landmarks', return_face_landmarks, 'bool')
        if return_face_attributes is not None:
            query_parameters['returnFaceAttributes'] = self._serialize.query('return_face_attributes', return_face_attributes, '[FaceAttributeType]', div=',')
        if recognition_model is not None:
            query_parameters['recognitionModel'] = self._serialize.query('recognition_model', recognition_model, 'str')
        if return_recognition_model is not None:
            query_parameters['returnRecognitionModel'] = self._serialize.query('return_recognition_model', return_recognition_model, 'bool')
        if detection_model is not None:
            query_parameters['detectionModel'] = self._serialize.query('detection_model', detection_model, 'str')
        if face_id_time_to_live is not None:
            query_parameters['faceIdTimeToLive'] = self._serialize.query('face_id_time_to_live', face_id_time_to_live, 'int', maximum=86400, minimum=60)
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
            deserialized = self._deserialize('[DetectedFace]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    detect_with_stream.metadata = {'url': '/detect'}