from msrest.pipeline import ClientRawResponse
from msrest.exceptions import HttpOperationError
from .. import models

class ComputerVisionClientOperationsMixin(object):

    def analyze_image(self, url, visual_features=None, details=None, language='en', description_exclude=None, model_version='latest', custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'This operation extracts a rich set of visual features based on the\n        image content.\n        Two input methods are supported -- (1) Uploading an image or (2)\n        specifying an image URL. Within your request, there is an optional\n        parameter to allow you to choose which features to return. By default,\n        image categories are returned in the response.\n        A successful response will be returned in JSON. If the request failed,\n        the response will contain an error code and a message to help\n        understand what went wrong.\n\n        :param url: Publicly reachable URL of an image.\n        :type url: str\n        :param visual_features: A string indicating what visual feature types\n         to return. Multiple values should be comma-separated. Valid visual\n         feature types include: Categories - categorizes image content\n         according to a taxonomy defined in documentation. Tags - tags the\n         image with a detailed list of words related to the image content.\n         Description - describes the image content with a complete English\n         sentence. Faces - detects if faces are present. If present, generate\n         coordinates, gender and age. ImageType - detects if image is clipart\n         or a line drawing. Color - determines the accent color, dominant\n         color, and whether an image is black&white. Adult - detects if the\n         image is pornographic in nature (depicts nudity or a sex act), or is\n         gory (depicts extreme violence or blood). Sexually suggestive content\n         (aka racy content) is also detected. Objects - detects various objects\n         within an image, including the approximate location. The Objects\n         argument is only available in English. Brands - detects various brands\n         within an image, including the approximate location. The Brands\n         argument is only available in English.\n        :type visual_features: list[str or\n         ~azure.cognitiveservices.vision.computervision.models.VisualFeatureTypes]\n        :param details: A string indicating which domain-specific details to\n         return. Multiple values should be comma-separated. Valid visual\n         feature types include: Celebrities - identifies celebrities if\n         detected in the image, Landmarks - identifies notable landmarks in the\n         image.\n        :type details: list[str or\n         ~azure.cognitiveservices.vision.computervision.models.Details]\n        :param language: The desired language for output generation. If this\n         parameter is not specified, the default value is\n         &quot;en&quot;.Supported languages:en - English, Default. es -\n         Spanish, ja - Japanese, pt - Portuguese, zh - Simplified Chinese.\n         Possible values include: \'en\', \'es\', \'ja\', \'pt\', \'zh\'\n        :type language: str\n        :param description_exclude: Turn off specified domain models when\n         generating the description.\n        :type description_exclude: list[str or\n         ~azure.cognitiveservices.vision.computervision.models.DescriptionExclude]\n        :param model_version: Optional parameter to specify the version of the\n         AI model. Accepted values are: "latest", "2021-04-01". Defaults to\n         "latest".\n        :type model_version: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ImageAnalysis or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.computervision.models.ImageAnalysis or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ComputerVisionErrorResponseException<azure.cognitiveservices.vision.computervision.models.ComputerVisionErrorResponseException>`\n        '
        image_url = models.ImageUrl(url=url)
        url = self.analyze_image.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if visual_features is not None:
            query_parameters['visualFeatures'] = self._serialize.query('visual_features', visual_features, '[VisualFeatureTypes]', div=',')
        if details is not None:
            query_parameters['details'] = self._serialize.query('details', details, '[Details]', div=',')
        if language is not None:
            query_parameters['language'] = self._serialize.query('language', language, 'str')
        if description_exclude is not None:
            query_parameters['descriptionExclude'] = self._serialize.query('description_exclude', description_exclude, '[DescriptionExclude]', div=',')
        if model_version is not None:
            query_parameters['model-version'] = self._serialize.query('model_version', model_version, 'str', pattern='^(latest|\\d{4}-\\d{2}-\\d{2})(-preview)?$')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(image_url, 'ImageUrl')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ComputerVisionErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ImageAnalysis', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    analyze_image.metadata = {'url': '/analyze'}

    def describe_image(self, url, max_candidates=1, language='en', description_exclude=None, model_version='latest', custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'This operation generates a description of an image in human readable\n        language with complete sentences. The description is based on a\n        collection of content tags, which are also returned by the operation.\n        More than one description can be generated for each image. Descriptions\n        are ordered by their confidence score. Descriptions may include results\n        from celebrity and landmark domain models, if applicable.\n        Two input methods are supported -- (1) Uploading an image or (2)\n        specifying an image URL.\n        A successful response will be returned in JSON. If the request failed,\n        the response will contain an error code and a message to help\n        understand what went wrong.\n\n        :param url: Publicly reachable URL of an image.\n        :type url: str\n        :param max_candidates: Maximum number of candidate descriptions to be\n         returned.  The default is 1.\n        :type max_candidates: int\n        :param language: The desired language for output generation. If this\n         parameter is not specified, the default value is\n         &quot;en&quot;.Supported languages:en - English, Default. es -\n         Spanish, ja - Japanese, pt - Portuguese, zh - Simplified Chinese.\n         Possible values include: \'en\', \'es\', \'ja\', \'pt\', \'zh\'\n        :type language: str\n        :param description_exclude: Turn off specified domain models when\n         generating the description.\n        :type description_exclude: list[str or\n         ~azure.cognitiveservices.vision.computervision.models.DescriptionExclude]\n        :param model_version: Optional parameter to specify the version of the\n         AI model. Accepted values are: "latest", "2021-04-01". Defaults to\n         "latest".\n        :type model_version: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ImageDescription or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.computervision.models.ImageDescription\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ComputerVisionErrorResponseException<azure.cognitiveservices.vision.computervision.models.ComputerVisionErrorResponseException>`\n        '
        image_url = models.ImageUrl(url=url)
        url = self.describe_image.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if max_candidates is not None:
            query_parameters['maxCandidates'] = self._serialize.query('max_candidates', max_candidates, 'int')
        if language is not None:
            query_parameters['language'] = self._serialize.query('language', language, 'str')
        if description_exclude is not None:
            query_parameters['descriptionExclude'] = self._serialize.query('description_exclude', description_exclude, '[DescriptionExclude]', div=',')
        if model_version is not None:
            query_parameters['model-version'] = self._serialize.query('model_version', model_version, 'str', pattern='^(latest|\\d{4}-\\d{2}-\\d{2})(-preview)?$')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(image_url, 'ImageUrl')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ComputerVisionErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ImageDescription', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    describe_image.metadata = {'url': '/describe'}

    def detect_objects(self, url, model_version='latest', custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Performs object detection on the specified image.\n        Two input methods are supported -- (1) Uploading an image or (2)\n        specifying an image URL.\n        A successful response will be returned in JSON. If the request failed,\n        the response will contain an error code and a message to help\n        understand what went wrong.\n\n        :param url: Publicly reachable URL of an image.\n        :type url: str\n        :param model_version: Optional parameter to specify the version of the\n         AI model. Accepted values are: "latest", "2021-04-01". Defaults to\n         "latest".\n        :type model_version: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: DetectResult or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.computervision.models.DetectResult or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ComputerVisionErrorResponseException<azure.cognitiveservices.vision.computervision.models.ComputerVisionErrorResponseException>`\n        '
        image_url = models.ImageUrl(url=url)
        url = self.detect_objects.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if model_version is not None:
            query_parameters['model-version'] = self._serialize.query('model_version', model_version, 'str', pattern='^(latest|\\d{4}-\\d{2}-\\d{2})(-preview)?$')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(image_url, 'ImageUrl')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ComputerVisionErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('DetectResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    detect_objects.metadata = {'url': '/detect'}

    def list_models(self, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'This operation returns the list of domain-specific models that are\n        supported by the Computer Vision API. Currently, the API supports\n        following domain-specific models: celebrity recognizer, landmark\n        recognizer.\n        A successful response will be returned in JSON. If the request failed,\n        the response will contain an error code and a message to help\n        understand what went wrong.\n\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ListModelsResult or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.computervision.models.ListModelsResult\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ComputerVisionErrorResponseException<azure.cognitiveservices.vision.computervision.models.ComputerVisionErrorResponseException>`\n        '
        url = self.list_models.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ComputerVisionErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ListModelsResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    list_models.metadata = {'url': '/models'}

    def analyze_image_by_domain(self, model, url, language='en', model_version='latest', custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'This operation recognizes content within an image by applying a\n        domain-specific model. The list of domain-specific models that are\n        supported by the Computer Vision API can be retrieved using the /models\n        GET request. Currently, the API provides following domain-specific\n        models: celebrities, landmarks.\n        Two input methods are supported -- (1) Uploading an image or (2)\n        specifying an image URL.\n        A successful response will be returned in JSON.\n        If the request failed, the response will contain an error code and a\n        message to help understand what went wrong.\n\n        :param model: The domain-specific content to recognize.\n        :type model: str\n        :param url: Publicly reachable URL of an image.\n        :type url: str\n        :param language: The desired language for output generation. If this\n         parameter is not specified, the default value is\n         &quot;en&quot;.Supported languages:en - English, Default. es -\n         Spanish, ja - Japanese, pt - Portuguese, zh - Simplified Chinese.\n         Possible values include: \'en\', \'es\', \'ja\', \'pt\', \'zh\'\n        :type language: str\n        :param model_version: Optional parameter to specify the version of the\n         AI model. Accepted values are: "latest", "2021-04-01". Defaults to\n         "latest".\n        :type model_version: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: DomainModelResults or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.computervision.models.DomainModelResults\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ComputerVisionErrorResponseException<azure.cognitiveservices.vision.computervision.models.ComputerVisionErrorResponseException>`\n        '
        image_url = models.ImageUrl(url=url)
        url = self.analyze_image_by_domain.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'model': self._serialize.url('model', model, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if language is not None:
            query_parameters['language'] = self._serialize.query('language', language, 'str')
        if model_version is not None:
            query_parameters['model-version'] = self._serialize.query('model_version', model_version, 'str', pattern='^(latest|\\d{4}-\\d{2}-\\d{2})(-preview)?$')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(image_url, 'ImageUrl')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ComputerVisionErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('DomainModelResults', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    analyze_image_by_domain.metadata = {'url': '/models/{model}/analyze'}

    def recognize_printed_text(self, url, detect_orientation=True, language='unk', model_version='latest', custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Optical Character Recognition (OCR) detects text in an image and\n        extracts the recognized characters into a machine-usable character\n        stream.\n        Upon success, the OCR results will be returned.\n        Upon failure, the error code together with an error message will be\n        returned. The error code can be one of InvalidImageUrl,\n        InvalidImageFormat, InvalidImageSize, NotSupportedImage,\n        NotSupportedLanguage, or InternalServerError.\n\n        :param detect_orientation: Whether detect the text orientation in the\n         image. With detectOrientation=true the OCR service tries to detect the\n         image orientation and correct it before further processing (e.g. if\n         it\'s upside-down).\n        :type detect_orientation: bool\n        :param url: Publicly reachable URL of an image.\n        :type url: str\n        :param language: The BCP-47 language code of the text to be detected\n         in the image. The default value is \'unk\'. Possible values include:\n         \'unk\', \'zh-Hans\', \'zh-Hant\', \'cs\', \'da\', \'nl\', \'en\', \'fi\', \'fr\', \'de\',\n         \'el\', \'hu\', \'it\', \'ja\', \'ko\', \'nb\', \'pl\', \'pt\', \'ru\', \'es\', \'sv\',\n         \'tr\', \'ar\', \'ro\', \'sr-Cyrl\', \'sr-Latn\', \'sk\'\n        :type language: str or\n         ~azure.cognitiveservices.vision.computervision.models.OcrLanguages\n        :param model_version: Optional parameter to specify the version of the\n         AI model. Accepted values are: "latest", "2021-04-01". Defaults to\n         "latest".\n        :type model_version: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: OcrResult or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.computervision.models.OcrResult or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ComputerVisionErrorResponseException<azure.cognitiveservices.vision.computervision.models.ComputerVisionErrorResponseException>`\n        '
        image_url = models.ImageUrl(url=url)
        url = self.recognize_printed_text.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['detectOrientation'] = self._serialize.query('detect_orientation', detect_orientation, 'bool')
        if language is not None:
            query_parameters['language'] = self._serialize.query('language', language, 'OcrLanguages')
        if model_version is not None:
            query_parameters['model-version'] = self._serialize.query('model_version', model_version, 'str', pattern='^(latest|\\d{4}-\\d{2}-\\d{2})(-preview)?$')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(image_url, 'ImageUrl')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ComputerVisionErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('OcrResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    recognize_printed_text.metadata = {'url': '/ocr'}

    def tag_image(self, url, language='en', model_version='latest', custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'This operation generates a list of words, or tags, that are relevant to\n        the content of the supplied image. The Computer Vision API can return\n        tags based on objects, living beings, scenery or actions found in\n        images. Unlike categories, tags are not organized according to a\n        hierarchical classification system, but correspond to image content.\n        Tags may contain hints to avoid ambiguity or provide context, for\n        example the tag "ascomycete" may be accompanied by the hint "fungus".\n        Two input methods are supported -- (1) Uploading an image or (2)\n        specifying an image URL.\n        A successful response will be returned in JSON. If the request failed,\n        the response will contain an error code and a message to help\n        understand what went wrong.\n\n        :param url: Publicly reachable URL of an image.\n        :type url: str\n        :param language: The desired language for output generation. If this\n         parameter is not specified, the default value is\n         &quot;en&quot;.Supported languages:en - English, Default. es -\n         Spanish, ja - Japanese, pt - Portuguese, zh - Simplified Chinese.\n         Possible values include: \'en\', \'es\', \'ja\', \'pt\', \'zh\'\n        :type language: str\n        :param model_version: Optional parameter to specify the version of the\n         AI model. Accepted values are: "latest", "2021-04-01". Defaults to\n         "latest".\n        :type model_version: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: TagResult or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.computervision.models.TagResult or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ComputerVisionErrorResponseException<azure.cognitiveservices.vision.computervision.models.ComputerVisionErrorResponseException>`\n        '
        image_url = models.ImageUrl(url=url)
        url = self.tag_image.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if language is not None:
            query_parameters['language'] = self._serialize.query('language', language, 'str')
        if model_version is not None:
            query_parameters['model-version'] = self._serialize.query('model_version', model_version, 'str', pattern='^(latest|\\d{4}-\\d{2}-\\d{2})(-preview)?$')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(image_url, 'ImageUrl')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ComputerVisionErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('TagResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    tag_image.metadata = {'url': '/tag'}

    def generate_thumbnail(self, width, height, url, smart_cropping=False, model_version='latest', custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            while True:
                i = 10
        'This operation generates a thumbnail image with the user-specified\n        width and height. By default, the service analyzes the image,\n        identifies the region of interest (ROI), and generates smart cropping\n        coordinates based on the ROI. Smart cropping helps when you specify an\n        aspect ratio that differs from that of the input image.\n        A successful response contains the thumbnail image binary. If the\n        request failed, the response contains an error code and a message to\n        help determine what went wrong.\n        Upon failure, the error code and an error message are returned. The\n        error code could be one of InvalidImageUrl, InvalidImageFormat,\n        InvalidImageSize, InvalidThumbnailSize, NotSupportedImage,\n        FailedToProcess, Timeout, or InternalServerError.\n\n        :param width: Width of the thumbnail, in pixels. It must be between 1\n         and 1024. Recommended minimum of 50.\n        :type width: int\n        :param height: Height of the thumbnail, in pixels. It must be between\n         1 and 1024. Recommended minimum of 50.\n        :type height: int\n        :param url: Publicly reachable URL of an image.\n        :type url: str\n        :param smart_cropping: Boolean flag for enabling smart cropping.\n        :type smart_cropping: bool\n        :param model_version: Optional parameter to specify the version of the\n         AI model. Accepted values are: "latest", "2021-04-01". Defaults to\n         "latest".\n        :type model_version: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: object or ClientRawResponse if raw=true\n        :rtype: Generator or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`HttpOperationError<msrest.exceptions.HttpOperationError>`\n        '
        image_url = models.ImageUrl(url=url)
        url = self.generate_thumbnail.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['width'] = self._serialize.query('width', width, 'int', maximum=1024, minimum=1)
        query_parameters['height'] = self._serialize.query('height', height, 'int', maximum=1024, minimum=1)
        if smart_cropping is not None:
            query_parameters['smartCropping'] = self._serialize.query('smart_cropping', smart_cropping, 'bool')
        if model_version is not None:
            query_parameters['model-version'] = self._serialize.query('model_version', model_version, 'str', pattern='^(latest|\\d{4}-\\d{2}-\\d{2})(-preview)?$')
        header_parameters = {}
        header_parameters['Accept'] = 'application/octet-stream'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(image_url, 'ImageUrl')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=True, **operation_config)
        if response.status_code not in [200]:
            raise HttpOperationError(self._deserialize, response)
        deserialized = self._client.stream_download(response, callback)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    generate_thumbnail.metadata = {'url': '/generateThumbnail'}

    def get_area_of_interest(self, url, model_version='latest', custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'This operation returns a bounding box around the most important area of\n        the image.\n        A successful response will be returned in JSON. If the request failed,\n        the response contains an error code and a message to help determine\n        what went wrong.\n        Upon failure, the error code and an error message are returned. The\n        error code could be one of InvalidImageUrl, InvalidImageFormat,\n        InvalidImageSize, NotSupportedImage, FailedToProcess, Timeout, or\n        InternalServerError.\n\n        :param url: Publicly reachable URL of an image.\n        :type url: str\n        :param model_version: Optional parameter to specify the version of the\n         AI model. Accepted values are: "latest", "2021-04-01". Defaults to\n         "latest".\n        :type model_version: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: AreaOfInterestResult or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.computervision.models.AreaOfInterestResult\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ComputerVisionErrorResponseException<azure.cognitiveservices.vision.computervision.models.ComputerVisionErrorResponseException>`\n        '
        image_url = models.ImageUrl(url=url)
        url = self.get_area_of_interest.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if model_version is not None:
            query_parameters['model-version'] = self._serialize.query('model_version', model_version, 'str', pattern='^(latest|\\d{4}-\\d{2}-\\d{2})(-preview)?$')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(image_url, 'ImageUrl')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ComputerVisionErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('AreaOfInterestResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_area_of_interest.metadata = {'url': '/areaOfInterest'}

    def read(self, url, language=None, pages=None, model_version='latest', reading_order='basic', custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Use this interface to get the result of a Read operation, employing the\n        state-of-the-art Optical Character Recognition (OCR) algorithms\n        optimized for text-heavy documents. When you use the Read interface,\n        the response contains a field called \'Operation-Location\'. The\n        \'Operation-Location\' field contains the URL that you must use for your\n        \'GetReadResult\' operation to access OCR results.\u200b.\n\n        :param url: Publicly reachable URL of an image.\n        :type url: str\n        :param language: The BCP-47 language code of the text in the document.\n         Read supports auto language identification and multi-language\n         documents, so only provide a language code if you would like to force\n         the document to be processed in that specific language. See\n         https://aka.ms/ocr-languages for list of supported languages. Possible\n         values include: \'af\', \'ast\', \'bi\', \'br\', \'ca\', \'ceb\', \'ch\', \'co\',\n         \'crh\', \'cs\', \'csb\', \'da\', \'de\', \'en\', \'es\', \'et\', \'eu\', \'fi\', \'fil\',\n         \'fj\', \'fr\', \'fur\', \'fy\', \'ga\', \'gd\', \'gil\', \'gl\', \'gv\', \'hni\', \'hsb\',\n         \'ht\', \'hu\', \'ia\', \'id\', \'it\', \'iu\', \'ja\', \'jv\', \'kaa\', \'kac\', \'kea\',\n         \'kha\', \'kl\', \'ko\', \'ku\', \'kw\', \'lb\', \'ms\', \'mww\', \'nap\', \'nl\', \'no\',\n         \'oc\', \'pl\', \'pt\', \'quc\', \'rm\', \'sco\', \'sl\', \'sq\', \'sv\', \'sw\', \'tet\',\n         \'tr\', \'tt\', \'uz\', \'vo\', \'wae\', \'yua\', \'za\', \'zh-Hans\', \'zh-Hant\', \'zu\'\n        :type language: str or\n         ~azure.cognitiveservices.vision.computervision.models.OcrDetectionLanguage\n        :param pages: Custom page numbers for multi-page documents(PDF/TIFF),\n         input the number of the pages you want to get OCR result. For a range\n         of pages, use a hyphen. Separate each page or range with a comma.\n        :type pages: list[str]\n        :param model_version: Optional parameter to specify the version of the\n         OCR model used for text extraction. Accepted values are: "latest",\n         "latest-preview", "2021-04-12". Defaults to "latest".\n        :type model_version: str\n        :param reading_order: Optional parameter to specify which reading\n         order algorithm should be applied when ordering the extract text\n         elements. Can be either \'basic\' or \'natural\'. Will default to \'basic\'\n         if not specified\n        :type reading_order: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ComputerVisionOcrErrorException<azure.cognitiveservices.vision.computervision.models.ComputerVisionOcrErrorException>`\n        '
        image_url = models.ImageUrl(url=url)
        url = self.read.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if language is not None:
            query_parameters['language'] = self._serialize.query('language', language, 'str')
        if pages is not None:
            query_parameters['pages'] = self._serialize.query('pages', pages, '[str]', div=',')
        if model_version is not None:
            query_parameters['model-version'] = self._serialize.query('model_version', model_version, 'str', pattern='^(latest|\\d{4}-\\d{2}-\\d{2})(-preview)?$')
        if reading_order is not None:
            query_parameters['readingOrder'] = self._serialize.query('reading_order', reading_order, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(image_url, 'ImageUrl')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [202]:
            raise models.ComputerVisionOcrErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'Operation-Location': 'str'})
            return client_raw_response
    read.metadata = {'url': '/read/analyze'}

    def get_read_result(self, operation_id, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        "This interface is used for getting OCR results of Read operation. The\n        URL to this interface should be retrieved from 'Operation-Location'\n        field returned from Read interface.\n\n        :param operation_id: Id of read operation returned in the response of\n         the 'Read' interface.\n        :type operation_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ReadOperationResult or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.computervision.models.ReadOperationResult\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ComputerVisionOcrErrorException<azure.cognitiveservices.vision.computervision.models.ComputerVisionOcrErrorException>`\n        "
        url = self.get_read_result.metadata['url']
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
            raise models.ComputerVisionOcrErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ReadOperationResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_read_result.metadata = {'url': '/read/analyzeResults/{operationId}'}

    def analyze_image_in_stream(self, image, visual_features=None, details=None, language='en', description_exclude=None, model_version='latest', custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            i = 10
            return i + 15
        'This operation extracts a rich set of visual features based on the\n        image content.\n        Two input methods are supported -- (1) Uploading an image or (2)\n        specifying an image URL. Within your request, there is an optional\n        parameter to allow you to choose which features to return. By default,\n        image categories are returned in the response.\n        A successful response will be returned in JSON. If the request failed,\n        the response will contain an error code and a message to help\n        understand what went wrong.\n\n        :param image: An image stream.\n        :type image: Generator\n        :param visual_features: A string indicating what visual feature types\n         to return. Multiple values should be comma-separated. Valid visual\n         feature types include: Categories - categorizes image content\n         according to a taxonomy defined in documentation. Tags - tags the\n         image with a detailed list of words related to the image content.\n         Description - describes the image content with a complete English\n         sentence. Faces - detects if faces are present. If present, generate\n         coordinates, gender and age. ImageType - detects if image is clipart\n         or a line drawing. Color - determines the accent color, dominant\n         color, and whether an image is black&white. Adult - detects if the\n         image is pornographic in nature (depicts nudity or a sex act), or is\n         gory (depicts extreme violence or blood). Sexually suggestive content\n         (aka racy content) is also detected. Objects - detects various objects\n         within an image, including the approximate location. The Objects\n         argument is only available in English. Brands - detects various brands\n         within an image, including the approximate location. The Brands\n         argument is only available in English.\n        :type visual_features: list[str or\n         ~azure.cognitiveservices.vision.computervision.models.VisualFeatureTypes]\n        :param details: A string indicating which domain-specific details to\n         return. Multiple values should be comma-separated. Valid visual\n         feature types include: Celebrities - identifies celebrities if\n         detected in the image, Landmarks - identifies notable landmarks in the\n         image.\n        :type details: list[str or\n         ~azure.cognitiveservices.vision.computervision.models.Details]\n        :param language: The desired language for output generation. If this\n         parameter is not specified, the default value is\n         &quot;en&quot;.Supported languages:en - English, Default. es -\n         Spanish, ja - Japanese, pt - Portuguese, zh - Simplified Chinese.\n         Possible values include: \'en\', \'es\', \'ja\', \'pt\', \'zh\'\n        :type language: str\n        :param description_exclude: Turn off specified domain models when\n         generating the description.\n        :type description_exclude: list[str or\n         ~azure.cognitiveservices.vision.computervision.models.DescriptionExclude]\n        :param model_version: Optional parameter to specify the version of the\n         AI model. Accepted values are: "latest", "2021-04-01". Defaults to\n         "latest".\n        :type model_version: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ImageAnalysis or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.computervision.models.ImageAnalysis or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ComputerVisionErrorResponseException<azure.cognitiveservices.vision.computervision.models.ComputerVisionErrorResponseException>`\n        '
        url = self.analyze_image_in_stream.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if visual_features is not None:
            query_parameters['visualFeatures'] = self._serialize.query('visual_features', visual_features, '[VisualFeatureTypes]', div=',')
        if details is not None:
            query_parameters['details'] = self._serialize.query('details', details, '[Details]', div=',')
        if language is not None:
            query_parameters['language'] = self._serialize.query('language', language, 'str')
        if description_exclude is not None:
            query_parameters['descriptionExclude'] = self._serialize.query('description_exclude', description_exclude, '[DescriptionExclude]', div=',')
        if model_version is not None:
            query_parameters['model-version'] = self._serialize.query('model_version', model_version, 'str', pattern='^(latest|\\d{4}-\\d{2}-\\d{2})(-preview)?$')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/octet-stream'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._client.stream_upload(image, callback)
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ComputerVisionErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ImageAnalysis', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    analyze_image_in_stream.metadata = {'url': '/analyze'}

    def get_area_of_interest_in_stream(self, image, model_version='latest', custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            i = 10
            return i + 15
        'This operation returns a bounding box around the most important area of\n        the image.\n        A successful response will be returned in JSON. If the request failed,\n        the response contains an error code and a message to help determine\n        what went wrong.\n        Upon failure, the error code and an error message are returned. The\n        error code could be one of InvalidImageUrl, InvalidImageFormat,\n        InvalidImageSize, NotSupportedImage, FailedToProcess, Timeout, or\n        InternalServerError.\n\n        :param image: An image stream.\n        :type image: Generator\n        :param model_version: Optional parameter to specify the version of the\n         AI model. Accepted values are: "latest", "2021-04-01". Defaults to\n         "latest".\n        :type model_version: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: AreaOfInterestResult or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.computervision.models.AreaOfInterestResult\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ComputerVisionErrorResponseException<azure.cognitiveservices.vision.computervision.models.ComputerVisionErrorResponseException>`\n        '
        url = self.get_area_of_interest_in_stream.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if model_version is not None:
            query_parameters['model-version'] = self._serialize.query('model_version', model_version, 'str', pattern='^(latest|\\d{4}-\\d{2}-\\d{2})(-preview)?$')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/octet-stream'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._client.stream_upload(image, callback)
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ComputerVisionErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('AreaOfInterestResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_area_of_interest_in_stream.metadata = {'url': '/areaOfInterest'}

    def describe_image_in_stream(self, image, max_candidates=1, language='en', description_exclude=None, model_version='latest', custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            return 10
        'This operation generates a description of an image in human readable\n        language with complete sentences. The description is based on a\n        collection of content tags, which are also returned by the operation.\n        More than one description can be generated for each image. Descriptions\n        are ordered by their confidence score. Descriptions may include results\n        from celebrity and landmark domain models, if applicable.\n        Two input methods are supported -- (1) Uploading an image or (2)\n        specifying an image URL.\n        A successful response will be returned in JSON. If the request failed,\n        the response will contain an error code and a message to help\n        understand what went wrong.\n\n        :param image: An image stream.\n        :type image: Generator\n        :param max_candidates: Maximum number of candidate descriptions to be\n         returned.  The default is 1.\n        :type max_candidates: int\n        :param language: The desired language for output generation. If this\n         parameter is not specified, the default value is\n         &quot;en&quot;.Supported languages:en - English, Default. es -\n         Spanish, ja - Japanese, pt - Portuguese, zh - Simplified Chinese.\n         Possible values include: \'en\', \'es\', \'ja\', \'pt\', \'zh\'\n        :type language: str\n        :param description_exclude: Turn off specified domain models when\n         generating the description.\n        :type description_exclude: list[str or\n         ~azure.cognitiveservices.vision.computervision.models.DescriptionExclude]\n        :param model_version: Optional parameter to specify the version of the\n         AI model. Accepted values are: "latest", "2021-04-01". Defaults to\n         "latest".\n        :type model_version: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ImageDescription or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.computervision.models.ImageDescription\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ComputerVisionErrorResponseException<azure.cognitiveservices.vision.computervision.models.ComputerVisionErrorResponseException>`\n        '
        url = self.describe_image_in_stream.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if max_candidates is not None:
            query_parameters['maxCandidates'] = self._serialize.query('max_candidates', max_candidates, 'int')
        if language is not None:
            query_parameters['language'] = self._serialize.query('language', language, 'str')
        if description_exclude is not None:
            query_parameters['descriptionExclude'] = self._serialize.query('description_exclude', description_exclude, '[DescriptionExclude]', div=',')
        if model_version is not None:
            query_parameters['model-version'] = self._serialize.query('model_version', model_version, 'str', pattern='^(latest|\\d{4}-\\d{2}-\\d{2})(-preview)?$')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/octet-stream'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._client.stream_upload(image, callback)
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ComputerVisionErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ImageDescription', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    describe_image_in_stream.metadata = {'url': '/describe'}

    def detect_objects_in_stream(self, image, model_version='latest', custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            print('Hello World!')
        'Performs object detection on the specified image.\n        Two input methods are supported -- (1) Uploading an image or (2)\n        specifying an image URL.\n        A successful response will be returned in JSON. If the request failed,\n        the response will contain an error code and a message to help\n        understand what went wrong.\n\n        :param image: An image stream.\n        :type image: Generator\n        :param model_version: Optional parameter to specify the version of the\n         AI model. Accepted values are: "latest", "2021-04-01". Defaults to\n         "latest".\n        :type model_version: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: DetectResult or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.computervision.models.DetectResult or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ComputerVisionErrorResponseException<azure.cognitiveservices.vision.computervision.models.ComputerVisionErrorResponseException>`\n        '
        url = self.detect_objects_in_stream.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if model_version is not None:
            query_parameters['model-version'] = self._serialize.query('model_version', model_version, 'str', pattern='^(latest|\\d{4}-\\d{2}-\\d{2})(-preview)?$')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/octet-stream'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._client.stream_upload(image, callback)
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ComputerVisionErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('DetectResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    detect_objects_in_stream.metadata = {'url': '/detect'}

    def generate_thumbnail_in_stream(self, width, height, image, smart_cropping=False, model_version='latest', custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            print('Hello World!')
        'This operation generates a thumbnail image with the user-specified\n        width and height. By default, the service analyzes the image,\n        identifies the region of interest (ROI), and generates smart cropping\n        coordinates based on the ROI. Smart cropping helps when you specify an\n        aspect ratio that differs from that of the input image.\n        A successful response contains the thumbnail image binary. If the\n        request failed, the response contains an error code and a message to\n        help determine what went wrong.\n        Upon failure, the error code and an error message are returned. The\n        error code could be one of InvalidImageUrl, InvalidImageFormat,\n        InvalidImageSize, InvalidThumbnailSize, NotSupportedImage,\n        FailedToProcess, Timeout, or InternalServerError.\n\n        :param width: Width of the thumbnail, in pixels. It must be between 1\n         and 1024. Recommended minimum of 50.\n        :type width: int\n        :param height: Height of the thumbnail, in pixels. It must be between\n         1 and 1024. Recommended minimum of 50.\n        :type height: int\n        :param image: An image stream.\n        :type image: Generator\n        :param smart_cropping: Boolean flag for enabling smart cropping.\n        :type smart_cropping: bool\n        :param model_version: Optional parameter to specify the version of the\n         AI model. Accepted values are: "latest", "2021-04-01". Defaults to\n         "latest".\n        :type model_version: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: object or ClientRawResponse if raw=true\n        :rtype: Generator or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`HttpOperationError<msrest.exceptions.HttpOperationError>`\n        '
        url = self.generate_thumbnail_in_stream.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['width'] = self._serialize.query('width', width, 'int', maximum=1024, minimum=1)
        query_parameters['height'] = self._serialize.query('height', height, 'int', maximum=1024, minimum=1)
        if smart_cropping is not None:
            query_parameters['smartCropping'] = self._serialize.query('smart_cropping', smart_cropping, 'bool')
        if model_version is not None:
            query_parameters['model-version'] = self._serialize.query('model_version', model_version, 'str', pattern='^(latest|\\d{4}-\\d{2}-\\d{2})(-preview)?$')
        header_parameters = {}
        header_parameters['Accept'] = 'application/octet-stream'
        header_parameters['Content-Type'] = 'application/octet-stream'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._client.stream_upload(image, callback)
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=True, **operation_config)
        if response.status_code not in [200]:
            raise HttpOperationError(self._deserialize, response)
        deserialized = self._client.stream_download(response, callback)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    generate_thumbnail_in_stream.metadata = {'url': '/generateThumbnail'}

    def analyze_image_by_domain_in_stream(self, model, image, language='en', model_version='latest', custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'This operation recognizes content within an image by applying a\n        domain-specific model. The list of domain-specific models that are\n        supported by the Computer Vision API can be retrieved using the /models\n        GET request. Currently, the API provides following domain-specific\n        models: celebrities, landmarks.\n        Two input methods are supported -- (1) Uploading an image or (2)\n        specifying an image URL.\n        A successful response will be returned in JSON.\n        If the request failed, the response will contain an error code and a\n        message to help understand what went wrong.\n\n        :param model: The domain-specific content to recognize.\n        :type model: str\n        :param image: An image stream.\n        :type image: Generator\n        :param language: The desired language for output generation. If this\n         parameter is not specified, the default value is\n         &quot;en&quot;.Supported languages:en - English, Default. es -\n         Spanish, ja - Japanese, pt - Portuguese, zh - Simplified Chinese.\n         Possible values include: \'en\', \'es\', \'ja\', \'pt\', \'zh\'\n        :type language: str\n        :param model_version: Optional parameter to specify the version of the\n         AI model. Accepted values are: "latest", "2021-04-01". Defaults to\n         "latest".\n        :type model_version: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: DomainModelResults or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.computervision.models.DomainModelResults\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ComputerVisionErrorResponseException<azure.cognitiveservices.vision.computervision.models.ComputerVisionErrorResponseException>`\n        '
        url = self.analyze_image_by_domain_in_stream.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'model': self._serialize.url('model', model, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if language is not None:
            query_parameters['language'] = self._serialize.query('language', language, 'str')
        if model_version is not None:
            query_parameters['model-version'] = self._serialize.query('model_version', model_version, 'str', pattern='^(latest|\\d{4}-\\d{2}-\\d{2})(-preview)?$')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/octet-stream'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._client.stream_upload(image, callback)
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ComputerVisionErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('DomainModelResults', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    analyze_image_by_domain_in_stream.metadata = {'url': '/models/{model}/analyze'}

    def recognize_printed_text_in_stream(self, image, detect_orientation=True, language='unk', model_version='latest', custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Optical Character Recognition (OCR) detects text in an image and\n        extracts the recognized characters into a machine-usable character\n        stream.\n        Upon success, the OCR results will be returned.\n        Upon failure, the error code together with an error message will be\n        returned. The error code can be one of InvalidImageUrl,\n        InvalidImageFormat, InvalidImageSize, NotSupportedImage,\n        NotSupportedLanguage, or InternalServerError.\n\n        :param detect_orientation: Whether detect the text orientation in the\n         image. With detectOrientation=true the OCR service tries to detect the\n         image orientation and correct it before further processing (e.g. if\n         it\'s upside-down).\n        :type detect_orientation: bool\n        :param image: An image stream.\n        :type image: Generator\n        :param language: The BCP-47 language code of the text to be detected\n         in the image. The default value is \'unk\'. Possible values include:\n         \'unk\', \'zh-Hans\', \'zh-Hant\', \'cs\', \'da\', \'nl\', \'en\', \'fi\', \'fr\', \'de\',\n         \'el\', \'hu\', \'it\', \'ja\', \'ko\', \'nb\', \'pl\', \'pt\', \'ru\', \'es\', \'sv\',\n         \'tr\', \'ar\', \'ro\', \'sr-Cyrl\', \'sr-Latn\', \'sk\'\n        :type language: str or\n         ~azure.cognitiveservices.vision.computervision.models.OcrLanguages\n        :param model_version: Optional parameter to specify the version of the\n         AI model. Accepted values are: "latest", "2021-04-01". Defaults to\n         "latest".\n        :type model_version: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: OcrResult or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.computervision.models.OcrResult or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ComputerVisionErrorResponseException<azure.cognitiveservices.vision.computervision.models.ComputerVisionErrorResponseException>`\n        '
        url = self.recognize_printed_text_in_stream.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['detectOrientation'] = self._serialize.query('detect_orientation', detect_orientation, 'bool')
        if language is not None:
            query_parameters['language'] = self._serialize.query('language', language, 'OcrLanguages')
        if model_version is not None:
            query_parameters['model-version'] = self._serialize.query('model_version', model_version, 'str', pattern='^(latest|\\d{4}-\\d{2}-\\d{2})(-preview)?$')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/octet-stream'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._client.stream_upload(image, callback)
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ComputerVisionErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('OcrResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    recognize_printed_text_in_stream.metadata = {'url': '/ocr'}

    def tag_image_in_stream(self, image, language='en', model_version='latest', custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            i = 10
            return i + 15
        'This operation generates a list of words, or tags, that are relevant to\n        the content of the supplied image. The Computer Vision API can return\n        tags based on objects, living beings, scenery or actions found in\n        images. Unlike categories, tags are not organized according to a\n        hierarchical classification system, but correspond to image content.\n        Tags may contain hints to avoid ambiguity or provide context, for\n        example the tag "ascomycete" may be accompanied by the hint "fungus".\n        Two input methods are supported -- (1) Uploading an image or (2)\n        specifying an image URL.\n        A successful response will be returned in JSON. If the request failed,\n        the response will contain an error code and a message to help\n        understand what went wrong.\n\n        :param image: An image stream.\n        :type image: Generator\n        :param language: The desired language for output generation. If this\n         parameter is not specified, the default value is\n         &quot;en&quot;.Supported languages:en - English, Default. es -\n         Spanish, ja - Japanese, pt - Portuguese, zh - Simplified Chinese.\n         Possible values include: \'en\', \'es\', \'ja\', \'pt\', \'zh\'\n        :type language: str\n        :param model_version: Optional parameter to specify the version of the\n         AI model. Accepted values are: "latest", "2021-04-01". Defaults to\n         "latest".\n        :type model_version: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: TagResult or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.computervision.models.TagResult or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ComputerVisionErrorResponseException<azure.cognitiveservices.vision.computervision.models.ComputerVisionErrorResponseException>`\n        '
        url = self.tag_image_in_stream.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if language is not None:
            query_parameters['language'] = self._serialize.query('language', language, 'str')
        if model_version is not None:
            query_parameters['model-version'] = self._serialize.query('model_version', model_version, 'str', pattern='^(latest|\\d{4}-\\d{2}-\\d{2})(-preview)?$')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/octet-stream'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._client.stream_upload(image, callback)
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ComputerVisionErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('TagResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    tag_image_in_stream.metadata = {'url': '/tag'}

    def read_in_stream(self, image, language=None, pages=None, model_version='latest', reading_order='basic', custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Use this interface to get the result of a Read operation, employing the\n        state-of-the-art Optical Character Recognition (OCR) algorithms\n        optimized for text-heavy documents. When you use the Read interface,\n        the response contains a field called \'Operation-Location\'. The\n        \'Operation-Location\' field contains the URL that you must use for your\n        \'GetReadResult\' operation to access OCR results.\u200b.\n\n        :param image: An image stream.\n        :type image: Generator\n        :param language: The BCP-47 language code of the text in the document.\n         Read supports auto language identification and multi-language\n         documents, so only provide a language code if you would like to force\n         the document to be processed in that specific language. See\n         https://aka.ms/ocr-languages for list of supported languages. Possible\n         values include: \'af\', \'ast\', \'bi\', \'br\', \'ca\', \'ceb\', \'ch\', \'co\',\n         \'crh\', \'cs\', \'csb\', \'da\', \'de\', \'en\', \'es\', \'et\', \'eu\', \'fi\', \'fil\',\n         \'fj\', \'fr\', \'fur\', \'fy\', \'ga\', \'gd\', \'gil\', \'gl\', \'gv\', \'hni\', \'hsb\',\n         \'ht\', \'hu\', \'ia\', \'id\', \'it\', \'iu\', \'ja\', \'jv\', \'kaa\', \'kac\', \'kea\',\n         \'kha\', \'kl\', \'ko\', \'ku\', \'kw\', \'lb\', \'ms\', \'mww\', \'nap\', \'nl\', \'no\',\n         \'oc\', \'pl\', \'pt\', \'quc\', \'rm\', \'sco\', \'sl\', \'sq\', \'sv\', \'sw\', \'tet\',\n         \'tr\', \'tt\', \'uz\', \'vo\', \'wae\', \'yua\', \'za\', \'zh-Hans\', \'zh-Hant\', \'zu\'\n        :type language: str or\n         ~azure.cognitiveservices.vision.computervision.models.OcrDetectionLanguage\n        :param pages: Custom page numbers for multi-page documents(PDF/TIFF),\n         input the number of the pages you want to get OCR result. For a range\n         of pages, use a hyphen. Separate each page or range with a comma.\n        :type pages: list[str]\n        :param model_version: Optional parameter to specify the version of the\n         OCR model used for text extraction. Accepted values are: "latest",\n         "latest-preview", "2021-04-12". Defaults to "latest".\n        :type model_version: str\n        :param reading_order: Optional parameter to specify which reading\n         order algorithm should be applied when ordering the extract text\n         elements. Can be either \'basic\' or \'natural\'. Will default to \'basic\'\n         if not specified\n        :type reading_order: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ComputerVisionOcrErrorException<azure.cognitiveservices.vision.computervision.models.ComputerVisionOcrErrorException>`\n        '
        url = self.read_in_stream.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if language is not None:
            query_parameters['language'] = self._serialize.query('language', language, 'str')
        if pages is not None:
            query_parameters['pages'] = self._serialize.query('pages', pages, '[str]', div=',')
        if model_version is not None:
            query_parameters['model-version'] = self._serialize.query('model_version', model_version, 'str', pattern='^(latest|\\d{4}-\\d{2}-\\d{2})(-preview)?$')
        if reading_order is not None:
            query_parameters['readingOrder'] = self._serialize.query('reading_order', reading_order, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/octet-stream'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._client.stream_upload(image, callback)
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [202]:
            raise models.ComputerVisionOcrErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            client_raw_response.add_headers({'Operation-Location': 'str'})
            return client_raw_response
    read_in_stream.metadata = {'url': '/read/analyze'}