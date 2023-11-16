from msrest.pipeline import ClientRawResponse
from .. import models

class ImageModerationOperations(object):
    """ImageModerationOperations operations.

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

    def find_faces(self, cache_image=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Returns the list of faces found.\n\n        :param cache_image: Whether to retain the submitted image for future\n         use; defaults to false if omitted.\n        :type cache_image: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: FoundFaces or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.contentmoderator.models.FoundFaces or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.find_faces.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if cache_image is not None:
            query_parameters['CacheImage'] = self._serialize.query('cache_image', cache_image, 'bool')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.post(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('FoundFaces', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    find_faces.metadata = {'url': '/contentmoderator/moderate/v1.0/ProcessImage/FindFaces'}

    def ocr_method(self, language, cache_image=None, enhanced=False, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Returns any text found in the image for the language specified. If no\n        language is specified in input then the detection defaults to English.\n\n        :param language: Language of the terms.\n        :type language: str\n        :param cache_image: Whether to retain the submitted image for future\n         use; defaults to false if omitted.\n        :type cache_image: bool\n        :param enhanced: When set to True, the image goes through additional\n         processing to come with additional candidates.\n         image/tiff is not supported when enhanced is set to true\n         Note: This impacts the response time.\n        :type enhanced: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: OCR or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.contentmoderator.models.OCR or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.ocr_method.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['language'] = self._serialize.query('language', language, 'str')
        if cache_image is not None:
            query_parameters['CacheImage'] = self._serialize.query('cache_image', cache_image, 'bool')
        if enhanced is not None:
            query_parameters['enhanced'] = self._serialize.query('enhanced', enhanced, 'bool')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.post(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('OCR', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    ocr_method.metadata = {'url': '/contentmoderator/moderate/v1.0/ProcessImage/OCR'}

    def evaluate_method(self, cache_image=None, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Returns probabilities of the image containing racy or adult content.\n\n        :param cache_image: Whether to retain the submitted image for future\n         use; defaults to false if omitted.\n        :type cache_image: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Evaluate or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.contentmoderator.models.Evaluate or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.evaluate_method.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if cache_image is not None:
            query_parameters['CacheImage'] = self._serialize.query('cache_image', cache_image, 'bool')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.post(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('Evaluate', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    evaluate_method.metadata = {'url': '/contentmoderator/moderate/v1.0/ProcessImage/Evaluate'}

    def match_method(self, list_id=None, cache_image=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Fuzzily match an image against one of your custom Image Lists. You can\n        create and manage your custom image lists using <a\n        href="/docs/services/578ff44d2703741568569ab9/operations/578ff7b12703741568569abe">this</a>\n        API.\n        Returns ID and tags of matching image.<br/>\n        <br/>\n        Note: Refresh Index must be run on the corresponding Image List before\n        additions and removals are reflected in the response.\n\n        :param list_id: The list Id.\n        :type list_id: str\n        :param cache_image: Whether to retain the submitted image for future\n         use; defaults to false if omitted.\n        :type cache_image: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: MatchResponse or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.contentmoderator.models.MatchResponse\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.match_method.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if list_id is not None:
            query_parameters['listId'] = self._serialize.query('list_id', list_id, 'str')
        if cache_image is not None:
            query_parameters['CacheImage'] = self._serialize.query('cache_image', cache_image, 'bool')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.post(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('MatchResponse', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    match_method.metadata = {'url': '/contentmoderator/moderate/v1.0/ProcessImage/Match'}

    def find_faces_file_input(self, image_stream, cache_image=None, custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            print('Hello World!')
        'Returns the list of faces found.\n\n        :param image_stream: The image file.\n        :type image_stream: Generator\n        :param cache_image: Whether to retain the submitted image for future\n         use; defaults to false if omitted.\n        :type cache_image: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: FoundFaces or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.contentmoderator.models.FoundFaces or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.find_faces_file_input.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if cache_image is not None:
            query_parameters['CacheImage'] = self._serialize.query('cache_image', cache_image, 'bool')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'image/gif'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._client.stream_upload(image_stream, callback)
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('FoundFaces', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    find_faces_file_input.metadata = {'url': '/contentmoderator/moderate/v1.0/ProcessImage/FindFaces'}

    def find_faces_url_input(self, content_type, cache_image=None, data_representation='URL', value=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Returns the list of faces found.\n\n        :param content_type: The content type.\n        :type content_type: str\n        :param cache_image: Whether to retain the submitted image for future\n         use; defaults to false if omitted.\n        :type cache_image: bool\n        :param data_representation:\n        :type data_representation: str\n        :param value:\n        :type value: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: FoundFaces or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.contentmoderator.models.FoundFaces or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        image_url = models.BodyModel(data_representation=data_representation, value=value)
        url = self.find_faces_url_input.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if cache_image is not None:
            query_parameters['CacheImage'] = self._serialize.query('cache_image', cache_image, 'bool')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        body_content = self._serialize.body(image_url, 'BodyModel')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('FoundFaces', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    find_faces_url_input.metadata = {'url': '/contentmoderator/moderate/v1.0/ProcessImage/FindFaces'}

    def ocr_url_input(self, language, content_type, cache_image=None, enhanced=False, data_representation='URL', value=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Returns any text found in the image for the language specified. If no\n        language is specified in input then the detection defaults to English.\n\n        :param language: Language of the terms.\n        :type language: str\n        :param content_type: The content type.\n        :type content_type: str\n        :param cache_image: Whether to retain the submitted image for future\n         use; defaults to false if omitted.\n        :type cache_image: bool\n        :param enhanced: When set to True, the image goes through additional\n         processing to come with additional candidates.\n         image/tiff is not supported when enhanced is set to true\n         Note: This impacts the response time.\n        :type enhanced: bool\n        :param data_representation:\n        :type data_representation: str\n        :param value:\n        :type value: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: OCR or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.contentmoderator.models.OCR or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        image_url = models.BodyModel(data_representation=data_representation, value=value)
        url = self.ocr_url_input.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['language'] = self._serialize.query('language', language, 'str')
        if cache_image is not None:
            query_parameters['CacheImage'] = self._serialize.query('cache_image', cache_image, 'bool')
        if enhanced is not None:
            query_parameters['enhanced'] = self._serialize.query('enhanced', enhanced, 'bool')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        body_content = self._serialize.body(image_url, 'BodyModel')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('OCR', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    ocr_url_input.metadata = {'url': '/contentmoderator/moderate/v1.0/ProcessImage/OCR'}

    def ocr_file_input(self, language, image_stream, cache_image=None, enhanced=False, custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            return 10
        'Returns any text found in the image for the language specified. If no\n        language is specified in input then the detection defaults to English.\n\n        :param language: Language of the terms.\n        :type language: str\n        :param image_stream: The image file.\n        :type image_stream: Generator\n        :param cache_image: Whether to retain the submitted image for future\n         use; defaults to false if omitted.\n        :type cache_image: bool\n        :param enhanced: When set to True, the image goes through additional\n         processing to come with additional candidates.\n         image/tiff is not supported when enhanced is set to true\n         Note: This impacts the response time.\n        :type enhanced: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: OCR or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.contentmoderator.models.OCR or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.ocr_file_input.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['language'] = self._serialize.query('language', language, 'str')
        if cache_image is not None:
            query_parameters['CacheImage'] = self._serialize.query('cache_image', cache_image, 'bool')
        if enhanced is not None:
            query_parameters['enhanced'] = self._serialize.query('enhanced', enhanced, 'bool')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'image/gif'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._client.stream_upload(image_stream, callback)
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('OCR', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    ocr_file_input.metadata = {'url': '/contentmoderator/moderate/v1.0/ProcessImage/OCR'}

    def evaluate_file_input(self, image_stream, cache_image=None, custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            return 10
        'Returns probabilities of the image containing racy or adult content.\n\n        :param image_stream: The image file.\n        :type image_stream: Generator\n        :param cache_image: Whether to retain the submitted image for future\n         use; defaults to false if omitted.\n        :type cache_image: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Evaluate or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.contentmoderator.models.Evaluate or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.evaluate_file_input.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if cache_image is not None:
            query_parameters['CacheImage'] = self._serialize.query('cache_image', cache_image, 'bool')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'image/gif'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._client.stream_upload(image_stream, callback)
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('Evaluate', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    evaluate_file_input.metadata = {'url': '/contentmoderator/moderate/v1.0/ProcessImage/Evaluate'}

    def evaluate_url_input(self, content_type, cache_image=None, data_representation='URL', value=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Returns probabilities of the image containing racy or adult content.\n\n        :param content_type: The content type.\n        :type content_type: str\n        :param cache_image: Whether to retain the submitted image for future\n         use; defaults to false if omitted.\n        :type cache_image: bool\n        :param data_representation:\n        :type data_representation: str\n        :param value:\n        :type value: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Evaluate or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.contentmoderator.models.Evaluate or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        image_url = models.BodyModel(data_representation=data_representation, value=value)
        url = self.evaluate_url_input.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if cache_image is not None:
            query_parameters['CacheImage'] = self._serialize.query('cache_image', cache_image, 'bool')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        body_content = self._serialize.body(image_url, 'BodyModel')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('Evaluate', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    evaluate_url_input.metadata = {'url': '/contentmoderator/moderate/v1.0/ProcessImage/Evaluate'}

    def match_url_input(self, content_type, list_id=None, cache_image=None, data_representation='URL', value=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Fuzzily match an image against one of your custom Image Lists. You can\n        create and manage your custom image lists using <a\n        href="/docs/services/578ff44d2703741568569ab9/operations/578ff7b12703741568569abe">this</a>\n        API.\n        Returns ID and tags of matching image.<br/>\n        <br/>\n        Note: Refresh Index must be run on the corresponding Image List before\n        additions and removals are reflected in the response.\n\n        :param content_type: The content type.\n        :type content_type: str\n        :param list_id: The list Id.\n        :type list_id: str\n        :param cache_image: Whether to retain the submitted image for future\n         use; defaults to false if omitted.\n        :type cache_image: bool\n        :param data_representation:\n        :type data_representation: str\n        :param value:\n        :type value: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: MatchResponse or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.contentmoderator.models.MatchResponse\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        image_url = models.BodyModel(data_representation=data_representation, value=value)
        url = self.match_url_input.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if list_id is not None:
            query_parameters['listId'] = self._serialize.query('list_id', list_id, 'str')
        if cache_image is not None:
            query_parameters['CacheImage'] = self._serialize.query('cache_image', cache_image, 'bool')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        body_content = self._serialize.body(image_url, 'BodyModel')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('MatchResponse', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    match_url_input.metadata = {'url': '/contentmoderator/moderate/v1.0/ProcessImage/Match'}

    def match_file_input(self, image_stream, list_id=None, cache_image=None, custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            i = 10
            return i + 15
        'Fuzzily match an image against one of your custom Image Lists. You can\n        create and manage your custom image lists using <a\n        href="/docs/services/578ff44d2703741568569ab9/operations/578ff7b12703741568569abe">this</a>\n        API.\n        Returns ID and tags of matching image.<br/>\n        <br/>\n        Note: Refresh Index must be run on the corresponding Image List before\n        additions and removals are reflected in the response.\n\n        :param image_stream: The image file.\n        :type image_stream: Generator\n        :param list_id: The list Id.\n        :type list_id: str\n        :param cache_image: Whether to retain the submitted image for future\n         use; defaults to false if omitted.\n        :type cache_image: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: MatchResponse or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.contentmoderator.models.MatchResponse\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.match_file_input.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if list_id is not None:
            query_parameters['listId'] = self._serialize.query('list_id', list_id, 'str')
        if cache_image is not None:
            query_parameters['CacheImage'] = self._serialize.query('cache_image', cache_image, 'bool')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'image/gif'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._client.stream_upload(image_stream, callback)
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('MatchResponse', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    match_file_input.metadata = {'url': '/contentmoderator/moderate/v1.0/ProcessImage/Match'}