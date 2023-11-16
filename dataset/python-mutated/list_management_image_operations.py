from msrest.pipeline import ClientRawResponse
from .. import models

class ListManagementImageOperations(object):
    """ListManagementImageOperations operations.

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

    def add_image(self, list_id, tag=None, label=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Add an image to the list with list Id equal to list Id passed.\n\n        :param list_id: List Id of the image list.\n        :type list_id: str\n        :param tag: Tag for the image.\n        :type tag: int\n        :param label: The image label.\n        :type label: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Image or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.contentmoderator.models.Image\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.add_image.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'listId': self._serialize.url('list_id', list_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if tag is not None:
            query_parameters['tag'] = self._serialize.query('tag', tag, 'int')
        if label is not None:
            query_parameters['label'] = self._serialize.query('label', label, 'str')
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
            deserialized = self._deserialize('Image', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    add_image.metadata = {'url': '/contentmoderator/lists/v1.0/imagelists/{listId}/images'}

    def delete_all_images(self, list_id, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Deletes all images from the list with list Id equal to list Id passed.\n\n        :param list_id: List Id of the image list.\n        :type list_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: str or ClientRawResponse if raw=true\n        :rtype: str or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.delete_all_images.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'listId': self._serialize.url('list_id', list_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.delete(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('str', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    delete_all_images.metadata = {'url': '/contentmoderator/lists/v1.0/imagelists/{listId}/images'}

    def get_all_image_ids(self, list_id, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Gets all image Ids from the list with list Id equal to list Id passed.\n\n        :param list_id: List Id of the image list.\n        :type list_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ImageIds or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.contentmoderator.models.ImageIds or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.get_all_image_ids.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'listId': self._serialize.url('list_id', list_id, 'str')}
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
            deserialized = self._deserialize('ImageIds', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_all_image_ids.metadata = {'url': '/contentmoderator/lists/v1.0/imagelists/{listId}/images'}

    def delete_image(self, list_id, image_id, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Deletes an image from the list with list Id and image Id passed.\n\n        :param list_id: List Id of the image list.\n        :type list_id: str\n        :param image_id: Id of the image.\n        :type image_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: str or ClientRawResponse if raw=true\n        :rtype: str or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.delete_image.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'listId': self._serialize.url('list_id', list_id, 'str'), 'ImageId': self._serialize.url('image_id', image_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.delete(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('str', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    delete_image.metadata = {'url': '/contentmoderator/lists/v1.0/imagelists/{listId}/images/{ImageId}'}

    def add_image_url_input(self, list_id, content_type, tag=None, label=None, data_representation='URL', value=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Add an image to the list with list Id equal to list Id passed.\n\n        :param list_id: List Id of the image list.\n        :type list_id: str\n        :param content_type: The content type.\n        :type content_type: str\n        :param tag: Tag for the image.\n        :type tag: int\n        :param label: The image label.\n        :type label: str\n        :param data_representation:\n        :type data_representation: str\n        :param value:\n        :type value: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Image or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.contentmoderator.models.Image\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        image_url = models.BodyModel(data_representation=data_representation, value=value)
        url = self.add_image_url_input.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'listId': self._serialize.url('list_id', list_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if tag is not None:
            query_parameters['tag'] = self._serialize.query('tag', tag, 'int')
        if label is not None:
            query_parameters['label'] = self._serialize.query('label', label, 'str')
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
            deserialized = self._deserialize('Image', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    add_image_url_input.metadata = {'url': '/contentmoderator/lists/v1.0/imagelists/{listId}/images'}

    def add_image_file_input(self, list_id, image_stream, tag=None, label=None, custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            return 10
        'Add an image to the list with list Id equal to list Id passed.\n\n        :param list_id: List Id of the image list.\n        :type list_id: str\n        :param image_stream: The image file.\n        :type image_stream: Generator\n        :param tag: Tag for the image.\n        :type tag: int\n        :param label: The image label.\n        :type label: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Image or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.contentmoderator.models.Image\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.add_image_file_input.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'listId': self._serialize.url('list_id', list_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if tag is not None:
            query_parameters['tag'] = self._serialize.query('tag', tag, 'int')
        if label is not None:
            query_parameters['label'] = self._serialize.query('label', label, 'str')
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
            deserialized = self._deserialize('Image', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    add_image_file_input.metadata = {'url': '/contentmoderator/lists/v1.0/imagelists/{listId}/images'}