from msrest.pipeline import ClientRawResponse
from .. import models

class CustomVisionPredictionClientOperationsMixin(object):

    def classify_image(self, project_id, published_name, image_data, application=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Classify an image and saves the result.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param published_name: Specifies the name of the model to evaluate\n         against.\n        :type published_name: str\n        :param image_data: Binary image data. Supported formats are JPEG, GIF,\n         PNG, and BMP. Supports images up to 4MB.\n        :type image_data: Generator\n        :param application: Optional. Specifies the name of application using\n         the endpoint.\n        :type application: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ImagePrediction or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.prediction.models.ImagePrediction\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.prediction.models.CustomVisionErrorException>`\n        '
        url = self.classify_image.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str'), 'publishedName': self._serialize.url('published_name', published_name, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if application is not None:
            query_parameters['application'] = self._serialize.query('application', application, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'multipart/form-data'
        if custom_headers:
            header_parameters.update(custom_headers)
        form_data_content = {'imageData': image_data}
        request = self._client.post(url, query_parameters, header_parameters, form_content=form_data_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ImagePrediction', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    classify_image.metadata = {'url': '/{projectId}/classify/iterations/{publishedName}/image'}

    def classify_image_with_no_store(self, project_id, published_name, image_data, application=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Classify an image without saving the result.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param published_name: Specifies the name of the model to evaluate\n         against.\n        :type published_name: str\n        :param image_data: Binary image data. Supported formats are JPEG, GIF,\n         PNG, and BMP. Supports images up to 4MB.\n        :type image_data: Generator\n        :param application: Optional. Specifies the name of application using\n         the endpoint.\n        :type application: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ImagePrediction or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.prediction.models.ImagePrediction\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.prediction.models.CustomVisionErrorException>`\n        '
        url = self.classify_image_with_no_store.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str'), 'publishedName': self._serialize.url('published_name', published_name, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if application is not None:
            query_parameters['application'] = self._serialize.query('application', application, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'multipart/form-data'
        if custom_headers:
            header_parameters.update(custom_headers)
        form_data_content = {'imageData': image_data}
        request = self._client.post(url, query_parameters, header_parameters, form_content=form_data_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ImagePrediction', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    classify_image_with_no_store.metadata = {'url': '/{projectId}/classify/iterations/{publishedName}/image/nostore'}

    def classify_image_url(self, project_id, published_name, url, application=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Classify an image url and saves the result.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param published_name: Specifies the name of the model to evaluate\n         against.\n        :type published_name: str\n        :param url: Url of the image.\n        :type url: str\n        :param application: Optional. Specifies the name of application using\n         the endpoint.\n        :type application: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ImagePrediction or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.prediction.models.ImagePrediction\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.prediction.models.CustomVisionErrorException>`\n        '
        image_url = models.ImageUrl(url=url)
        url = self.classify_image_url.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str'), 'publishedName': self._serialize.url('published_name', published_name, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if application is not None:
            query_parameters['application'] = self._serialize.query('application', application, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(image_url, 'ImageUrl')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ImagePrediction', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    classify_image_url.metadata = {'url': '/{projectId}/classify/iterations/{publishedName}/url'}

    def classify_image_url_with_no_store(self, project_id, published_name, url, application=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Classify an image url without saving the result.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param published_name: Specifies the name of the model to evaluate\n         against.\n        :type published_name: str\n        :param url: Url of the image.\n        :type url: str\n        :param application: Optional. Specifies the name of application using\n         the endpoint.\n        :type application: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ImagePrediction or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.prediction.models.ImagePrediction\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.prediction.models.CustomVisionErrorException>`\n        '
        image_url = models.ImageUrl(url=url)
        url = self.classify_image_url_with_no_store.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str'), 'publishedName': self._serialize.url('published_name', published_name, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if application is not None:
            query_parameters['application'] = self._serialize.query('application', application, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(image_url, 'ImageUrl')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ImagePrediction', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    classify_image_url_with_no_store.metadata = {'url': '/{projectId}/classify/iterations/{publishedName}/url/nostore'}

    def detect_image(self, project_id, published_name, image_data, application=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Detect objects in an image and saves the result.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param published_name: Specifies the name of the model to evaluate\n         against.\n        :type published_name: str\n        :param image_data: Binary image data. Supported formats are JPEG, GIF,\n         PNG, and BMP. Supports images up to 4MB.\n        :type image_data: Generator\n        :param application: Optional. Specifies the name of application using\n         the endpoint.\n        :type application: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ImagePrediction or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.prediction.models.ImagePrediction\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.prediction.models.CustomVisionErrorException>`\n        '
        url = self.detect_image.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str'), 'publishedName': self._serialize.url('published_name', published_name, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if application is not None:
            query_parameters['application'] = self._serialize.query('application', application, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'multipart/form-data'
        if custom_headers:
            header_parameters.update(custom_headers)
        form_data_content = {'imageData': image_data}
        request = self._client.post(url, query_parameters, header_parameters, form_content=form_data_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ImagePrediction', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    detect_image.metadata = {'url': '/{projectId}/detect/iterations/{publishedName}/image'}

    def detect_image_with_no_store(self, project_id, published_name, image_data, application=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Detect objects in an image without saving the result.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param published_name: Specifies the name of the model to evaluate\n         against.\n        :type published_name: str\n        :param image_data: Binary image data. Supported formats are JPEG, GIF,\n         PNG, and BMP. Supports images up to 4MB.\n        :type image_data: Generator\n        :param application: Optional. Specifies the name of application using\n         the endpoint.\n        :type application: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ImagePrediction or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.prediction.models.ImagePrediction\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.prediction.models.CustomVisionErrorException>`\n        '
        url = self.detect_image_with_no_store.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str'), 'publishedName': self._serialize.url('published_name', published_name, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if application is not None:
            query_parameters['application'] = self._serialize.query('application', application, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'multipart/form-data'
        if custom_headers:
            header_parameters.update(custom_headers)
        form_data_content = {'imageData': image_data}
        request = self._client.post(url, query_parameters, header_parameters, form_content=form_data_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ImagePrediction', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    detect_image_with_no_store.metadata = {'url': '/{projectId}/detect/iterations/{publishedName}/image/nostore'}

    def detect_image_url(self, project_id, published_name, url, application=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Detect objects in an image url and saves the result.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param published_name: Specifies the name of the model to evaluate\n         against.\n        :type published_name: str\n        :param url: Url of the image.\n        :type url: str\n        :param application: Optional. Specifies the name of application using\n         the endpoint.\n        :type application: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ImagePrediction or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.prediction.models.ImagePrediction\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.prediction.models.CustomVisionErrorException>`\n        '
        image_url = models.ImageUrl(url=url)
        url = self.detect_image_url.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str'), 'publishedName': self._serialize.url('published_name', published_name, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if application is not None:
            query_parameters['application'] = self._serialize.query('application', application, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(image_url, 'ImageUrl')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ImagePrediction', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    detect_image_url.metadata = {'url': '/{projectId}/detect/iterations/{publishedName}/url'}

    def detect_image_url_with_no_store(self, project_id, published_name, url, application=None, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Detect objects in an image url without saving the result.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param published_name: Specifies the name of the model to evaluate\n         against.\n        :type published_name: str\n        :param url: Url of the image.\n        :type url: str\n        :param application: Optional. Specifies the name of application using\n         the endpoint.\n        :type application: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ImagePrediction or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.prediction.models.ImagePrediction\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.prediction.models.CustomVisionErrorException>`\n        '
        image_url = models.ImageUrl(url=url)
        url = self.detect_image_url_with_no_store.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str'), 'publishedName': self._serialize.url('published_name', published_name, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if application is not None:
            query_parameters['application'] = self._serialize.query('application', application, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(image_url, 'ImageUrl')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ImagePrediction', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    detect_image_url_with_no_store.metadata = {'url': '/{projectId}/detect/iterations/{publishedName}/url/nostore'}