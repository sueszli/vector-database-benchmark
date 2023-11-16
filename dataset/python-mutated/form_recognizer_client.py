from msrest.service_client import SDKClient
from msrest import Configuration, Serializer, Deserializer
from .version import VERSION
from msrest.pipeline import ClientRawResponse
from . import models

class FormRecognizerClientConfiguration(Configuration):
    """Configuration for FormRecognizerClient
    Note that all parameters used to create this instance are saved as instance
    attributes.

    :param endpoint: Supported Cognitive Services endpoints (protocol and
     hostname, for example: https://westus2.api.cognitive.microsoft.com).
    :type endpoint: str
    :param credentials: Subscription credentials which uniquely identify
     client subscription.
    :type credentials: None
    """

    def __init__(self, endpoint, credentials):
        if False:
            while True:
                i = 10
        if endpoint is None:
            raise ValueError("Parameter 'endpoint' must not be None.")
        if credentials is None:
            raise ValueError("Parameter 'credentials' must not be None.")
        base_url = '{Endpoint}/formrecognizer/v1.0-preview'
        super(FormRecognizerClientConfiguration, self).__init__(base_url)
        self.add_user_agent('azure-cognitiveservices-formrecognizer/{}'.format(VERSION))
        self.endpoint = endpoint
        self.credentials = credentials

class FormRecognizerClient(SDKClient):
    """Extracts information from forms and images into structured data based on a model created by a set of representative training forms.

    :ivar config: Configuration for client.
    :vartype config: FormRecognizerClientConfiguration

    :param endpoint: Supported Cognitive Services endpoints (protocol and
     hostname, for example: https://westus2.api.cognitive.microsoft.com).
    :type endpoint: str
    :param credentials: Subscription credentials which uniquely identify
     client subscription.
    :type credentials: None
    """

    def __init__(self, endpoint, credentials):
        if False:
            while True:
                i = 10
        self.config = FormRecognizerClientConfiguration(endpoint, credentials)
        super(FormRecognizerClient, self).__init__(self.config.credentials, self.config)
        client_models = {k: v for (k, v) in models.__dict__.items() if isinstance(v, type)}
        self.api_version = '1.0-preview'
        self._serialize = Serializer(client_models)
        self._deserialize = Deserializer(client_models)

    def train_custom_model(self, source, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Train Model.\n\n        Create and train a custom model. The train request must include a\n        source parameter that is either an externally accessible Azure Storage\n        blob container Uri (preferably a Shared Access Signature Uri) or valid\n        path to a data folder in a locally mounted drive. When local paths are\n        specified, they must follow the Linux/Unix path format and be an\n        absolute path rooted to the input mount configuration\n        setting value e.g., if \'{Mounts:Input}\' configuration setting value is\n        \'/input\' then a valid source path would be \'/input/contosodataset\'. All\n        data to be trained is expected to be directly under the source folder.\n        Subfolders are not supported. Models are trained using documents that\n        are of the following content type - \'application/pdf\', \'image/jpeg\' and\n        \'image/png\'."\n        Other type of content is ignored.\n\n        :param source: Get or set source path.\n        :type source: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: TrainResult or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.formrecognizer.models.TrainResult or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.formrecognizer.models.ErrorResponseException>`\n        '
        train_request = models.TrainRequest(source=source)
        url = self.train_custom_model.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(train_request, 'TrainRequest')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('TrainResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    train_custom_model.metadata = {'url': '/custom/train'}

    def get_extracted_keys(self, id, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Get Keys.\n\n        Retrieve the keys that were\n        extracted during the training of the specified model.\n\n        :param id: Model identifier.\n        :type id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: KeysResult or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.formrecognizer.models.KeysResult or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.formrecognizer.models.ErrorResponseException>`\n        '
        url = self.get_extracted_keys.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'id': self._serialize.url('id', id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('KeysResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_extracted_keys.metadata = {'url': '/custom/models/{id}/keys'}

    def get_custom_models(self, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Get Models.\n\n        Get information about all trained custom models.\n\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ModelsResult or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.formrecognizer.models.ModelsResult or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.formrecognizer.models.ErrorResponseException>`\n        '
        url = self.get_custom_models.metadata['url']
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
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ModelsResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_custom_models.metadata = {'url': '/custom/models'}

    def get_custom_model(self, id, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Get Model.\n\n        Get information about a model.\n\n        :param id: Model identifier.\n        :type id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ModelResult or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.formrecognizer.models.ModelResult or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.formrecognizer.models.ErrorResponseException>`\n        '
        url = self.get_custom_model.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'id': self._serialize.url('id', id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ModelResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_custom_model.metadata = {'url': '/custom/models/{id}'}

    def delete_custom_model(self, id, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Delete Model.\n\n        Delete model artifacts.\n\n        :param id: The identifier of the model to delete.\n        :type id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.formrecognizer.models.ErrorResponseException>`\n        '
        url = self.delete_custom_model.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'id': self._serialize.url('id', id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.delete(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [204]:
            raise models.ErrorResponseException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    delete_custom_model.metadata = {'url': '/custom/models/{id}'}

    def analyze_with_custom_model(self, id, form_stream, keys=None, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        "Analyze Form.\n\n        Extract key-value pairs from a given document. The input document must\n        be of one of the supported content types - 'application/pdf',\n        'image/jpeg' or 'image/png'. A success response is returned in JSON.\n\n        :param id: Model Identifier to analyze the document with.\n        :type id: str\n        :param form_stream: A pdf document or image (jpg,png) file to analyze.\n        :type form_stream: Generator\n        :param keys: An optional list of known keys to extract the values for.\n        :type keys: list[str]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: AnalyzeResult or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.formrecognizer.models.AnalyzeResult\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.formrecognizer.models.ErrorResponseException>`\n        "
        url = self.analyze_with_custom_model.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'id': self._serialize.url('id', id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if keys is not None:
            query_parameters['keys'] = self._serialize.query('keys', keys, '[str]', div=',')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'multipart/form-data'
        if custom_headers:
            header_parameters.update(custom_headers)
        form_data_content = {'form_stream': form_stream}
        request = self._client.post(url, query_parameters, header_parameters, form_content=form_data_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('AnalyzeResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    analyze_with_custom_model.metadata = {'url': '/custom/models/{id}/analyze'}