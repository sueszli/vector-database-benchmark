from msrest.service_client import SDKClient
from msrest import Configuration, Serializer, Deserializer
from .version import VERSION
from msrest.pipeline import ClientRawResponse
from msrest.exceptions import HttpOperationError
from . import models

class TextAnalyticsClientConfiguration(Configuration):
    """Configuration for TextAnalyticsClient
    Note that all parameters used to create this instance are saved as instance
    attributes.

    :param endpoint: Supported Cognitive Services endpoints (protocol and
     hostname, for example: https://westus.api.cognitive.microsoft.com).
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
        base_url = '{Endpoint}/text/analytics/v2.1'
        super(TextAnalyticsClientConfiguration, self).__init__(base_url)
        self.add_user_agent('azure-cognitiveservices-language-textanalytics/{}'.format(VERSION))
        self.endpoint = endpoint
        self.credentials = credentials

class TextAnalyticsClient(SDKClient):
    """The Text Analytics API is a suite of text analytics web services built with best-in-class Microsoft machine learning algorithms. The API can be used to analyze unstructured text for tasks such as sentiment analysis, key phrase extraction and language detection. No training data is needed to use this API; just bring your text data. This API uses advanced natural language processing techniques to deliver best in class predictions. Further documentation can be found in https://docs.microsoft.com/en-us/azure/cognitive-services/text-analytics/overview

    :ivar config: Configuration for client.
    :vartype config: TextAnalyticsClientConfiguration

    :param endpoint: Supported Cognitive Services endpoints (protocol and
     hostname, for example: https://westus.api.cognitive.microsoft.com).
    :type endpoint: str
    :param credentials: Subscription credentials which uniquely identify
     client subscription.
    :type credentials: None
    """

    def __init__(self, endpoint, credentials):
        if False:
            print('Hello World!')
        self.config = TextAnalyticsClientConfiguration(endpoint, credentials)
        super(TextAnalyticsClient, self).__init__(self.config.credentials, self.config)
        client_models = {k: v for (k, v) in models.__dict__.items() if isinstance(v, type)}
        self.api_version = 'v2.1'
        self._serialize = Serializer(client_models)
        self._deserialize = Deserializer(client_models)

    def detect_language(self, show_stats=None, documents=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'The API returns the detected language and a numeric score between 0 and\n        1.\n\n        Scores close to 1 indicate 100% certainty that the identified language\n        is true. A total of 120 languages are supported.\n\n        :param show_stats: (optional) if set to true, response will contain\n         input and document level statistics.\n        :type show_stats: bool\n        :param documents:\n        :type documents:\n         list[~azure.cognitiveservices.language.textanalytics.models.LanguageInput]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: LanguageBatchResult or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.textanalytics.models.LanguageBatchResult\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.textanalytics.models.ErrorResponseException>`\n        '
        language_batch_input = None
        if documents is not None:
            language_batch_input = models.LanguageBatchInput(documents=documents)
        url = self.detect_language.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if show_stats is not None:
            query_parameters['showStats'] = self._serialize.query('show_stats', show_stats, 'bool')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        if language_batch_input is not None:
            body_content = self._serialize.body(language_batch_input, 'LanguageBatchInput')
        else:
            body_content = None
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('LanguageBatchResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    detect_language.metadata = {'url': '/languages'}

    def entities(self, show_stats=None, documents=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'The API returns a list of recognized entities in a given document.\n\n        To get even more information on each recognized entity we recommend\n        using the Bing Entity Search API by querying for the recognized\n        entities names. See the <a\n        href="https://docs.microsoft.com/en-us/azure/cognitive-services/text-analytics/text-analytics-supported-languages">Supported\n        languages in Text Analytics API</a> for the list of enabled languages.\n\n        :param show_stats: (optional) if set to true, response will contain\n         input and document level statistics.\n        :type show_stats: bool\n        :param documents:\n        :type documents:\n         list[~azure.cognitiveservices.language.textanalytics.models.MultiLanguageInput]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: EntitiesBatchResult or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.textanalytics.models.EntitiesBatchResult\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.textanalytics.models.ErrorResponseException>`\n        '
        multi_language_batch_input = None
        if documents is not None:
            multi_language_batch_input = models.MultiLanguageBatchInput(documents=documents)
        url = self.entities.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if show_stats is not None:
            query_parameters['showStats'] = self._serialize.query('show_stats', show_stats, 'bool')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        if multi_language_batch_input is not None:
            body_content = self._serialize.body(multi_language_batch_input, 'MultiLanguageBatchInput')
        else:
            body_content = None
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('EntitiesBatchResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    entities.metadata = {'url': '/entities'}

    def key_phrases(self, show_stats=None, documents=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'The API returns a list of strings denoting the key talking points in\n        the input text.\n\n        See the <a\n        href="https://docs.microsoft.com/en-us/azure/cognitive-services/text-analytics/overview#supported-languages">Text\n        Analytics Documentation</a> for details about the languages that are\n        supported by key phrase extraction.\n\n        :param show_stats: (optional) if set to true, response will contain\n         input and document level statistics.\n        :type show_stats: bool\n        :param documents:\n        :type documents:\n         list[~azure.cognitiveservices.language.textanalytics.models.MultiLanguageInput]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: KeyPhraseBatchResult or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.textanalytics.models.KeyPhraseBatchResult\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.language.textanalytics.models.ErrorResponseException>`\n        '
        multi_language_batch_input = None
        if documents is not None:
            multi_language_batch_input = models.MultiLanguageBatchInput(documents=documents)
        url = self.key_phrases.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if show_stats is not None:
            query_parameters['showStats'] = self._serialize.query('show_stats', show_stats, 'bool')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        if multi_language_batch_input is not None:
            body_content = self._serialize.body(multi_language_batch_input, 'MultiLanguageBatchInput')
        else:
            body_content = None
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('KeyPhraseBatchResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    key_phrases.metadata = {'url': '/keyPhrases'}

    def sentiment(self, show_stats=None, documents=None, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'The API returns a numeric score between 0 and 1.\n\n        Scores close to 1 indicate positive sentiment, while scores close to 0\n        indicate negative sentiment. A score of 0.5 indicates the lack of\n        sentiment (e.g. a factoid statement). See the <a\n        href="https://docs.microsoft.com/en-us/azure/cognitive-services/text-analytics/overview#supported-languages">Text\n        Analytics Documentation</a> for details about the languages that are\n        supported by sentiment analysis.\n\n        :param show_stats: (optional) if set to true, response will contain\n         input and document level statistics.\n        :type show_stats: bool\n        :param documents:\n        :type documents:\n         list[~azure.cognitiveservices.language.textanalytics.models.MultiLanguageInput]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: object or ClientRawResponse if raw=true\n        :rtype: object or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`HttpOperationError<msrest.exceptions.HttpOperationError>`\n        '
        multi_language_batch_input = None
        if documents is not None:
            multi_language_batch_input = models.MultiLanguageBatchInput(documents=documents)
        url = self.sentiment.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if show_stats is not None:
            query_parameters['showStats'] = self._serialize.query('show_stats', show_stats, 'bool')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        if multi_language_batch_input is not None:
            body_content = self._serialize.body(multi_language_batch_input, 'MultiLanguageBatchInput')
        else:
            body_content = None
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200, 500]:
            raise HttpOperationError(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('SentimentBatchResult', response)
        if response.status_code == 500:
            deserialized = self._deserialize('ErrorResponse', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    sentiment.metadata = {'url': '/sentiment'}