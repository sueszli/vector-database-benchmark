from msrest.service_client import SDKClient
from msrest import Configuration, Serializer, Deserializer
from .version import VERSION
from msrest.pipeline import ClientRawResponse
from msrest.exceptions import HttpOperationError
from .operations.events_operations import EventsOperations
from . import models

class PersonalizerClientConfiguration(Configuration):
    """Configuration for PersonalizerClient
    Note that all parameters used to create this instance are saved as instance
    attributes.

    :param endpoint: Supported Cognitive Services endpoint.
    :type endpoint: str
    :param credentials: Subscription credentials which uniquely identify
     client subscription.
    :type credentials: None
    """

    def __init__(self, endpoint, credentials):
        if False:
            i = 10
            return i + 15
        if endpoint is None:
            raise ValueError("Parameter 'endpoint' must not be None.")
        if credentials is None:
            raise ValueError("Parameter 'credentials' must not be None.")
        base_url = '{Endpoint}/personalizer/v1.0'
        super(PersonalizerClientConfiguration, self).__init__(base_url)
        self.add_user_agent('azure-cognitiveservices-personalizer/{}'.format(VERSION))
        self.endpoint = endpoint
        self.credentials = credentials

class PersonalizerClient(SDKClient):
    """Personalizer Service is an Azure Cognitive Service that makes it easy to target content and experiences without complex pre-analysis or cleanup of past data. Given a context and featurized content, the Personalizer Service returns your content in a ranked list. As rewards are sent in response to the ranked list, the reinforcement learning algorithm will improve the model and improve performance of future rank calls.

    :ivar config: Configuration for client.
    :vartype config: PersonalizerClientConfiguration

    :ivar events: Events operations
    :vartype events: azure.cognitiveservices.personalizer.operations.EventsOperations

    :param endpoint: Supported Cognitive Services endpoint.
    :type endpoint: str
    :param credentials: Subscription credentials which uniquely identify
     client subscription.
    :type credentials: None
    """

    def __init__(self, endpoint, credentials):
        if False:
            return 10
        self.config = PersonalizerClientConfiguration(endpoint, credentials)
        super(PersonalizerClient, self).__init__(self.config.credentials, self.config)
        client_models = {k: v for (k, v) in models.__dict__.items() if isinstance(v, type)}
        self.api_version = 'v1.0'
        self._serialize = Serializer(client_models)
        self._deserialize = Deserializer(client_models)
        self.events = EventsOperations(self._client, self.config, self._serialize, self._deserialize)

    def rank(self, rank_request, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'A Personalizer rank request.\n\n        :param rank_request: A Personalizer request.\n        :type rank_request:\n         ~azure.cognitiveservices.personalizer.models.RankRequest\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: RankResponse or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.personalizer.models.RankResponse or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.personalizer.models.ErrorResponseException>`\n        '
        url = self.rank.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(rank_request, 'RankRequest')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [201]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 201:
            deserialized = self._deserialize('RankResponse', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    rank.metadata = {'url': '/rank'}