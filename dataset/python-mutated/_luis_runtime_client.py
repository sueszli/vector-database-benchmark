from msrest.service_client import SDKClient
from msrest import Serializer, Deserializer
from ._configuration import LUISRuntimeClientConfiguration
from .operations import PredictionOperations
from . import models

class LUISRuntimeClient(SDKClient):
    """LUISRuntimeClient

    :ivar config: Configuration for client.
    :vartype config: LUISRuntimeClientConfiguration

    :ivar prediction: Prediction operations
    :vartype prediction: azure.cognitiveservices.language.luis.runtime.operations.PredictionOperations

    :param endpoint: Supported Cognitive Services endpoints (protocol and
     hostname, for example: https://westus.api.cognitive.microsoft.com).
    :type endpoint: str
    :param credentials: Subscription credentials which uniquely identify
     client subscription.
    :type credentials: None
    """

    def __init__(self, endpoint, credentials):
        if False:
            for i in range(10):
                print('nop')
        self.config = LUISRuntimeClientConfiguration(endpoint, credentials)
        super(LUISRuntimeClient, self).__init__(self.config.credentials, self.config)
        client_models = {k: v for (k, v) in models.__dict__.items() if isinstance(v, type)}
        self.api_version = '3.0'
        self._serialize = Serializer(client_models)
        self._deserialize = Deserializer(client_models)
        self.prediction = PredictionOperations(self._client, self.config, self._serialize, self._deserialize)