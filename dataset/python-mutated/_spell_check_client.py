from msrest.service_client import SDKClient
from msrest import Serializer, Deserializer
from ._configuration import SpellCheckClientConfiguration
from .operations import SpellCheckClientOperationsMixin
from . import models

class SpellCheckClient(SpellCheckClientOperationsMixin, SDKClient):
    """The Spell Check API - V7 lets you check a text string for spelling and grammar errors.

    :ivar config: Configuration for client.
    :vartype config: SpellCheckClientConfiguration

    :param endpoint: Supported Cognitive Services endpoints (protocol and
     hostname, for example: "https://westus.api.cognitive.microsoft.com",
     "https://api.cognitive.microsoft.com").
    :type endpoint: str
    :param credentials: Subscription credentials which uniquely identify
     client subscription.
    :type credentials: None
    """

    def __init__(self, endpoint, credentials):
        if False:
            return 10
        self.config = SpellCheckClientConfiguration(endpoint, credentials)
        super(SpellCheckClient, self).__init__(self.config.credentials, self.config)
        client_models = {k: v for (k, v) in models.__dict__.items() if isinstance(v, type)}
        self.api_version = '1.0'
        self._serialize = Serializer(client_models)
        self._deserialize = Deserializer(client_models)