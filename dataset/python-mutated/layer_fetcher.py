from abc import abstractmethod
from localstack.services.lambda_.invocation.lambda_models import Layer

class LayerFetcher:

    @abstractmethod
    def fetch_layer(self, layer_version_arn: str) -> Layer | None:
        if False:
            print('Hello World!')
        'Fetches a shared Lambda layer for a given layer_version_arn\n\n        :param layer_version_arn: The layer arn including its version to be fetched. Example:\n               "arn:aws:lambda:us-east-1:770693421928:layer:Klayers-p39-PyYAML:1"\n        :return: A Lambda layer model if layer could be fetched, None otherwise (e.g., not available or accessible)\n        '
        pass