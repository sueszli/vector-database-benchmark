import logging
from abc import ABC, abstractmethod
from typing import Any, Mapping, Tuple
from airbyte_cdk.sources.source import Source

class ConnectionChecker(ABC):
    """
    Abstract base class for checking a connection
    """

    @abstractmethod
    def check_connection(self, source: Source, logger: logging.Logger, config: Mapping[str, Any]) -> Tuple[bool, any]:
        if False:
            while True:
                i = 10
        '\n        Tests if the input configuration can be used to successfully connect to the integration e.g: if a provided Stripe API token can be used to connect\n        to the Stripe API.\n\n        :param source: source\n        :param logger: source logger\n        :param config: The user-provided configuration as specified by the source\'s spec.\n          This usually contains information required to check connection e.g. tokens, secrets and keys etc.\n        :return: A tuple of (boolean, error). If boolean is true, then the connection check is successful\n          and we can connect to the underlying data source using the provided configuration.\n          Otherwise, the input config cannot be used to connect to the underlying data source,\n          and the "error" object should describe what went wrong.\n          The error object will be cast to string to display the problem to the user.\n        '
        pass