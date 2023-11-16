from abc import abstractmethod
from typing import Tuple
from airbyte_cdk.sources.abstract_source import AbstractSource
from airbyte_cdk.sources.declarative.checks.connection_checker import ConnectionChecker

class DeclarativeSource(AbstractSource):
    """
    Base class for declarative Source. Concrete sources need to define the connection_checker to use
    """

    @property
    @abstractmethod
    def connection_checker(self) -> ConnectionChecker:
        if False:
            while True:
                i = 10
        'Returns the ConnectionChecker to use for the `check` operation'

    def check_connection(self, logger, config) -> Tuple[bool, any]:
        if False:
            print('Hello World!')
        '\n        :param logger: The source logger\n        :param config: The user-provided configuration as specified by the source\'s spec.\n          This usually contains information required to check connection e.g. tokens, secrets and keys etc.\n        :return: A tuple of (boolean, error). If boolean is true, then the connection check is successful\n          and we can connect to the underlying data source using the provided configuration.\n          Otherwise, the input config cannot be used to connect to the underlying data source,\n          and the "error" object should describe what went wrong.\n          The error object will be cast to string to display the problem to the user.\n        '
        return self.connection_checker.check_connection(self, logger, config)