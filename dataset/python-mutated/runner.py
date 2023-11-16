import logging
from abc import ABC, abstractmethod
from typing import Generic, Iterable, Optional
from airbyte_cdk.connector import TConfig
from airbyte_cdk.models import AirbyteCatalog, AirbyteMessage, AirbyteStateMessage, ConfiguredAirbyteCatalog, ConnectorSpecification
from airbyte_cdk.sources.source import Source

class SourceRunner(ABC, Generic[TConfig]):

    @abstractmethod
    def spec(self) -> ConnectorSpecification:
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def discover(self, config: TConfig) -> AirbyteCatalog:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def read(self, config: TConfig, catalog: ConfiguredAirbyteCatalog, state: Optional[AirbyteStateMessage]) -> Iterable[AirbyteMessage]:
        if False:
            print('Hello World!')
        pass

class CDKRunner(SourceRunner[TConfig]):

    def __init__(self, source: Source, name: str):
        if False:
            return 10
        self._source = source
        self._logger = logging.getLogger(name)

    def spec(self) -> ConnectorSpecification:
        if False:
            print('Hello World!')
        return self._source.spec(self._logger)

    def discover(self, config: TConfig) -> AirbyteCatalog:
        if False:
            for i in range(10):
                print('nop')
        return self._source.discover(self._logger, config)

    def read(self, config: TConfig, catalog: ConfiguredAirbyteCatalog, state: Optional[AirbyteStateMessage]) -> Iterable[AirbyteMessage]:
        if False:
            for i in range(10):
                print('nop')
        return self._source.read(self._logger, config, catalog, state=[state] if state else [])