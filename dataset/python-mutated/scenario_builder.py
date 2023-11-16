from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Generic, List, Mapping, Optional, Set, Tuple, Type, TypeVar
from airbyte_cdk.models import AirbyteAnalyticsTraceMessage, SyncMode
from airbyte_cdk.sources import AbstractSource

@dataclass
class IncrementalScenarioConfig:
    input_state: List[Mapping[str, Any]] = field(default_factory=list)
    expected_output_state: Optional[Mapping[str, Any]] = None
SourceType = TypeVar('SourceType', bound=AbstractSource)

class SourceBuilder(ABC, Generic[SourceType]):
    """
    A builder that creates a source instance of type SourceType
    """

    @abstractmethod
    def build(self, configured_catalog: Optional[Mapping[str, Any]]) -> SourceType:
        if False:
            while True:
                i = 10
        raise NotImplementedError()

class TestScenario(Generic[SourceType]):

    def __init__(self, name: str, config: Mapping[str, Any], source: SourceType, expected_spec: Optional[Mapping[str, Any]], expected_check_status: Optional[str], expected_catalog: Optional[Mapping[str, Any]], expected_logs: Optional[Mapping[str, List[Mapping[str, Any]]]], expected_records: List[Mapping[str, Any]], expected_check_error: Tuple[Optional[Type[Exception]], Optional[str]], expected_discover_error: Tuple[Optional[Type[Exception]], Optional[str]], expected_read_error: Tuple[Optional[Type[Exception]], Optional[str]], incremental_scenario_config: Optional[IncrementalScenarioConfig], expected_analytics: Optional[List[AirbyteAnalyticsTraceMessage]]=None, log_levels: Optional[Set[str]]=None):
        if False:
            print('Hello World!')
        if log_levels is None:
            log_levels = {'ERROR', 'WARN', 'WARNING'}
        self.name = name
        self.config = config
        self.source = source
        self.expected_spec = expected_spec
        self.expected_check_status = expected_check_status
        self.expected_catalog = expected_catalog
        self.expected_logs = expected_logs
        self.expected_records = expected_records
        self.expected_check_error = expected_check_error
        self.expected_discover_error = expected_discover_error
        self.expected_read_error = expected_read_error
        self.incremental_scenario_config = incremental_scenario_config
        self.expected_analytics = expected_analytics
        self.log_levels = log_levels
        self.validate()

    def validate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        assert self.name
        if not self.expected_catalog:
            return
        if self.expected_read_error or self.expected_check_error:
            return
        streams = set([s.name for s in self.source.streams(self.config)])
        expected_streams = {s['name'] for s in self.expected_catalog['streams']}
        assert expected_streams <= streams

    def configured_catalog(self, sync_mode: SyncMode) -> Optional[Mapping[str, Any]]:
        if False:
            i = 10
            return i + 15
        catalog: Mapping[str, Any] = {'streams': []}
        for stream in self.source.streams(self.config):
            catalog['streams'].append({'stream': stream.name, 'sync_mode': sync_mode.value, 'destination_sync_mode': 'append'})
        return catalog

    def input_state(self) -> List[Mapping[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        if self.incremental_scenario_config:
            return self.incremental_scenario_config.input_state
        else:
            return []

class TestScenarioBuilder(Generic[SourceType]):
    """
    A builder that creates a TestScenario instance for a source of type SourceType
    """

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self._name = ''
        self._config: Mapping[str, Any] = {}
        self._expected_spec: Optional[Mapping[str, Any]] = None
        self._expected_check_status: Optional[str] = None
        self._expected_catalog: Mapping[str, Any] = {}
        self._expected_logs: Optional[Mapping[str, Any]] = None
        self._expected_records: List[Mapping[str, Any]] = []
        self._expected_check_error: Tuple[Optional[Type[Exception]], Optional[str]] = (None, None)
        self._expected_discover_error: Tuple[Optional[Type[Exception]], Optional[str]] = (None, None)
        self._expected_read_error: Tuple[Optional[Type[Exception]], Optional[str]] = (None, None)
        self._incremental_scenario_config: Optional[IncrementalScenarioConfig] = None
        self._expected_analytics: Optional[List[AirbyteAnalyticsTraceMessage]] = None
        self.source_builder: Optional[SourceBuilder[SourceType]] = None
        self._log_levels = None

    def set_name(self, name: str) -> 'TestScenarioBuilder[SourceType]':
        if False:
            while True:
                i = 10
        self._name = name
        return self

    def set_config(self, config: Mapping[str, Any]) -> 'TestScenarioBuilder[SourceType]':
        if False:
            print('Hello World!')
        self._config = config
        return self

    def set_expected_spec(self, expected_spec: Mapping[str, Any]) -> 'TestScenarioBuilder[SourceType]':
        if False:
            return 10
        self._expected_spec = expected_spec
        return self

    def set_expected_check_status(self, expected_check_status: str) -> 'TestScenarioBuilder[SourceType]':
        if False:
            while True:
                i = 10
        self._expected_check_status = expected_check_status
        return self

    def set_expected_catalog(self, expected_catalog: Mapping[str, Any]) -> 'TestScenarioBuilder[SourceType]':
        if False:
            return 10
        self._expected_catalog = expected_catalog
        return self

    def set_expected_logs(self, expected_logs: Mapping[str, List[Mapping[str, Any]]]) -> 'TestScenarioBuilder[SourceType]':
        if False:
            print('Hello World!')
        self._expected_logs = expected_logs
        return self

    def set_expected_records(self, expected_records: List[Mapping[str, Any]]) -> 'TestScenarioBuilder[SourceType]':
        if False:
            print('Hello World!')
        self._expected_records = expected_records
        return self

    def set_incremental_scenario_config(self, incremental_scenario_config: IncrementalScenarioConfig) -> 'TestScenarioBuilder[SourceType]':
        if False:
            for i in range(10):
                print('nop')
        self._incremental_scenario_config = incremental_scenario_config
        return self

    def set_expected_check_error(self, error: Optional[Type[Exception]], message: str) -> 'TestScenarioBuilder[SourceType]':
        if False:
            return 10
        self._expected_check_error = (error, message)
        return self

    def set_expected_discover_error(self, error: Type[Exception], message: str) -> 'TestScenarioBuilder[SourceType]':
        if False:
            for i in range(10):
                print('nop')
        self._expected_discover_error = (error, message)
        return self

    def set_expected_read_error(self, error: Type[Exception], message: str) -> 'TestScenarioBuilder[SourceType]':
        if False:
            print('Hello World!')
        self._expected_read_error = (error, message)
        return self

    def set_log_levels(self, levels: Set[str]) -> 'TestScenarioBuilder':
        if False:
            while True:
                i = 10
        self._log_levels = levels
        return self

    def set_source_builder(self, source_builder: SourceBuilder[SourceType]) -> 'TestScenarioBuilder[SourceType]':
        if False:
            return 10
        self.source_builder = source_builder
        return self

    def set_expected_analytics(self, expected_analytics: Optional[List[AirbyteAnalyticsTraceMessage]]) -> 'TestScenarioBuilder[SourceType]':
        if False:
            for i in range(10):
                print('nop')
        self._expected_analytics = expected_analytics
        return self

    def copy(self) -> 'TestScenarioBuilder[SourceType]':
        if False:
            return 10
        return deepcopy(self)

    def build(self) -> 'TestScenario[SourceType]':
        if False:
            print('Hello World!')
        if self.source_builder is None:
            raise ValueError('source_builder is not set')
        source = self.source_builder.build(self._configured_catalog(SyncMode.incremental if self._incremental_scenario_config else SyncMode.full_refresh))
        return TestScenario(self._name, self._config, source, self._expected_spec, self._expected_check_status, self._expected_catalog, self._expected_logs, self._expected_records, self._expected_check_error, self._expected_discover_error, self._expected_read_error, self._incremental_scenario_config, self._expected_analytics, self._log_levels)

    def _configured_catalog(self, sync_mode: SyncMode) -> Optional[Mapping[str, Any]]:
        if False:
            return 10
        if not self._expected_catalog:
            return None
        catalog: Mapping[str, Any] = {'streams': []}
        for stream in self._expected_catalog['streams']:
            catalog['streams'].append({'stream': stream, 'sync_mode': sync_mode.value, 'destination_sync_mode': 'append'})
        return catalog