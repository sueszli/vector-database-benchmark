from typing import Any, Mapping, Optional
import pytest
from dagster._core.launcher.default_run_launcher import DefaultRunLauncher
from dagster._core.run_coordinator.default_run_coordinator import DefaultRunCoordinator
from dagster._core.storage.noop_compute_log_manager import NoOpComputeLogManager
from dagster._core.test_utils import instance_for_test
from dagster._serdes import ConfigurableClass
from dagster._serdes.config_class import ConfigurableClassData
from typing_extensions import Self

class InitFailRunLauncher(DefaultRunLauncher, ConfigurableClass):

    def __init__(self, inst_data: Optional[ConfigurableClassData]=None):
        if False:
            while True:
                i = 10
        super().__init__()
        self._inst_data = inst_data
        raise Exception('Expected init fail')

    @property
    def inst_data(self):
        if False:
            return 10
        return self._inst_data

    @classmethod
    def config_type(cls):
        if False:
            i = 10
            return i + 15
        return {}

    @classmethod
    def from_config_value(cls, inst_data: ConfigurableClassData, config_value: Mapping[str, Any]) -> Self:
        if False:
            i = 10
            return i + 15
        return InitFailRunLauncher(inst_data=inst_data)

def test_lazy_run_launcher():
    if False:
        return 10
    with instance_for_test(overrides={'run_launcher': {'module': 'dagster_tests.core_tests.instance_tests.test_instance_lazy_load', 'class': 'InitFailRunLauncher', 'config': {}}}) as instance:
        with pytest.raises(Exception, match='Expected init fail'):
            print(instance.run_launcher)

class InitFailComputeLogManager(NoOpComputeLogManager, ConfigurableClass):

    def __init__(self, inst_data: Optional[ConfigurableClassData]=None):
        if False:
            while True:
                i = 10
        super().__init__(inst_data)
        raise Exception('Expected init fail')

    @classmethod
    def from_config_value(cls, inst_data: ConfigurableClassData, config_value: Mapping[str, Any]) -> Self:
        if False:
            return 10
        return InitFailComputeLogManager(inst_data=inst_data)

def test_lazy_compute_log_manager():
    if False:
        while True:
            i = 10
    with instance_for_test(overrides={'compute_logs': {'module': 'dagster_tests.core_tests.instance_tests.test_instance_lazy_load', 'class': 'InitFailComputeLogManager', 'config': {}}}) as instance:
        with pytest.raises(Exception, match='Expected init fail'):
            print(instance.compute_log_manager)

class InitFailRunCoordinator(DefaultRunCoordinator, ConfigurableClass):

    def __init__(self, inst_data: Optional[ConfigurableClassData]=None):
        if False:
            i = 10
            return i + 15
        super().__init__(inst_data)
        raise Exception('Expected init fail')

    @classmethod
    def from_config_value(cls, inst_data: ConfigurableClassData, config_value: Mapping[str, Any]) -> Self:
        if False:
            i = 10
            return i + 15
        return InitFailRunCoordinator(inst_data=inst_data)

def test_lazy_run_coordinator():
    if False:
        return 10
    with instance_for_test(overrides={'run_coordinator': {'module': 'dagster_tests.core_tests.instance_tests.test_instance_lazy_load', 'class': 'InitFailRunCoordinator', 'config': {}}}) as instance:
        with pytest.raises(Exception, match='Expected init fail'):
            print(instance.run_coordinator)