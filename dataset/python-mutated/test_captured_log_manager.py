import sys
import tempfile
from contextlib import contextmanager
from typing import Any, Generator, Mapping, Sequence
import pytest
from dagster import job, op
from dagster._core.events import DagsterEventType
from dagster._core.storage.captured_log_manager import CapturedLogContext
from dagster._core.storage.local_compute_log_manager import LocalComputeLogManager
from dagster._core.storage.noop_compute_log_manager import NoOpComputeLogManager
from dagster._core.test_utils import instance_for_test
from dagster._serdes import ConfigurableClassData
from typing_extensions import Self
from .utils.captured_log_manager import TestCapturedLogManager

def test_compute_log_manager_instance():
    if False:
        for i in range(10):
            print('nop')
    with instance_for_test() as instance:
        assert instance.compute_log_manager
        assert instance.compute_log_manager._instance

class TestLocalCapturedLogManager(TestCapturedLogManager):
    __test__ = True

    @pytest.fixture(name='captured_log_manager')
    def captured_log_manager(self):
        if False:
            i = 10
            return i + 15
        with tempfile.TemporaryDirectory() as tmpdir_path:
            return LocalComputeLogManager(tmpdir_path)

class ExternalTestComputeLogManager(NoOpComputeLogManager):
    """Test compute log manager that does not actually capture logs, but generates an external url
    to be shown within the Dagster UI.
    """

    @classmethod
    def from_config_value(cls, inst_data: ConfigurableClassData, config_value: Mapping[str, Any]) -> Self:
        if False:
            for i in range(10):
                print('nop')
        return ExternalTestComputeLogManager(inst_data=inst_data, **config_value)

    def enabled(self, _dagster_run, _step_key):
        if False:
            while True:
                i = 10
        return True

    @contextmanager
    def capture_logs(self, log_key: Sequence[str]) -> Generator[CapturedLogContext, None, None]:
        if False:
            return 10
        yield CapturedLogContext(log_key=log_key, external_stdout_url='https://fake.com/stdout', external_stderr_url='https://fake.com/stderr')

def test_external_captured_log_manager():
    if False:
        while True:
            i = 10

    @op
    def my_op():
        if False:
            print('Hello World!')
        print('hello out')
        print('hello error', file=sys.stderr)

    @job
    def my_job():
        if False:
            print('Hello World!')
        my_op()
    with instance_for_test(overrides={'compute_logs': {'module': 'dagster_tests.storage_tests.test_captured_log_manager', 'class': 'ExternalTestComputeLogManager'}}) as instance:
        result = my_job.execute_in_process(instance=instance)
        assert result.success
        assert result.run_id
        captured_log_entries = instance.all_logs(result.run_id, of_type=DagsterEventType.LOGS_CAPTURED)
        assert len(captured_log_entries) == 1
        entry = captured_log_entries[0]
        assert entry.dagster_event.logs_captured_data.external_stdout_url == 'https://fake.com/stdout'
        assert entry.dagster_event.logs_captured_data.external_stderr_url == 'https://fake.com/stderr'