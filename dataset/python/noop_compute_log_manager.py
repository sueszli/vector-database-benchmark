from contextlib import contextmanager
from typing import IO, Any, Generator, Mapping, Optional, Sequence

from typing_extensions import Self

import dagster._check as check
from dagster._core.storage.captured_log_manager import (
    CapturedLogContext,
    CapturedLogData,
    CapturedLogManager,
    CapturedLogMetadata,
    CapturedLogSubscription,
)
from dagster._serdes import ConfigurableClass, ConfigurableClassData

from .compute_log_manager import (
    MAX_BYTES_FILE_READ,
    ComputeIOType,
    ComputeLogFileData,
    ComputeLogManager,
)


class NoOpComputeLogManager(CapturedLogManager, ComputeLogManager, ConfigurableClass):
    """When enabled for a Dagster instance, stdout and stderr will not be available for any step."""

    def __init__(self, inst_data: Optional[ConfigurableClassData] = None):
        self._inst_data = check.opt_inst_param(inst_data, "inst_data", ConfigurableClassData)

    @property
    def inst_data(self):
        return self._inst_data

    @classmethod
    def config_type(cls):
        return {}

    @classmethod
    def from_config_value(
        cls, inst_data: ConfigurableClassData, config_value: Mapping[str, Any]
    ) -> Self:
        return NoOpComputeLogManager(inst_data=inst_data, **config_value)

    def enabled(self, _dagster_run, _step_key):
        return False

    def _watch_logs(self, dagster_run, step_key=None):
        pass

    def get_local_path(self, run_id: str, key: str, io_type: ComputeIOType) -> str:
        raise NotImplementedError()

    def is_watch_completed(self, run_id, key):
        return True

    def on_watch_start(self, dagster_run, step_key):
        pass

    def on_watch_finish(self, dagster_run, step_key):
        pass

    def download_url(self, run_id, key, io_type):
        return None

    def read_logs_file(self, run_id, key, io_type, cursor=0, max_bytes=MAX_BYTES_FILE_READ):
        return ComputeLogFileData(
            path=f"{key}.{io_type}", data=None, cursor=0, size=0, download_url=None
        )

    def on_subscribe(self, subscription):
        pass

    def on_unsubscribe(self, subscription):
        pass

    @contextmanager
    def capture_logs(self, log_key: Sequence[str]) -> Generator[CapturedLogContext, None, None]:
        yield CapturedLogContext(log_key=log_key)

    def is_capture_complete(self, log_key: Sequence[str]):
        return True

    @contextmanager
    def open_log_stream(
        self, log_key: Sequence[str], io_type: ComputeIOType
    ) -> Generator[Optional[IO], None, None]:
        yield None

    def get_log_data(
        self,
        log_key: Sequence[str],
        cursor: Optional[str] = None,
        max_bytes: Optional[int] = None,
    ) -> CapturedLogData:
        return CapturedLogData(log_key=log_key)

    def get_log_metadata(self, log_key: Sequence[str]) -> CapturedLogMetadata:
        return CapturedLogMetadata()

    def delete_logs(
        self, log_key: Optional[Sequence[str]] = None, prefix: Optional[Sequence[str]] = None
    ):
        pass

    def subscribe(
        self, log_key: Sequence[str], cursor: Optional[str] = None
    ) -> CapturedLogSubscription:
        return CapturedLogSubscription(self, log_key, cursor)

    def unsubscribe(self, subscription: CapturedLogSubscription):
        pass
