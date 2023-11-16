from typing import Any, Mapping, Optional
from typing_extensions import Self
import dagster._check as check
from dagster._config.config_schema import UserConfigSchema
from dagster._core.execution.api import execute_run
from dagster._core.launcher import LaunchRunContext, RunLauncher
from dagster._serdes import ConfigurableClass
from dagster._serdes.config_class import ConfigurableClassData
from dagster._utils.hosted_user_process import recon_job_from_origin

class SyncInMemoryRunLauncher(RunLauncher, ConfigurableClass):
    """This run launcher launches runs synchronously, in memory, and is intended only for test.

    Use the :py:class:`dagster.DefaultRunLauncher`.
    """

    def __init__(self, inst_data: Optional[ConfigurableClassData]=None):
        if False:
            return 10
        self._inst_data = inst_data
        self._repository = None
        self._instance_ref = None
        super().__init__()

    @property
    def inst_data(self) -> Optional[ConfigurableClassData]:
        if False:
            print('Hello World!')
        return self._inst_data

    @classmethod
    def config_type(cls) -> UserConfigSchema:
        if False:
            for i in range(10):
                print('nop')
        return {}

    @classmethod
    def from_config_value(cls, inst_data: ConfigurableClassData, config_value: Mapping[str, Any]) -> Self:
        if False:
            while True:
                i = 10
        return SyncInMemoryRunLauncher(inst_data=inst_data)

    def launch_run(self, context: LaunchRunContext) -> None:
        if False:
            i = 10
            return i + 15
        recon_job = recon_job_from_origin(context.job_code_origin)
        execute_run(recon_job, context.dagster_run, self._instance)

    def terminate(self, run_id):
        if False:
            for i in range(10):
                print('nop')
        check.not_implemented('Termination not supported.')