from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
import dagster._check as check
from dagster._core.definitions.assets_job import build_assets_job
from dagster._core.definitions.definitions_class import Definitions
from dagster._utils.warnings import disable_dagster_warnings
from ..instance import DagsterInstance
from .source_asset import SourceAsset
if TYPE_CHECKING:
    from ..execution.execute_in_process_result import ExecuteInProcessResult

def observe(source_assets: Sequence[SourceAsset], run_config: Any=None, instance: Optional[DagsterInstance]=None, resources: Optional[Mapping[str, object]]=None, partition_key: Optional[str]=None, raise_on_error: bool=True, tags: Optional[Mapping[str, str]]=None) -> 'ExecuteInProcessResult':
    if False:
        for i in range(10):
            print('nop')
    'Executes a single-threaded, in-process run which observes provided source assets.\n\n    By default, will materialize assets to the local filesystem.\n\n    Args:\n        source_assets (Sequence[SourceAsset]):\n            The source assets to materialize.\n        resources (Optional[Mapping[str, object]]):\n            The resources needed for execution. Can provide resource instances\n            directly, or resource definitions. Note that if provided resources\n            conflict with resources directly on assets, an error will be thrown.\n        run_config (Optional[Any]): The run config to use for the run that materializes the assets.\n        partition_key: (Optional[str])\n            The string partition key that specifies the run config to execute. Can only be used\n            to select run config for assets with partitioned config.\n        tags (Optional[Mapping[str, str]]): Tags for the run.\n\n    Returns:\n        ExecuteInProcessResult: The result of the execution.\n    '
    source_assets = check.sequence_param(source_assets, 'assets', of_type=SourceAsset)
    instance = check.opt_inst_param(instance, 'instance', DagsterInstance)
    partition_key = check.opt_str_param(partition_key, 'partition_key')
    resources = check.opt_mapping_param(resources, 'resources', key_type=str)
    with disable_dagster_warnings():
        observation_job = build_assets_job('in_process_observation_job', [], source_assets)
        defs = Definitions(assets=source_assets, jobs=[observation_job], resources=resources)
        return defs.get_job_def('in_process_observation_job').execute_in_process(run_config=run_config, instance=instance, partition_key=partition_key, raise_on_error=raise_on_error, tags=tags)