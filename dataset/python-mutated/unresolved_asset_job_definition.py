from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, AbstractSet, Any, Mapping, NamedTuple, Optional, Sequence, Union
import dagster._check as check
from dagster._annotations import deprecated
from dagster._core.definitions import AssetKey
from dagster._core.definitions.run_request import RunRequest
from dagster._core.errors import DagsterInvalidDefinitionError
from dagster._core.instance import DynamicPartitionsStore
from .asset_layer import build_asset_selection_job
from .config import ConfigMapping
from .metadata import RawMetadataValue
if TYPE_CHECKING:
    from dagster._core.definitions import AssetSelection, ExecutorDefinition, HookDefinition, JobDefinition, PartitionedConfig, PartitionsDefinition, ResourceDefinition
    from dagster._core.definitions.asset_graph import InternalAssetGraph
    from dagster._core.definitions.asset_selection import CoercibleToAssetSelection
    from dagster._core.definitions.run_config import RunConfig

class UnresolvedAssetJobDefinition(NamedTuple('_UnresolvedAssetJobDefinition', [('name', str), ('selection', 'AssetSelection'), ('config', Optional[Union[ConfigMapping, Mapping[str, Any], 'PartitionedConfig']]), ('description', Optional[str]), ('tags', Optional[Mapping[str, Any]]), ('metadata', Optional[Mapping[str, RawMetadataValue]]), ('partitions_def', Optional['PartitionsDefinition']), ('executor_def', Optional['ExecutorDefinition']), ('hooks', Optional[AbstractSet['HookDefinition']])])):

    def __new__(cls, name: str, selection: 'AssetSelection', config: Optional[Union[ConfigMapping, Mapping[str, Any], 'PartitionedConfig', 'RunConfig']]=None, description: Optional[str]=None, tags: Optional[Mapping[str, Any]]=None, metadata: Optional[Mapping[str, RawMetadataValue]]=None, partitions_def: Optional['PartitionsDefinition']=None, executor_def: Optional['ExecutorDefinition']=None, hooks: Optional[AbstractSet['HookDefinition']]=None):
        if False:
            return 10
        from dagster._core.definitions import AssetSelection, ExecutorDefinition, HookDefinition, PartitionsDefinition
        from dagster._core.definitions.run_config import convert_config_input
        return super(UnresolvedAssetJobDefinition, cls).__new__(cls, name=check.str_param(name, 'name'), selection=check.inst_param(selection, 'selection', AssetSelection), config=convert_config_input(config), description=check.opt_str_param(description, 'description'), tags=check.opt_mapping_param(tags, 'tags'), metadata=check.opt_mapping_param(metadata, 'metadata'), partitions_def=check.opt_inst_param(partitions_def, 'partitions_def', PartitionsDefinition), executor_def=check.opt_inst_param(executor_def, 'partitions_def', ExecutorDefinition), hooks=check.opt_nullable_set_param(hooks, 'hooks', of_type=HookDefinition))

    @deprecated(breaking_version='2.0.0', additional_warn_text='Directly instantiate `RunRequest(partition_key=...)` instead.')
    def run_request_for_partition(self, partition_key: str, run_key: Optional[str]=None, tags: Optional[Mapping[str, str]]=None, asset_selection: Optional[Sequence[AssetKey]]=None, run_config: Optional[Mapping[str, Any]]=None, current_time: Optional[datetime]=None, dynamic_partitions_store: Optional[DynamicPartitionsStore]=None) -> RunRequest:
        if False:
            for i in range(10):
                print('nop')
        'Creates a RunRequest object for a run that processes the given partition.\n\n        Args:\n            partition_key: The key of the partition to request a run for.\n            run_key (Optional[str]): A string key to identify this launched run. For sensors, ensures that\n                only one run is created per run key across all sensor evaluations.  For schedules,\n                ensures that one run is created per tick, across failure recoveries. Passing in a `None`\n                value means that a run will always be launched per evaluation.\n            tags (Optional[Dict[str, str]]): A dictionary of tags (string key-value pairs) to attach\n                to the launched run.\n            run_config (Optional[Mapping[str, Any]]: Configuration for the run. If the job has\n                a :py:class:`PartitionedConfig`, this value will override replace the config\n                provided by it.\n            current_time (Optional[datetime]): Used to determine which time-partitions exist.\n                Defaults to now.\n            dynamic_partitions_store (Optional[DynamicPartitionsStore]): The DynamicPartitionsStore\n                object that is responsible for fetching dynamic partitions. Required when the\n                partitions definition is a DynamicPartitionsDefinition with a name defined. Users\n                can pass the DagsterInstance fetched via `context.instance` to this argument.\n\n        Returns:\n            RunRequest: an object that requests a run to process the given partition.\n        '
        from dagster._core.definitions.partition import DynamicPartitionsDefinition, PartitionedConfig
        if not self.partitions_def:
            check.failed('Called run_request_for_partition on a non-partitioned job')
        partitioned_config = PartitionedConfig.from_flexible_config(self.config, self.partitions_def)
        if isinstance(self.partitions_def, DynamicPartitionsDefinition) and self.partitions_def.name:
            check.failed('run_request_for_partition is not supported for dynamic partitions. Instead, use RunRequest(partition_key=...)')
        self.partitions_def.validate_partition_key(partition_key, current_time=current_time, dynamic_partitions_store=dynamic_partitions_store)
        run_config = run_config if run_config is not None else partitioned_config.get_run_config_for_partition_key(partition_key)
        run_request_tags = {**(tags or {}), **partitioned_config.get_tags_for_partition_key(partition_key)}
        return RunRequest(job_name=self.name, run_key=run_key, run_config=run_config, tags=run_request_tags, asset_selection=asset_selection, partition_key=partition_key)

    def resolve(self, asset_graph: 'InternalAssetGraph', default_executor_def: Optional['ExecutorDefinition']=None, resource_defs: Optional[Mapping[str, 'ResourceDefinition']]=None) -> 'JobDefinition':
        if False:
            i = 10
            return i + 15
        'Resolve this UnresolvedAssetJobDefinition into a JobDefinition.'
        assets = asset_graph.assets
        source_assets = asset_graph.source_assets
        selected_asset_keys = self.selection.resolve(asset_graph)
        selected_asset_checks = self.selection.resolve_checks(asset_graph)
        asset_keys_by_partitions_def = defaultdict(set)
        for asset_key in selected_asset_keys:
            partitions_def = asset_graph.get_partitions_def(asset_key)
            if partitions_def is not None:
                asset_keys_by_partitions_def[partitions_def].add(asset_key)
        if len(asset_keys_by_partitions_def) > 1:
            keys_by_partitions_def_str = '\n'.join((f'{partitions_def}: {asset_keys}' for (partitions_def, asset_keys) in asset_keys_by_partitions_def.items()))
            raise DagsterInvalidDefinitionError(f"Multiple partitioned assets exist in assets job '{self.name}'. Selected assets must have the same partitions definitions, but the selected assets have different partitions definitions: \n{keys_by_partitions_def_str}")
        inferred_partitions_def = next(iter(asset_keys_by_partitions_def.keys())) if asset_keys_by_partitions_def else None
        if inferred_partitions_def and self.partitions_def != inferred_partitions_def and (self.partitions_def is not None):
            raise DagsterInvalidDefinitionError(f"Job '{self.name}' received a partitions_def of {self.partitions_def}, but the selected assets {next(iter(asset_keys_by_partitions_def.values()))} have a non-matching partitions_def of {inferred_partitions_def}")
        return build_asset_selection_job(name=self.name, assets=assets, asset_checks=asset_graph.asset_checks, config=self.config, source_assets=source_assets, description=self.description, tags=self.tags, metadata=self.metadata, asset_selection=selected_asset_keys, asset_check_selection=selected_asset_checks, partitions_def=self.partitions_def if self.partitions_def else inferred_partitions_def, executor_def=self.executor_def or default_executor_def, hooks=self.hooks, resource_defs=resource_defs)

def define_asset_job(name: str, selection: Optional['CoercibleToAssetSelection']=None, config: Optional[Union[ConfigMapping, Mapping[str, Any], 'PartitionedConfig', 'RunConfig']]=None, description: Optional[str]=None, tags: Optional[Mapping[str, Any]]=None, metadata: Optional[Mapping[str, RawMetadataValue]]=None, partitions_def: Optional['PartitionsDefinition']=None, executor_def: Optional['ExecutorDefinition']=None, hooks: Optional[AbstractSet['HookDefinition']]=None) -> UnresolvedAssetJobDefinition:
    if False:
        for i in range(10):
            print('nop')
    'Creates a definition of a job which will either materialize a selection of assets or observe\n    a selection of source assets. This will only be resolved to a JobDefinition once placed in a\n    code location.\n\n    Args:\n        name (str):\n            The name for the job.\n        selection (Union[str, Sequence[str], Sequence[AssetKey], Sequence[Union[AssetsDefinition, SourceAsset]], AssetSelection]):\n            The assets that will be materialized or observed when the job is run.\n\n            The selected assets must all be included in the assets that are passed to the assets\n            argument of the Definitions object that this job is included on.\n\n            The string "my_asset*" selects my_asset and all downstream assets within the code\n            location. A list of strings represents the union of all assets selected by strings\n            within the list.\n\n            The selection will be resolved to a set of assets when the location is loaded. If the\n            selection resolves to all source assets, the created job will perform source asset\n            observations. If the selection resolves to all regular assets, the created job will\n            materialize assets. If the selection resolves to a mixed set of source assets and\n            regular assets, an error will be thrown.\n\n        config:\n            Describes how the Job is parameterized at runtime.\n\n            If no value is provided, then the schema for the job\'s run config is a standard\n            format based on its ops and resources.\n\n            If a dictionary is provided, then it must conform to the standard config schema, and\n            it will be used as the job\'s run config for the job whenever the job is executed.\n            The values provided will be viewable and editable in the Dagster UI, so be\n            careful with secrets.\n\n            If a :py:class:`ConfigMapping` object is provided, then the schema for the job\'s run config is\n            determined by the config mapping, and the ConfigMapping, which should return\n            configuration in the standard format to configure the job.\n        tags (Optional[Mapping[str, Any]]):\n            Arbitrary information that will be attached to the execution of the Job.\n            Values that are not strings will be json encoded and must meet the criteria that\n            `json.loads(json.dumps(value)) == value`.  These tag values may be overwritten by tag\n            values provided at invocation time.\n        metadata (Optional[Mapping[str, RawMetadataValue]]): Arbitrary metadata about the job.\n            Keys are displayed string labels, and values are one of the following: string, float,\n            int, JSON-serializable dict, JSON-serializable list, and one of the data classes\n            returned by a MetadataValue static method.\n        description (Optional[str]):\n            A description for the Job.\n        partitions_def (Optional[PartitionsDefinition]):\n            Defines the set of partitions for this job. All AssetDefinitions selected for this job\n            must have a matching PartitionsDefinition. If no PartitionsDefinition is provided, the\n            PartitionsDefinition will be inferred from the selected AssetDefinitions.\n        executor_def (Optional[ExecutorDefinition]):\n            How this Job will be executed. Defaults to :py:class:`multi_or_in_process_executor`,\n            which can be switched between multi-process and in-process modes of execution. The\n            default mode of execution is multi-process.\n\n\n    Returns:\n        UnresolvedAssetJobDefinition: The job, which can be placed inside a code location.\n\n    Examples:\n        .. code-block:: python\n\n            # A job that targets all assets in the code location:\n            @asset\n            def asset1():\n                ...\n\n            defs = Definitions(\n                assets=[asset1],\n                jobs=[define_asset_job("all_assets")],\n            )\n\n            # A job that targets a single asset\n            @asset\n            def asset1():\n                ...\n\n            defs = Definitions(\n                assets=[asset1],\n                jobs=[define_asset_job("all_assets", selection=[asset1])],\n            )\n\n            # A job that targets all the assets in a group:\n            defs = Definitions(\n                assets=assets,\n                jobs=[define_asset_job("marketing_job", selection=AssetSelection.groups("marketing"))],\n            )\n\n            @observable_source_asset\n            def source_asset():\n                ...\n\n            # A job that observes a source asset:\n            defs = Definitions(\n                assets=assets,\n                jobs=[define_asset_job("observation_job", selection=[source_asset])],\n            )\n\n            # Resources are supplied to the assets, not the job:\n            @asset(required_resource_keys={"slack_client"})\n            def asset1():\n                ...\n\n            defs = Definitions(\n                assets=[asset1],\n                jobs=[define_asset_job("all_assets")],\n                resources={"slack_client": prod_slack_client},\n            )\n\n    '
    from dagster._core.definitions import AssetSelection
    if selection is None:
        resolved_selection = AssetSelection.all()
    else:
        resolved_selection = AssetSelection.from_coercible(selection)
    return UnresolvedAssetJobDefinition(name=name, selection=resolved_selection, config=config, description=description, tags=tags, metadata=metadata, partitions_def=partitions_def, executor_def=executor_def, hooks=hooks)