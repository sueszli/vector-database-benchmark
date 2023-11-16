from typing import Any, Optional
from dagster import InputContext, OutputContext
from dagster._config.pythonic_config import ConfigurableIOManager, ResourceDependency
from dagster._core.definitions.events import AssetKey, AssetMaterialization
from dagster._core.definitions.metadata import TextMetadataValue
from dagster._core.event_api import EventRecordsFilter
from dagster._core.events import DagsterEventType
from dagster._core.events.log import EventLogEntry
from dagster._core.instance import DagsterInstance
from dagster._core.storage.io_manager import IOManager

def get_text_metadata_value(materialization: AssetMaterialization, key: str) -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    metadata_value = materialization.metadata.get(key)
    return metadata_value.value if isinstance(metadata_value, TextMetadataValue) else None

def latest_materialization_log_entry(instance: DagsterInstance, asset_key: AssetKey, partition_key: Optional[str]=None) -> Optional[EventLogEntry]:
    if False:
        while True:
            i = 10
    event_records = [*instance.get_event_records(event_records_filter=EventRecordsFilter(event_type=DagsterEventType.ASSET_MATERIALIZATION, asset_key=asset_key, asset_partitions=[partition_key] if partition_key else None), limit=1)]
    return event_records[0].event_log_entry if event_records else None

class BranchingIOManager(ConfigurableIOManager):
    """A branching I/O manager composes two I/O managers.

    1) The parent I/O manager, typically your production environment.
    2) The branch I/O manager, typically a development or branched environment.

    The objective of this to allow a developer to safely read from a production
    environment and then write to a separate development environment. Once data
    has been written to the branch environment subsequent reads of that asset
    are sourced from the branch environment. This bookkeeping is done in Dagster's
    asset catalog by emitting AssetMaterializations with metadata.

    This is designed for iterative development on asset graphs, especially
    where assets early in the graph are large and expensive to compute. One can
    iteratively develop on downstream assets in that graph safely.

    Some storage systems branching functionality natively. Examples include Snowflake's
    CLONE feature. Branching I/O managers allow users to implement that functionality
    in more flexible software layer over arbitrary storage systems.
    """
    parent_io_manager: ResourceDependency[IOManager]
    branch_io_manager: ResourceDependency[IOManager]
    branch_name: str = 'dev'
    branch_metadata_key: str = 'io_manager_branch'

    def load_input(self, context: InputContext) -> Any:
        if False:
            return 10
        if not context.has_asset_key:
            return self.branch_io_manager.load_input(context)
        else:
            event_log_entry = latest_materialization_log_entry(instance=context.instance, asset_key=context.asset_key, partition_key=context.partition_key if context.has_partition_key else None)
            if event_log_entry and event_log_entry.asset_materialization and (get_text_metadata_value(event_log_entry.asset_materialization, self.branch_metadata_key) == self.branch_name):
                context.log.info(f'Branching Manager: Loading "{context.asset_key.to_user_string()}" from "{self.branch_name}"')
                return self.branch_io_manager.load_input(context)
            context.log.info(f'Branching Manager Loading "{context.asset_key.to_user_string()}" from parent')
            return self.parent_io_manager.load_input(context)

    def handle_output(self, context: OutputContext, obj: Any) -> None:
        if False:
            i = 10
            return i + 15
        self.branch_io_manager.handle_output(context, obj)
        if context.has_asset_key:
            context.add_output_metadata({self.branch_metadata_key: self.branch_name})
            context.log.info(f'Branching Manager: Writing "{context.asset_key.to_user_string()}" to branch "{self.branch_name}"')