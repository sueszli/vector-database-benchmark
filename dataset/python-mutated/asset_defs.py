import re
from typing import Any, Dict, List, Optional, Union
from dagster import AssetExecutionContext, AssetsDefinition, AssetSpec, MaterializeResult, multi_asset
from dagster._annotations import experimental
from dagster_embedded_elt.sling.resources import SlingMode, SlingResource

@experimental
def build_sling_asset(asset_spec: AssetSpec, source_stream: str, target_object: str, mode: SlingMode=SlingMode.FULL_REFRESH, primary_key: Optional[Union[str, List[str]]]=None, update_key: Optional[str]=None, source_options: Optional[Dict[str, Any]]=None, target_options: Optional[Dict[str, Any]]=None, sling_resource_key: str='sling') -> AssetsDefinition:
    if False:
        print('Hello World!')
    'Asset Factory for using Sling to sync data from a source stream to a target object.\n\n    Args:\n        asset_spec (AssetSpec): The AssetSpec to use to materialize this asset.\n        source_stream (str): The source stream to sync from. This can be a table, a query, or a path.\n        target_object (str): The target object to sync to. This can be a table, or a path.\n        mode (SlingMode, optional): The sync mode to use when syncing. Defaults to SlingMode.FULL_REFRESH.\n        primary_key (Optional[Union[str, List[str]]], optional): The optional primary key to use when syncing.\n        update_key (Optional[str], optional): The optional update key to use when syncing.\n        source_options (Optional[Dict[str, Any]], optional): Any optional Sling source options to use when syncing.\n        target_options (Optional[Dict[str, Any]], optional): Any optional target options to use when syncing.\n        sling_resource_key (str, optional): The resource key for the SlingResource. Defaults to "sling".\n\n    Examples:\n        Creating a Sling asset that syncs from a file to a table:\n\n        .. code-block:: python\n\n            asset_spec = AssetSpec(key=["main", "dest_tbl"])\n            asset_def = build_sling_asset(\n                    asset_spec=asset_spec,\n                    source_stream="file:///tmp/test.csv",\n                    target_object="main.dest_table",\n                    mode=SlingMode.INCREMENTAL,\n                    primary_key="id"\n            )\n\n        Creating a Sling asset that syncs from a table to a file with a full refresh:\n\n        .. code-block:: python\n\n            asset_spec = AssetSpec(key="test.csv")\n            asset_def = build_sling_asset(\n                    asset_spec=asset_spec,\n                    source_stream="main.dest_table",\n                    target_object="file:///tmp/test.csv",\n                    mode=SlingMode.FULL_REFRESH\n            )\n\n\n    '
    if primary_key is not None and (not isinstance(primary_key, list)):
        primary_key = [primary_key]

    @multi_asset(name=asset_spec.key.to_python_identifier(), compute_kind='sling', specs=[asset_spec], required_resource_keys={sling_resource_key})
    def sync(context: AssetExecutionContext) -> MaterializeResult:
        if False:
            return 10
        sling: SlingResource = getattr(context.resources, sling_resource_key)
        last_row_count_observed = None
        for stdout_line in sling.sync(source_stream=source_stream, target_object=target_object, mode=mode, primary_key=primary_key, update_key=update_key, source_options=source_options, target_options=target_options):
            match = re.search('(\\d+) rows', stdout_line)
            if match:
                last_row_count_observed = int(match.group(1))
            context.log.info(stdout_line)
        return MaterializeResult(metadata={} if last_row_count_observed is None else {'row_count': last_row_count_observed})
    return sync