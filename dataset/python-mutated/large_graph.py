import itertools
from typing import List, NamedTuple, Optional, Sequence
from dagster import AssetsDefinition, AutoMaterializePolicy, DailyPartitionsDefinition, HourlyPartitionsDefinition, PartitionsDefinition, StaticPartitionsDefinition, asset, repository

class AssetLayerConfig(NamedTuple):
    n_assets: int
    n_upstreams_per_asset: int = 0
    partitions_def: Optional[PartitionsDefinition] = None

def build_assets(id: str, layer_configs: Sequence[AssetLayerConfig], auto_materialize_policy: AutoMaterializePolicy=AutoMaterializePolicy.eager()) -> List[AssetsDefinition]:
    if False:
        print('Hello World!')
    layers = []
    for layer_config in layer_configs:
        parent_index = 0
        layer = []
        for i in range(layer_config.n_assets):
            if layer_config.n_upstreams_per_asset > 0:
                non_argument_deps = {layers[-1][(parent_index + j) % len(layers[-1])].key for j in range(layer_config.n_upstreams_per_asset)}
                parent_index += layer_config.n_upstreams_per_asset
            else:
                non_argument_deps = set()

            @asset(partitions_def=layer_config.partitions_def, name=f'{id}_{len(layers)}_{i}', auto_materialize_policy=auto_materialize_policy, non_argument_deps=non_argument_deps)
            def _asset():
                if False:
                    for i in range(10):
                        print('nop')
                pass
            layer.append(_asset)
        layers.append(layer)
    return list(itertools.chain(*layers))
hourly = HourlyPartitionsDefinition(start_date='2022-01-01-00:00')
daily = DailyPartitionsDefinition(start_date='2022-01-01')
static = StaticPartitionsDefinition(partition_keys=[f'p{i}' for i in range(100)])

@repository
def auto_materialize_large_time_graph():
    if False:
        i = 10
        return i + 15
    return build_assets(id='hourly_to_daily', layer_configs=[AssetLayerConfig(n_assets=10, partitions_def=hourly), AssetLayerConfig(n_assets=50, n_upstreams_per_asset=5, partitions_def=hourly), AssetLayerConfig(n_assets=50, n_upstreams_per_asset=5, partitions_def=hourly), AssetLayerConfig(n_assets=100, n_upstreams_per_asset=4, partitions_def=daily), AssetLayerConfig(n_assets=100, n_upstreams_per_asset=4, partitions_def=daily)])

@repository
def auto_materialize_large_static_graph():
    if False:
        for i in range(10):
            print('nop')
    return build_assets(id='static_and_unpartitioned', layer_configs=[AssetLayerConfig(n_assets=10, partitions_def=None), AssetLayerConfig(n_assets=50, n_upstreams_per_asset=5, partitions_def=static), AssetLayerConfig(n_assets=50, n_upstreams_per_asset=5, partitions_def=static), AssetLayerConfig(n_assets=100, n_upstreams_per_asset=4, partitions_def=static), AssetLayerConfig(n_assets=100, n_upstreams_per_asset=4, partitions_def=None)])