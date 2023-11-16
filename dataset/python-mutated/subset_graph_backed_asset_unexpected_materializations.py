from dagster import AssetKey, Out, op
from .subset_graph_backed_asset import defs

@op(out={'foo_1': Out(), 'foo_2': Out()})
def foo():
    if False:
        for i in range(10):
            print('nop')
    return (1, 2)
defs.get_job_def('my_graph_assets').execute_in_process(asset_selection=[AssetKey('baz_asset')])