from dagster import Array, Output, graph, op

@op(config_schema={'asset_key': Array(str), 'ops': str})
def read_materialization(context):
    if False:
        i = 10
        return i + 15
    asset_key = context.op_config['asset_key']
    from_graph = context.op_config['graph']
    context.log.info(f'Found materialization for asset key {asset_key} in {from_graph}')
    yield Output(asset_key)

@graph
def log_asset():
    if False:
        for i in range(10):
            print('nop')
    read_materialization()
log_asset_job = log_asset.to_job(description='Demo job that logs asset materializations from graphs')