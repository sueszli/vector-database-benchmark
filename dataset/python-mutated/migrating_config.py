from dagster import asset

@asset
def upstream_asset() -> int:
    if False:
        i = 10
        return i + 15
    return 1

def old_config() -> None:
    if False:
        print('Hello World!')
    from dagster import AssetExecutionContext, Definitions, asset

    @asset(config_schema={'conn_string': str, 'port': int})
    def an_asset(context: AssetExecutionContext, upstream_asset):
        if False:
            i = 10
            return i + 15
        assert context.op_config['conn_string']
        assert context.op_config['port']
    defs = Definitions(assets=[an_asset, upstream_asset])
    job_def = defs.get_implicit_global_asset_job_def()
    result = job_def.execute_in_process(run_config={'ops': {'an_asset': {'config': {'conn_string': 'foo', 'port': 1}}}})
    assert result.success

def new_config_schema() -> None:
    if False:
        print('Hello World!')
    from dagster import Config, Definitions, asset

    class AnAssetConfig(Config):
        conn_string: str
        port: int

    @asset
    def an_asset(upstream_asset, config: AnAssetConfig):
        if False:
            while True:
                i = 10
        assert config.conn_string
        assert config.port
    defs = Definitions(assets=[an_asset, upstream_asset])
    job_def = defs.get_implicit_global_asset_job_def()
    result = job_def.execute_in_process(run_config={'ops': {'an_asset': {'config': {'conn_string': 'foo', 'port': 1}}}})
    assert result.success

def new_config_schema_and_typed_run_config() -> None:
    if False:
        return 10
    from dagster import Config, Definitions, RunConfig, asset

    class AnAssetConfig(Config):
        conn_string: str
        port: int

    @asset
    def an_asset(upstream_asset, config: AnAssetConfig):
        if False:
            i = 10
            return i + 15
        assert config.conn_string
        assert config.port
    defs = Definitions(assets=[an_asset, upstream_asset])
    job_def = defs.get_implicit_global_asset_job_def()
    result = job_def.execute_in_process(run_config=RunConfig(ops={'an_asset': AnAssetConfig(conn_string='foo', port=1)}))
    assert result.success