from dagster import AssetMaterialization, AssetSpec, Definitions, OpExecutionContext, external_asset_from_spec, job, op

@op
def an_op(context: OpExecutionContext) -> None:
    if False:
        while True:
            i = 10
    context.log_event(AssetMaterialization(asset_key='external_asset'))

@job
def a_job() -> None:
    if False:
        i = 10
        return i + 15
    an_op()
defs = Definitions(assets=[external_asset_from_spec(AssetSpec('external_asset'))], jobs=[a_job])