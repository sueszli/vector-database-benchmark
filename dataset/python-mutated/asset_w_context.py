from dagster import AssetExecutionContext, asset

@asset
def context_asset(context: AssetExecutionContext):
    if False:
        i = 10
        return i + 15
    context.log.info(f'My run ID is {context.run_id}')
    ...