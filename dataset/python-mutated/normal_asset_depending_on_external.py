from dagster import AssetSpec, Definitions, asset, external_assets_from_specs
raw_logs = AssetSpec('raw_logs')
processed_logs = AssetSpec('processed_logs', deps=[raw_logs])

@asset(deps=[processed_logs])
def aggregated_logs() -> None:
    if False:
        while True:
            i = 10
    ...
defs = Definitions(assets=[aggregated_logs, *external_assets_from_specs([raw_logs, processed_logs])])