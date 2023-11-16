from dagster import AssetSpec, Definitions, external_assets_from_specs
asset_one = AssetSpec('asset_one')
asset_two = AssetSpec('asset_two', deps=[asset_one])
defs = Definitions(assets=external_assets_from_specs([asset_one, asset_two]))

def do_report_runless_asset_event(instance) -> None:
    if False:
        print('Hello World!')
    from dagster import AssetMaterialization
    instance.report_runless_asset_event(AssetMaterialization('asset_one', metadata={'nrows': 10, 'source': 'From this script.'}))