import datetime
from dagster import AssetMaterialization, AssetSpec, Definitions, SensorEvaluationContext, SensorResult, external_asset_from_spec, sensor

def utc_now_str() -> str:
    if False:
        i = 10
        return i + 15
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d, %H:%M:%S')

@sensor()
def keep_external_asset_a_up_to_date(context: SensorEvaluationContext) -> SensorResult:
    if False:
        print('Hello World!')
    return SensorResult(asset_events=[AssetMaterialization(asset_key='external_asset_a', metadata={'source': f'From sensor "{context.sensor_name}" at UTC time "{utc_now_str()}"'})])
defs = Definitions(assets=[external_asset_from_spec(AssetSpec('external_asset_a'))], sensors=[keep_external_asset_a_up_to_date])