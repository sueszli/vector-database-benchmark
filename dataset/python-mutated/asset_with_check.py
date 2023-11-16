import pandas as pd
from dagster import AssetCheckResult, AssetCheckSpec, AssetExecutionContext, Definitions, Output, asset

@asset(check_specs=[AssetCheckSpec(name='orders_id_has_no_nulls', asset='orders')])
def orders(context: AssetExecutionContext):
    if False:
        while True:
            i = 10
    orders_df = pd.DataFrame({'order_id': [1, 2], 'item_id': [432, 878]})
    orders_df.to_csv('orders')
    yield Output(value=None)
    num_null_order_ids = orders_df['order_id'].isna().sum()
    yield AssetCheckResult(passed=bool(num_null_order_ids == 0))
defs = Definitions(assets=[orders])