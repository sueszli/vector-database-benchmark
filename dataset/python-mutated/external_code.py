import pandas as pd
from dagster_pipes import PipesContext, open_dagster_pipes

def main():
    if False:
        print('Hello World!')
    orders_df = pd.DataFrame({'order_id': [1, 2], 'item_id': [432, 878]})
    context = PipesContext.get()
    context.report_asset_check(asset_key='my_asset', passed=orders_df[['item_id']].notnull().all().bool(), check_name='no_empty_order_check')
if __name__ == '__main__':
    with open_dagster_pipes():
        main()