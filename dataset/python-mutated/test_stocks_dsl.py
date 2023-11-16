EXAMPLE_TEXT = '\nstocks_to_index:\n  - ticker: MSFT\n  - ticker: AAPL\n  - ticker: GOOG\n  - ticker: AMZN\n  - ticker: META\n  - ticker: NVDA\n  - ticker: TSLA\nindex_strategy:\n  type: weighted_average\nforecast:\n  days: 30\n'
import yaml
from assets_yaml_dsl.domain_specific_dsl.stocks_dsl import assets_defs_from_stock_assets
from dagster import AssetKey
from dagster._core.definitions import materialize
from dagster._core.pipes.subprocess import PipesSubprocessClient
from examples.experimental.assets_yaml_dsl.assets_yaml_dsl.domain_specific_dsl.stocks_dsl import build_stock_assets_object

def test_stocks_dsl():
    if False:
        while True:
            i = 10
    stocks_dsl_document = yaml.safe_load(EXAMPLE_TEXT)
    stock_assets = build_stock_assets_object(stocks_dsl_document)
    assets_defs = assets_defs_from_stock_assets(stock_assets)
    fetch_ticker_assets_def = assets_defs[0]
    assert fetch_ticker_assets_def.keys == {AssetKey('MSFT'), AssetKey('AAPL'), AssetKey('GOOG'), AssetKey('AMZN'), AssetKey('META'), AssetKey('NVDA'), AssetKey('TSLA')}
    index_strategy_asset_def = assets_defs[1]
    assert index_strategy_asset_def.keys == {AssetKey('index_strategy')}
    forecast_asset_def = assets_defs[2]
    assert forecast_asset_def.keys == {AssetKey('forecast')}

def test_materialize_stocks_dsl():
    if False:
        for i in range(10):
            print('nop')
    stocks_dsl_document = yaml.safe_load(EXAMPLE_TEXT)
    stock_assets = build_stock_assets_object(stocks_dsl_document)
    assets_defs = assets_defs_from_stock_assets(stock_assets)
    assert materialize(assets=assets_defs, resources={'pipes_subprocess_client': PipesSubprocessClient()}).success