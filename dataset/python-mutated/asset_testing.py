from dagster import AssetExecutionContext, asset, build_asset_context

@asset
def my_simple_asset():
    if False:
        return 10
    return [1, 2, 3]

def test_my_simple_asset():
    if False:
        print('Hello World!')
    result = my_simple_asset()
    assert result == [1, 2, 3]

@asset
def more_complex_asset(my_simple_asset):
    if False:
        i = 10
        return i + 15
    return my_simple_asset + [4, 5, 6]

def test_more_complex_asset():
    if False:
        return 10
    result = more_complex_asset([0])
    assert result == [0, 4, 5, 6]

@asset
def uses_context(context: AssetExecutionContext):
    if False:
        print('Hello World!')
    context.log.info(context.run_id)
    return 'bar'

def test_uses_context():
    if False:
        for i in range(10):
            print('nop')
    context = build_asset_context()
    result = uses_context(context)
    assert result == 'bar'
from typing import Any, Dict
import requests
from dagster import Config, ConfigurableResource

class MyConfig(Config):
    api_url: str

class MyAPIResource(ConfigurableResource):

    def query(self, url) -> Dict[str, Any]:
        if False:
            return 10
        return requests.get(url).json()

@asset
def uses_config_and_resource(config: MyConfig, my_api: MyAPIResource):
    if False:
        print('Hello World!')
    return my_api.query(config.api_url)

def test_uses_resource() -> None:
    if False:
        while True:
            i = 10
    result = uses_config_and_resource(config=MyConfig(api_url='https://dagster.io'), my_api=MyAPIResource())
    assert result == {'foo': 'bar'}