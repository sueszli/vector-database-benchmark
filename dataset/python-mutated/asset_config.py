import requests
from dagster import Config, asset

@asset
def my_upstream_asset() -> int:
    if False:
        print('Hello World!')
    return 5

class MyDownstreamAssetConfig(Config):
    api_endpoint: str

@asset
def my_downstream_asset(config: MyDownstreamAssetConfig, my_upstream_asset: int) -> int:
    if False:
        return 10
    data = requests.get(f'{config.api_endpoint}/data').json()
    return data['value'] + my_upstream_asset