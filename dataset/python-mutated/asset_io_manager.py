from dagster_aws.s3 import S3PickleIOManager, S3Resource
from dagster import Definitions, asset

@asset
def upstream_asset():
    if False:
        print('Hello World!')
    return [1, 2, 3]

@asset
def downstream_asset(upstream_asset):
    if False:
        i = 10
        return i + 15
    return upstream_asset + [4]
defs = Definitions(assets=[upstream_asset, downstream_asset], resources={'io_manager': S3PickleIOManager(s3_resource=S3Resource(), s3_bucket='my-bucket')})