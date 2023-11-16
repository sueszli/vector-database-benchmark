from dagster_aws.s3 import S3PickleIOManager, S3Resource
from dagster import Definitions, FilesystemIOManager, asset

@asset(io_manager_key='s3_io_manager')
def upstream_asset():
    if False:
        i = 10
        return i + 15
    return [1, 2, 3]

@asset(io_manager_key='fs_io_manager')
def downstream_asset(upstream_asset):
    if False:
        while True:
            i = 10
    return upstream_asset + [4]
defs = Definitions(assets=[upstream_asset, downstream_asset], resources={'s3_io_manager': S3PickleIOManager(s3_resource=S3Resource(), s3_bucket='my-bucket'), 'fs_io_manager': FilesystemIOManager()})