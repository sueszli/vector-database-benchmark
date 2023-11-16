from azure.ai.ml import MLClient
from azure.ai.resources.operations import IndexDataSource, ACSOutputConfig
from ._index_config import IndexConfig

def ingest_data_to_index(*, client: MLClient, index_config: IndexConfig, source_config: IndexDataSource, acs_config: ACSOutputConfig=None):
    if False:
        for i in range(10):
            print('nop')
    pipeline = source_config._createComponent(index_config=index_config, acs_config=acs_config)
    pipeline.settings = {'default_compute': 'serverless'}
    client.jobs.create_or_update(pipeline)