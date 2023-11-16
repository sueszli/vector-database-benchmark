from dagster import asset

@asset
def iris_data():
    if False:
        while True:
            i = 10
    return None
from dagster_gcp_pyspark import BigQueryPySparkIOManager
from dagster import Definitions
defs = Definitions(assets=[iris_data], resources={'io_manager': BigQueryPySparkIOManager(project='my-gcp-project', location='us-east5', dataset='IRIS', temporary_gcs_bucket='my-gcs-bucket')})