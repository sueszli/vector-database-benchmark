from dagster import asset

@asset
def iris_data():
    if False:
        for i in range(10):
            print('nop')
    return None

@asset
def rose_data():
    if False:
        print('Hello World!')
    return None
from typing import Optional, Sequence, Type
import pandas as pd
from dagster_gcp import BigQueryIOManager
from dagster_gcp_pandas import BigQueryPandasTypeHandler
from dagster_gcp_pyspark import BigQueryPySparkTypeHandler
from dagster import DbTypeHandler, Definitions

class MyBigQueryIOManager(BigQueryIOManager):

    @staticmethod
    def type_handlers() -> Sequence[DbTypeHandler]:
        if False:
            return 10
        'type_handlers should return a list of the TypeHandlers that the I/O manager can use.\n\n        Here we return the BigQueryPandasTypeHandler and BigQueryPySparkTypeHandler so that the I/O\n        manager can store Pandas DataFrames and PySpark DataFrames.\n        '
        return [BigQueryPandasTypeHandler(), BigQueryPySparkTypeHandler()]

    @staticmethod
    def default_load_type() -> Optional[Type]:
        if False:
            i = 10
            return i + 15
        'If an asset is not annotated with an return type, default_load_type will be used to\n        determine which TypeHandler to use to store and load the output.\n\n        In this case, unannotated assets will be stored and loaded as Pandas DataFrames.\n        '
        return pd.DataFrame
defs = Definitions(assets=[iris_data, rose_data], resources={'io_manager': MyBigQueryIOManager(project='my-gcp-project', dataset='FLOWERS')})