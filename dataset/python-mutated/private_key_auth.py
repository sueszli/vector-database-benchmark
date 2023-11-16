from dagster import asset

@asset
def iris_dataset():
    if False:
        return 10
    return None
from dagster_snowflake_pandas import SnowflakePandasIOManager
from dagster import Definitions, EnvVar
defs = Definitions(assets=[iris_dataset], resources={'io_manager': SnowflakePandasIOManager(account='abc1234.us-east-1', user=EnvVar('SNOWFLAKE_USER'), private_key=EnvVar('SNOWFLAKE_PK'), private_key_password=EnvVar('SNOWFLAKE_PK_PASSWORD'), database='FLOWERS')})
defs = Definitions(assets=[iris_dataset], resources={'io_manager': SnowflakePandasIOManager(account='abc1234.us-east-1', user=EnvVar('SNOWFLAKE_USER'), private_key_path='/path/to/private/key/file.p8', private_key_password=EnvVar('SNOWFLAKE_PK_PASSWORD'), database='FLOWERS')})