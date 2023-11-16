from pandas import DataFrame
from dagster import Definitions, SourceAsset, asset, define_asset_job
from .mylib import s3_io_manager, snowflake_io_manager, train_recommender_model
raw_users = SourceAsset(key='raw_users', io_manager_key='warehouse')

@asset(io_manager_key='warehouse')
def users(raw_users: DataFrame) -> DataFrame:
    if False:
        print('Hello World!')
    users_df = raw_users.dropna()
    return users_df

@asset(io_manager_key='object_store')
def user_recommender_model(users: DataFrame):
    if False:
        for i in range(10):
            print('nop')
    users_recommender_model = train_recommender_model(users)
    return users_recommender_model
users_recommender_job = define_asset_job(name='users_recommender_job')
defs = Definitions(assets=[raw_users, users, user_recommender_model], jobs=[users_recommender_job], resources={'warehouse': snowflake_io_manager, 'object_store': s3_io_manager})