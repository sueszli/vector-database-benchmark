from pandas import DataFrame
from dagster import Definitions, In, Out, job, op
from .mylib import s3_io_manager, snowflake_io_manager, train_recommender_model

@op(ins={'raw_users': In(input_manager_key='warehouse')}, out={'users': Out(io_manager_key='warehouse')})
def build_users(raw_users: DataFrame) -> DataFrame:
    if False:
        return 10
    users_df = raw_users.dropna()
    return users_df

@op(out={'users_recommender_model': Out(io_manager_key='object_store')})
def build_user_recommender_model(users: DataFrame):
    if False:
        for i in range(10):
            print('nop')
    users_recommender_model = train_recommender_model(users)
    return users_recommender_model

@job(resource_defs={'warehouse': snowflake_io_manager, 'object_store': s3_io_manager})
def users_recommender_job():
    if False:
        print('Hello World!')
    build_user_recommender_model(build_users())
defs = Definitions(jobs=[users_recommender_job])