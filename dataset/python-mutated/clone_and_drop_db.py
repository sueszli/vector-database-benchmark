import os
from dagster_snowflake import SnowflakeResource
from dagster import In, Nothing, graph, op

@op
def drop_database_clone(snowflake: SnowflakeResource):
    if False:
        for i in range(10):
            print('nop')
    with snowflake.get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"DROP DATABASE IF EXISTS PRODUCTION_CLONE_{os.environ['DAGSTER_CLOUD_PULL_REQUEST_ID']}")

@op(ins={'start': In(Nothing)})
def clone_production_database(snowflake: SnowflakeResource):
    if False:
        for i in range(10):
            print('nop')
    with snowflake.get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f'''CREATE DATABASE PRODUCTION_CLONE_{os.environ['DAGSTER_CLOUD_PULL_REQUEST_ID']} CLONE "PRODUCTION"''')

@graph
def clone_prod():
    if False:
        while True:
            i = 10
    clone_production_database(start=drop_database_clone())

@graph
def drop_prod_clone():
    if False:
        print('Hello World!')
    drop_database_clone()