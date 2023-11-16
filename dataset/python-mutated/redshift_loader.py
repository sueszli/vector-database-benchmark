from db.models import DetailedEvent
from psycopg2.errors import InternalError_

def transit_insert_to_redshift(db, df, table):
    if False:
        print('Hello World!')
    try:
        insert_df(db.pdredshift, df, table)
    except InternalError_ as e:
        print(repr(e))
        print('loading failed. check stl_load_errors')

def insert_df(pr, df, table):
    if False:
        print('Hello World!')
    pr.pandas_to_redshift(data_frame=df, redshift_table_name=table, append=True, delimiter='|')