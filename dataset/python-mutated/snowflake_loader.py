def insert_to_snowflake(db, df, table):
    if False:
        for i in range(10):
            print('nop')
    df.to_sql(table, db.engine, if_exists='append', index=False)