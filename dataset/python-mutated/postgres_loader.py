def insert_to_postgres(db, df, table: str):
    if False:
        print('Hello World!')
    df.to_sql(table, db.engine, if_exists='append', index=False)