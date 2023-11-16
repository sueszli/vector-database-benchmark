from dagster import asset

def get_snowflake_connection():
    if False:
        for i in range(10):
            print('nop')
    pass

@asset
def orders():
    if False:
        i = 10
        return i + 15
    pass

@asset(deps=[orders])
def returns():
    if False:
        print('Hello World!')
    conn = get_snowflake_connection()
    conn.execute("CREATE TABLE returns AS SELECT * from orders WHERE status = 'RETURNED'")