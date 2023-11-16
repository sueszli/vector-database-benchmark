from datasette import hookimpl
import time

@hookimpl
def prepare_connection(conn):
    if False:
        return 10
    conn.create_function('sleep', 1, lambda n: time.sleep(float(n)))