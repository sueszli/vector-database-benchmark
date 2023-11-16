from datasette import hookimpl
from datasette.utils import escape_fts

@hookimpl
def prepare_connection(conn):
    if False:
        while True:
            i = 10
    conn.create_function('escape_fts', 1, escape_fts)