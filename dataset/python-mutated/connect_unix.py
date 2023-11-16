import os
import sqlalchemy

def connect_unix_socket() -> sqlalchemy.engine.base.Engine:
    if False:
        i = 10
        return i + 15
    'Initializes a Unix socket connection pool for a Cloud SQL instance of Postgres.'
    db_user = os.environ['DB_USER']
    db_pass = os.environ['DB_PASS']
    db_name = os.environ['DB_NAME']
    unix_socket_path = os.environ['INSTANCE_UNIX_SOCKET']
    pool = sqlalchemy.create_engine(sqlalchemy.engine.url.URL.create(drivername='postgresql+pg8000', username=db_user, password=db_pass, database=db_name, query={'unix_sock': f'{unix_socket_path}/.s.PGSQL.5432'}), pool_size=5, max_overflow=2, pool_timeout=30, pool_recycle=1800)
    return pool