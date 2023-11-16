import os
import ssl
import sqlalchemy

def connect_tcp_socket() -> sqlalchemy.engine.base.Engine:
    if False:
        i = 10
        return i + 15
    'Initializes a TCP connection pool for a Cloud SQL instance of Postgres.'
    db_host = os.environ['INSTANCE_HOST']
    db_user = os.environ['DB_USER']
    db_pass = os.environ['DB_PASS']
    db_name = os.environ['DB_NAME']
    db_port = os.environ['DB_PORT']
    connect_args = {}
    if os.environ.get('DB_ROOT_CERT'):
        db_root_cert = os.environ['DB_ROOT_CERT']
        db_cert = os.environ['DB_CERT']
        db_key = os.environ['DB_KEY']
        ssl_context = ssl.SSLContext()
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        ssl_context.load_verify_locations(db_root_cert)
        ssl_context.load_cert_chain(db_cert, db_key)
        connect_args['ssl_context'] = ssl_context
    pool = sqlalchemy.create_engine(sqlalchemy.engine.url.URL.create(drivername='postgresql+pg8000', username=db_user, password=db_pass, host=db_host, port=db_port, database=db_name), connect_args=connect_args, pool_size=5, max_overflow=2, pool_timeout=30, pool_recycle=1800)
    return pool