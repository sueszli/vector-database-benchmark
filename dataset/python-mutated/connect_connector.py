import os
from google.cloud.sql.connector import Connector, IPTypes
import pg8000
import sqlalchemy

def connect_with_connector() -> sqlalchemy.engine.base.Engine:
    if False:
        i = 10
        return i + 15
    '\n    Initializes a connection pool for a Cloud SQL instance of Postgres.\n\n    Uses the Cloud SQL Python Connector package.\n    '
    instance_connection_name = os.environ['INSTANCE_CONNECTION_NAME']
    db_user = os.environ['DB_USER']
    db_pass = os.environ['DB_PASS']
    db_name = os.environ['DB_NAME']
    ip_type = IPTypes.PRIVATE if os.environ.get('PRIVATE_IP') else IPTypes.PUBLIC
    connector = Connector()

    def getconn() -> pg8000.dbapi.Connection:
        if False:
            print('Hello World!')
        conn: pg8000.dbapi.Connection = connector.connect(instance_connection_name, 'pg8000', user=db_user, password=db_pass, db=db_name, ip_type=ip_type)
        return conn
    pool = sqlalchemy.create_engine('postgresql+pg8000://', creator=getconn, pool_size=5, max_overflow=2, pool_timeout=30, pool_recycle=1800)
    return pool