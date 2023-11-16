import sqlalchemy

def init_tcp_connection_engine(db_user: str, db_pass: str, db_name: str, db_host: str) -> sqlalchemy.engine.base.Engine:
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates a connection to the database using tcp socket.\n    '
    host_args = db_host.split(':')
    (db_hostname, db_port) = (host_args[0], int(host_args[1]))
    pool = sqlalchemy.create_engine(sqlalchemy.engine.url.URL.create(drivername='postgresql+pg8000', username=db_user, password=db_pass, host=db_hostname, port=db_port, database=db_name))
    print('Created TCP connection pool')
    return pool

def init_unix_connection_engine(db_user: str, db_pass: str, db_name: str, instance_connection_name: str, db_socket_dir: str) -> sqlalchemy.engine.base.Engine:
    if False:
        print('Hello World!')
    '\n    Creates a connection to the database using unix socket.\n    '
    pool = sqlalchemy.create_engine(sqlalchemy.engine.url.URL.create(drivername='postgresql+pg8000', username=db_user, password=db_pass, database=db_name, query={'unix_sock': '{}/{}/.s.PGSQL.5432'.format(db_socket_dir, instance_connection_name)}))
    print('Created Unix socket connection pool')
    return pool

def init_db(db_user: str, db_pass: str, db_name: str, table_name: str, instance_connection_name: str=None, db_socket_dir: str=None, db_host: str=None) -> sqlalchemy.engine.base.Engine:
    if False:
        print('Hello World!')
    "Starts a connection to the database and creates voting table if it doesn't exist."
    if db_host:
        db = init_tcp_connection_engine(db_user, db_pass, db_name, db_host)
    else:
        db = init_unix_connection_engine(db_user, db_pass, db_name, instance_connection_name, db_socket_dir)
    with db.connect() as conn:
        conn.execute(f'CREATE TABLE IF NOT EXISTS {table_name} ( vote_id SERIAL NOT NULL, time_cast timestamp NOT NULL, team VARCHAR(6) NOT NULL, voter_email BYTEA, PRIMARY KEY (vote_id) );')
    print(f'Created table {table_name} in db {db_name}')
    return db