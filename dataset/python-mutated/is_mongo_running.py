import sys

def _is_mongo_running(db_host: str, db_port: int, db_name: str, connection_timeout_ms: int) -> bool:
    if False:
        print('Hello World!')
    'Connect to mongo with connection logic that mirrors the st2 code.\n\n    In particular, this is based on st2common.models.db.db_setup().\n    This should not import the st2 code as it should be self-contained.\n    '
    import mongoengine
    from pymongo.errors import ConnectionFailure
    from pymongo.errors import ServerSelectionTimeoutError
    connection = mongoengine.connection.connect(db_name, host=db_host, port=db_port, connectTimeoutMS=connection_timeout_ms, serverSelectionTimeoutMS=connection_timeout_ms)
    try:
        connection.admin.command('ismaster')
    except (ConnectionFailure, ServerSelectionTimeoutError):
        return False
    return True
if __name__ == '__main__':
    args = dict(((k, v) for (k, v) in enumerate(sys.argv)))
    db_host = args.get(1, '127.0.0.1')
    db_port = args.get(2, 27017)
    db_name = args.get(3, 'st2-test')
    connection_timeout_ms = args.get(4, 3000)
    is_running = _is_mongo_running(db_host, int(db_port), db_name, int(connection_timeout_ms))
    exit_code = 0 if is_running else 1
    sys.exit(exit_code)