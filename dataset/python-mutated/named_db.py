def create_named_db_client():
    if False:
        for i in range(10):
            print('nop')
    import os
    from google.cloud import ndb
    database = os.environ.get('DATASTORE_DATABASE', 'prod-database')
    client = ndb.Client(database=database)
    return client