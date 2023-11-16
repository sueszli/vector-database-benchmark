import os
from named_db import create_named_db_client

def test_named_db():
    if False:
        for i in range(10):
            print('nop')
    os.environ['DATASTORE_DATABASE'] = 'test'
    client = create_named_db_client()
    assert client.database == 'test'