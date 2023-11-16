import os
from azure.storage.blob.blockblobservice import BlockBlobService
from wal_e.cmd import parse_boolean_envvar

def no_real_wabs_credentials():
    if False:
        while True:
            i = 10
    "Helps skip integration tests without live credentials.\n\n    Phrased in the negative to make it read better with 'skipif'.\n    "
    if parse_boolean_envvar(os.getenv('WALE_WABS_INTEGRATION_TESTS')) is not True:
        return True
    for e_var in ('WABS_ACCOUNT_NAME', 'WABS_ACCESS_KEY'):
        if os.getenv(e_var) is None:
            return True
    return False

def apathetic_container_delete(container_name, *args, **kwargs):
    if False:
        while True:
            i = 10
    conn = BlockBlobService(*args, **kwargs)
    conn.delete_container(container_name)
    return conn

def insistent_container_delete(conn, container_name):
    if False:
        i = 10
        return i + 15
    while True:
        success = conn.delete_container(container_name)
        if not success:
            continue
        break

def insistent_container_create(conn, container_name, *args, **kwargs):
    if False:
        return 10
    while True:
        success = conn.create_container(container_name)
        if not success:
            continue
        break
    return success

class FreshContainer(object):

    def __init__(self, container_name, *args, **kwargs):
        if False:
            print('Hello World!')
        self.container_name = container_name
        self.conn_args = args or [os.environ.get('WABS_ACCOUNT_NAME'), os.environ.get('WABS_ACCESS_KEY')]
        self.conn_kwargs = kwargs
        self.created_container = False

    def __enter__(self):
        if False:
            return 10
        self.conn = apathetic_container_delete(self.container_name, *self.conn_args, **self.conn_kwargs)
        return self

    def create(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        container = insistent_container_create(self.conn, self.container_name, *args, **kwargs)
        self.created_container = True
        return container

    def __exit__(self, typ, value, traceback):
        if False:
            while True:
                i = 10
        if not self.created_container:
            return False
        insistent_container_delete(self.conn, self.container_name)
        return False