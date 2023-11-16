from __future__ import annotations
DOCUMENTATION = '\nconnection: test_connection_default\nshort_description: test connection plugin used in tests\ndescription:\n- This is a test connection plugin used for shell testing\nauthor: ansible (@core)\nversion_added: historical\noptions:\n'
from ansible.plugins.connection import ConnectionBase

class Connection(ConnectionBase):
    """ test connnection """
    transport = 'test_connection_default'

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(Connection, self).__init__(*args, **kwargs)

    def _connect(self):
        if False:
            print('Hello World!')
        pass

    def exec_command(self, cmd, in_data=None, sudoable=True):
        if False:
            print('Hello World!')
        pass

    def put_file(self, in_path, out_path):
        if False:
            for i in range(10):
                print('nop')
        pass

    def fetch_file(self, in_path, out_path):
        if False:
            for i in range(10):
                print('nop')
        pass

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        pass