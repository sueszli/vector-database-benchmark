from __future__ import annotations
DOCUMENTATION = '\nconnection: test_connection_override\nshort_description: test connection plugin used in tests\ndescription:\n- This is a test connection plugin used for shell testing\nauthor: ansible (@core)\nversion_added: historical\noptions:\n'
from ansible.plugins.connection import ConnectionBase

class Connection(ConnectionBase):
    """ test connection """
    transport = 'test_connection_override'

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._shell_type = 'powershell'
        super(Connection, self).__init__(*args, **kwargs)

    def _connect(self):
        if False:
            while True:
                i = 10
        pass

    def exec_command(self, cmd, in_data=None, sudoable=True):
        if False:
            for i in range(10):
                print('nop')
        pass

    def put_file(self, in_path, out_path):
        if False:
            i = 10
            return i + 15
        pass

    def fetch_file(self, in_path, out_path):
        if False:
            for i in range(10):
                print('nop')
        pass

    def close(self):
        if False:
            return 10
        pass