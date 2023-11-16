from __future__ import annotations
DOCUMENTATION = "\nauthor: Ansible Core Team\nconnection: delegation_connection\nshort_description: Test connection for delegated host check\ndescription:\n- Some further description that you don't care about.\noptions:\n  remote_password:\n    description: The remote password\n    type: str\n    vars:\n    - name: ansible_password\n    # Tests that an aliased key gets the -k option which hardcodes the value to password\n    aliases:\n    - password\n"
from ansible.plugins.connection import ConnectionBase

class Connection(ConnectionBase):
    transport = 'delegation_connection'
    has_pipelining = True

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(Connection, self).__init__(*args, **kwargs)

    def _connect(self):
        if False:
            while True:
                i = 10
        super(Connection, self)._connect()

    def exec_command(self, cmd, in_data=None, sudoable=True):
        if False:
            return 10
        super(Connection, self).exec_command(cmd, in_data, sudoable)

    def put_file(self, in_path, out_path):
        if False:
            i = 10
            return i + 15
        super(Connection, self).put_file(in_path, out_path)

    def fetch_file(self, in_path, out_path):
        if False:
            for i in range(10):
                print('nop')
        super(Connection, self).fetch_file(in_path, out_path)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        super(Connection, self).close()