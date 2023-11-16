from __future__ import annotations
from ansible.module_utils.common.text.converters import to_native
from ansible.plugins.connection import ConnectionBase
DOCUMENTATION = '\n    connection: localconn\n    short_description: do stuff local\n    description:\n        - does stuff\n    options:\n      connectionvar:\n        description:\n            - something we set\n        default: the_default\n        vars:\n            - name: ansible_localconn_connectionvar\n'

class Connection(ConnectionBase):
    transport = 'local'
    has_pipelining = True

    def _connect(self):
        if False:
            i = 10
            return i + 15
        return self

    def exec_command(self, cmd, in_data=None, sudoable=True):
        if False:
            while True:
                i = 10
        stdout = 'localconn ran {0}'.format(to_native(cmd))
        stderr = 'connectionvar is {0}'.format(to_native(self.get_option('connectionvar')))
        return (0, stdout, stderr)

    def put_file(self, in_path, out_path):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('just a test')

    def fetch_file(self, in_path, out_path):
        if False:
            while True:
                i = 10
        raise NotImplementedError('just a test')

    def close(self):
        if False:
            while True:
                i = 10
        self._connected = False