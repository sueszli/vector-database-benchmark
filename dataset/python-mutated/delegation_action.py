from __future__ import annotations
from ansible.plugins.action import ActionBase

class ActionModule(ActionBase):

    def run(self, tmp=None, task_vars=None):
        if False:
            for i in range(10):
                print('nop')
        return {'remote_password': self._connection.get_option('remote_password')}