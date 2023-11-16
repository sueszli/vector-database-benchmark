from __future__ import annotations
from ansible.plugins.action import ActionBase

class ActionModule(ActionBase):
    TRANSFERS_FILES = False
    _VALID_ARGS = frozenset()

    def run(self, tmp=None, task_vars=None):
        if False:
            for i in range(10):
                print('nop')
        return {'changed': False}