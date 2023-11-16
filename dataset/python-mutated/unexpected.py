from __future__ import annotations
from ansible.plugins.action import ActionBase

class ActionModule(ActionBase):
    TRANSFERS_FILES = False

    def run(self, tmp=None, task_vars=None):
        if False:
            return 10
        raise Exception('boom')