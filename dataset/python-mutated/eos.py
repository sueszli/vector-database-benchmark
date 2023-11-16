from __future__ import annotations
from ansible.plugins.action.normal import ActionModule as ActionBase

class ActionModule(ActionBase):

    def run(self, tmp=None, task_vars=None):
        if False:
            while True:
                i = 10
        result = super(ActionModule, self).run(tmp, task_vars)
        result['action_plugin'] = 'eos'
        return result