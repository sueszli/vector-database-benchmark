from __future__ import annotations
from ansible.plugins.action import ActionBase

class ActionModule(ActionBase):
    BYPASS_HOST_LOOP = True

    def run(self, tmp=None, task_vars=None):
        if False:
            for i in range(10):
                print('nop')
        result = super(ActionModule, self).run(tmp, task_vars)
        result['bypass_inventory_hostname'] = task_vars['inventory_hostname']
        return result