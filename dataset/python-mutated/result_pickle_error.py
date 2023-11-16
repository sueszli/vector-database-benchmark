from __future__ import annotations
from ansible.plugins.action import ActionBase
from jinja2 import Undefined

class ActionModule(ActionBase):

    def run(self, tmp=None, task_vars=None):
        if False:
            for i in range(10):
                print('nop')
        return {'obj': Undefined('obj')}