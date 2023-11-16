from __future__ import annotations
from ansible.plugins.action import ActionBase
import sys
try:
    mod = sys.modules[__name__]
except KeyError:
    raise Exception(f'module {__name__} is not accessible via sys.modules, likely a pluginloader bug')

class ActionModule(ActionBase):
    TRANSFERS_FILES = False

    def run(self, tmp=None, task_vars=None):
        if False:
            return 10
        if task_vars is None:
            task_vars = dict()
        result = super(ActionModule, self).run(tmp, task_vars)
        del tmp
        result['changed'] = False
        result['msg'] = 'self-referential action loaded and ran successfully'
        return result