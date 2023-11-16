from __future__ import annotations
from ansible.plugins.action import ActionBase

class ActionModule(ActionBase):
    TRANSFERS_FILES = False

    def run(self, tmp=None, task_vars=None):
        if False:
            print('Hello World!')
        if task_vars is None:
            task_vars = dict()
        if self._task.environment and any(self._task.environment):
            self._display.warning('raw module does not support the environment keyword')
        result = super(ActionModule, self).run(tmp, task_vars)
        del tmp
        if self._play_context.check_mode:
            result['skipped'] = True
            return result
        executable = self._task.args.get('executable', False)
        result.update(self._low_level_execute_command(self._task.args.get('_raw_params'), executable=executable))
        result['changed'] = True
        if 'rc' in result and result['rc'] != 0:
            result['failed'] = True
            result['msg'] = 'non-zero return code'
        return result