from __future__ import annotations
from ansible.plugins.action import ActionBase

class ActionModule(ActionBase):

    def run(self, tmp=None, task_vars=None):
        if False:
            while True:
                i = 10
        del tmp
        try:
            self._task.args['_raw_params'] = self._task.args.pop('cmd')
        except KeyError:
            pass
        shell_action = self._shared_loader_obj.action_loader.get('ansible.legacy.shell', task=self._task, connection=self._connection, play_context=self._play_context, loader=self._loader, templar=self._templar, shared_loader_obj=self._shared_loader_obj)
        result = shell_action.run(task_vars=task_vars)
        result.pop('failed', None)
        return result