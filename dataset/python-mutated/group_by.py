from __future__ import annotations
from ansible.plugins.action import ActionBase
from ansible.module_utils.six import string_types

class ActionModule(ActionBase):
    """ Create inventory groups based on variables """
    TRANSFERS_FILES = False
    _VALID_ARGS = frozenset(('key', 'parents'))
    _requires_connection = False

    def run(self, tmp=None, task_vars=None):
        if False:
            return 10
        if task_vars is None:
            task_vars = dict()
        result = super(ActionModule, self).run(tmp, task_vars)
        del tmp
        if 'key' not in self._task.args:
            result['failed'] = True
            result['msg'] = "the 'key' param is required when using group_by"
            return result
        group_name = self._task.args.get('key')
        parent_groups = self._task.args.get('parents', ['all'])
        if isinstance(parent_groups, string_types):
            parent_groups = [parent_groups]
        result['changed'] = False
        result['add_group'] = group_name.replace(' ', '-')
        result['parent_groups'] = [name.replace(' ', '-') for name in parent_groups]
        return result