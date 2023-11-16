from __future__ import annotations
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.plugins.action import ActionBase
from ansible.utils.vars import isidentifier

class ActionModule(ActionBase):
    TRANSFERS_FILES = False
    _VALID_ARGS = frozenset(('aggregate', 'data', 'per_host'))
    _requires_connection = False

    def run(self, tmp=None, task_vars=None):
        if False:
            i = 10
            return i + 15
        if task_vars is None:
            task_vars = dict()
        result = super(ActionModule, self).run(tmp, task_vars)
        del tmp
        stats = {'data': {}, 'per_host': False, 'aggregate': True}
        if self._task.args:
            data = self._task.args.get('data', {})
            if not isinstance(data, dict):
                data = self._templar.template(data, convert_bare=False, fail_on_undefined=True)
            if not isinstance(data, dict):
                result['failed'] = True
                result['msg'] = "The 'data' option needs to be a dictionary/hash"
                return result
            for opt in ['per_host', 'aggregate']:
                val = self._task.args.get(opt, None)
                if val is not None:
                    if not isinstance(val, bool):
                        stats[opt] = boolean(self._templar.template(val), strict=False)
                    else:
                        stats[opt] = val
            for (k, v) in data.items():
                k = self._templar.template(k)
                if not isidentifier(k):
                    result['failed'] = True
                    result['msg'] = "The variable name '%s' is not valid. Variables must start with a letter or underscore character, and contain only letters, numbers and underscores." % k
                    return result
                stats['data'][k] = self._templar.template(v)
        result['changed'] = False
        result['ansible_stats'] = stats
        return result