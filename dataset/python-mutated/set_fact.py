from __future__ import annotations
from ansible.errors import AnsibleActionFail
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.plugins.action import ActionBase
from ansible.utils.vars import isidentifier
import ansible.constants as C

class ActionModule(ActionBase):
    TRANSFERS_FILES = False
    _requires_connection = False

    def run(self, tmp=None, task_vars=None):
        if False:
            while True:
                i = 10
        if task_vars is None:
            task_vars = dict()
        result = super(ActionModule, self).run(tmp, task_vars)
        del tmp
        facts = {}
        cacheable = boolean(self._task.args.pop('cacheable', False))
        if self._task.args:
            for (k, v) in self._task.args.items():
                k = self._templar.template(k)
                if not isidentifier(k):
                    raise AnsibleActionFail("The variable name '%s' is not valid. Variables must start with a letter or underscore character, and contain only letters, numbers and underscores." % k)
                if not C.DEFAULT_JINJA2_NATIVE and isinstance(v, string_types) and (v.lower() in ('true', 'false', 'yes', 'no')):
                    v = boolean(v, strict=False)
                facts[k] = v
        else:
            raise AnsibleActionFail('No key/value pairs provided, at least one is required for this action to succeed')
        if facts:
            result['ansible_facts'] = facts
            result['_ansible_facts_cacheable'] = cacheable
        else:
            raise AnsibleActionFail('Unable to create any variables with provided arguments')
        return result