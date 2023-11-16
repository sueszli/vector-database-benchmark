from __future__ import annotations
from ansible.errors import AnsibleError
from ansible.playbook.conditional import Conditional
from ansible.plugins.action import ActionBase
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import boolean

class ActionModule(ActionBase):
    """ Fail with custom message """
    _requires_connection = False
    _VALID_ARGS = frozenset(('fail_msg', 'msg', 'quiet', 'success_msg', 'that'))

    def run(self, tmp=None, task_vars=None):
        if False:
            print('Hello World!')
        if task_vars is None:
            task_vars = dict()
        result = super(ActionModule, self).run(tmp, task_vars)
        del tmp
        if 'that' not in self._task.args:
            raise AnsibleError('conditional required in "that" string')
        fail_msg = None
        success_msg = None
        fail_msg = self._task.args.get('fail_msg', self._task.args.get('msg'))
        if fail_msg is None:
            fail_msg = 'Assertion failed'
        elif isinstance(fail_msg, list):
            if not all((isinstance(x, string_types) for x in fail_msg)):
                raise AnsibleError('Type of one of the elements in fail_msg or msg list is not string type')
        elif not isinstance(fail_msg, (string_types, list)):
            raise AnsibleError('Incorrect type for fail_msg or msg, expected a string or list and got %s' % type(fail_msg))
        success_msg = self._task.args.get('success_msg')
        if success_msg is None:
            success_msg = 'All assertions passed'
        elif isinstance(success_msg, list):
            if not all((isinstance(x, string_types) for x in success_msg)):
                raise AnsibleError('Type of one of the elements in success_msg list is not string type')
        elif not isinstance(success_msg, (string_types, list)):
            raise AnsibleError('Incorrect type for success_msg, expected a string or list and got %s' % type(success_msg))
        quiet = boolean(self._task.args.get('quiet', False), strict=False)
        thats = self._task.args['that']
        if not isinstance(thats, list):
            thats = [thats]
        cond = Conditional(loader=self._loader)
        if not quiet:
            result['_ansible_verbose_always'] = True
        for that in thats:
            cond.when = [that]
            test_result = cond.evaluate_conditional(templar=self._templar, all_vars=task_vars)
            if not test_result:
                result['failed'] = True
                result['evaluated_to'] = test_result
                result['assertion'] = that
                result['msg'] = fail_msg
                return result
        result['changed'] = False
        result['msg'] = success_msg
        return result