from __future__ import annotations
from ansible.plugins.action import ActionBase
from ansible.utils.vars import merge_hash

class ActionModule(ActionBase):

    def _get_async_dir(self):
        if False:
            for i in range(10):
                print('nop')
        async_dir = self.get_shell_option('async_dir', default='~/.ansible_async')
        return self._remote_expand_user(async_dir)

    def run(self, tmp=None, task_vars=None):
        if False:
            i = 10
            return i + 15
        results = super(ActionModule, self).run(tmp, task_vars)
        (validation_result, new_module_args) = self.validate_argument_spec(argument_spec={'jid': {'type': 'str', 'required': True}, 'mode': {'type': 'str', 'choices': ['status', 'cleanup'], 'default': 'status'}})
        results['started'] = results['finished'] = 0
        results['stdout'] = results['stderr'] = ''
        results['stdout_lines'] = results['stderr_lines'] = []
        jid = new_module_args['jid']
        mode = new_module_args['mode']
        results['ansible_job_id'] = jid
        async_dir = self._get_async_dir()
        log_path = self._connection._shell.join_path(async_dir, jid)
        if mode == 'cleanup':
            results['erased'] = log_path
        else:
            results['results_file'] = log_path
            results['started'] = 1
        new_module_args['_async_dir'] = async_dir
        results = merge_hash(results, self._execute_module(module_name='ansible.legacy.async_status', task_vars=task_vars, module_args=new_module_args))
        return results