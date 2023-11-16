from __future__ import annotations
import os
from ansible.errors import AnsibleError, AnsibleAction, AnsibleActionFail, AnsibleActionSkip
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.plugins.action import ActionBase

class ActionModule(ActionBase):
    TRANSFERS_FILES = True

    def run(self, tmp=None, task_vars=None):
        if False:
            print('Hello World!')
        ' handler for unarchive operations '
        if task_vars is None:
            task_vars = dict()
        result = super(ActionModule, self).run(tmp, task_vars)
        del tmp
        source = self._task.args.get('src', None)
        dest = self._task.args.get('dest', None)
        remote_src = boolean(self._task.args.get('remote_src', False), strict=False)
        creates = self._task.args.get('creates', None)
        decrypt = self._task.args.get('decrypt', True)
        try:
            if 'copy' in self._task.args:
                if 'remote_src' in self._task.args:
                    raise AnsibleActionFail("parameters are mutually exclusive: ('copy', 'remote_src')")
                self._task.args['remote_src'] = remote_src = not boolean(self._task.args.pop('copy'), strict=False)
            if source is None or dest is None:
                raise AnsibleActionFail('src (or content) and dest are required')
            if creates:
                creates = self._remote_expand_user(creates)
                if self._remote_file_exists(creates):
                    raise AnsibleActionSkip('skipped, since %s exists' % creates)
            dest = self._remote_expand_user(dest)
            source = os.path.expanduser(source)
            if not remote_src:
                try:
                    source = self._loader.get_real_file(self._find_needle('files', source), decrypt=decrypt)
                except AnsibleError as e:
                    raise AnsibleActionFail(to_text(e))
            try:
                remote_stat = self._execute_remote_stat(dest, all_vars=task_vars, follow=True)
            except AnsibleError as e:
                raise AnsibleActionFail(to_text(e))
            if not remote_stat['exists'] or not remote_stat['isdir']:
                raise AnsibleActionFail("dest '%s' must be an existing dir" % dest)
            if not remote_src:
                tmp_src = self._connection._shell.join_path(self._connection._shell.tmpdir, 'source')
                self._transfer_file(source, tmp_src)
            new_module_args = self._task.args.copy()
            for key in ('decrypt',):
                if key in new_module_args:
                    del new_module_args[key]
            if not remote_src:
                self._fixup_perms2((self._connection._shell.tmpdir, tmp_src))
                new_module_args['src'] = tmp_src
            result.update(self._execute_module(module_name='ansible.legacy.unarchive', module_args=new_module_args, task_vars=task_vars))
        except AnsibleAction as e:
            result.update(e.result)
        finally:
            self._remove_tmp_path(self._connection._shell.tmpdir)
        return result