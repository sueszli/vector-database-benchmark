from __future__ import annotations
import os
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.executor.task_executor import remove_omit
from ansible.module_utils.common.text.converters import to_text
from ansible.playbook.handler import Handler
from ansible.playbook.task_include import TaskInclude
from ansible.playbook.role_include import IncludeRole
from ansible.template import Templar
from ansible.utils.display import Display
display = Display()

class IncludedFile:

    def __init__(self, filename, args, vars, task, is_role=False):
        if False:
            print('Hello World!')
        self._filename = filename
        self._args = args
        self._vars = vars
        self._task = task
        self._hosts = []
        self._is_role = is_role
        self._results = []

    def add_host(self, host):
        if False:
            return 10
        if host not in self._hosts:
            self._hosts.append(host)
            return
        raise ValueError()

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return other._filename == self._filename and other._args == self._args and (other._vars == self._vars) and (other._task._uuid == self._task._uuid) and (other._task._parent._uuid == self._task._parent._uuid)

    def __repr__(self):
        if False:
            return 10
        return '%s (args=%s vars=%s): %s' % (self._filename, self._args, self._vars, self._hosts)

    @staticmethod
    def process_include_results(results, iterator, loader, variable_manager):
        if False:
            print('Hello World!')
        included_files = []
        task_vars_cache = {}
        for res in results:
            original_host = res._host
            original_task = res._task
            if original_task.action in C._ACTION_ALL_INCLUDES:
                if original_task.loop:
                    if 'results' not in res._result:
                        continue
                    include_results = res._result['results']
                else:
                    include_results = [res._result]
                for include_result in include_results:
                    if 'skipped' in include_result and include_result['skipped'] or ('failed' in include_result and include_result['failed']):
                        continue
                    cache_key = (iterator._play, original_host, original_task)
                    try:
                        task_vars = task_vars_cache[cache_key]
                    except KeyError:
                        task_vars = task_vars_cache[cache_key] = variable_manager.get_vars(play=iterator._play, host=original_host, task=original_task)
                    include_args = include_result.get('include_args', dict())
                    special_vars = {}
                    loop_var = include_result.get('ansible_loop_var', 'item')
                    index_var = include_result.get('ansible_index_var')
                    if loop_var in include_result:
                        task_vars[loop_var] = special_vars[loop_var] = include_result[loop_var]
                    if index_var and index_var in include_result:
                        task_vars[index_var] = special_vars[index_var] = include_result[index_var]
                    if '_ansible_item_label' in include_result:
                        task_vars['_ansible_item_label'] = special_vars['_ansible_item_label'] = include_result['_ansible_item_label']
                    if 'ansible_loop' in include_result:
                        task_vars['ansible_loop'] = special_vars['ansible_loop'] = include_result['ansible_loop']
                    if original_task.no_log and '_ansible_no_log' not in include_args:
                        task_vars['_ansible_no_log'] = special_vars['_ansible_no_log'] = original_task.no_log
                    task_vars['ansible_search_path'] = original_task.get_search_path()
                    if loader.get_basedir() not in task_vars['ansible_search_path']:
                        task_vars['ansible_search_path'].append(loader.get_basedir())
                    templar = Templar(loader=loader, variables=task_vars)
                    if original_task.action in C._ACTION_INCLUDE_TASKS:
                        include_file = None
                        if original_task._parent:
                            parent_include = original_task._parent
                            cumulative_path = None
                            while parent_include is not None:
                                if not isinstance(parent_include, TaskInclude):
                                    parent_include = parent_include._parent
                                    continue
                                if isinstance(parent_include, IncludeRole):
                                    parent_include_dir = parent_include._role_path
                                else:
                                    try:
                                        parent_include_dir = os.path.dirname(templar.template(parent_include.args.get('_raw_params')))
                                    except AnsibleError as e:
                                        parent_include_dir = ''
                                        display.warning('Templating the path of the parent %s failed. The path to the included file may not be found. The error was: %s.' % (original_task.action, to_text(e)))
                                if cumulative_path is not None and (not os.path.isabs(cumulative_path)):
                                    cumulative_path = os.path.join(parent_include_dir, cumulative_path)
                                else:
                                    cumulative_path = parent_include_dir
                                include_target = templar.template(include_result['include'])
                                if original_task._role:
                                    dirname = 'handlers' if isinstance(original_task, Handler) else 'tasks'
                                    new_basedir = os.path.join(original_task._role._role_path, dirname, cumulative_path)
                                    candidates = [loader.path_dwim_relative(original_task._role._role_path, dirname, include_target), loader.path_dwim_relative(new_basedir, dirname, include_target)]
                                    for include_file in candidates:
                                        try:
                                            os.stat(include_file)
                                            break
                                        except OSError:
                                            pass
                                else:
                                    include_file = loader.path_dwim_relative(loader.get_basedir(), cumulative_path, include_target)
                                if os.path.exists(include_file):
                                    break
                                else:
                                    parent_include = parent_include._parent
                        if include_file is None:
                            if original_task._role:
                                include_target = templar.template(include_result['include'])
                                include_file = loader.path_dwim_relative(original_task._role._role_path, 'handlers' if isinstance(original_task, Handler) else 'tasks', include_target, is_role=True)
                            else:
                                include_file = loader.path_dwim(include_result['include'])
                        include_file = templar.template(include_file)
                        inc_file = IncludedFile(include_file, include_args, special_vars, original_task)
                    else:
                        role_name = include_args.pop('name', include_args.pop('role', None))
                        if role_name is not None:
                            role_name = templar.template(role_name)
                        new_task = original_task.copy()
                        new_task.post_validate(templar=templar)
                        new_task._role_name = role_name
                        for from_arg in new_task.FROM_ARGS:
                            if from_arg in include_args:
                                from_key = from_arg.removesuffix('_from')
                                new_task._from_files[from_key] = templar.template(include_args.pop(from_arg))
                        omit_token = task_vars.get('omit')
                        if omit_token:
                            new_task._from_files = remove_omit(new_task._from_files, omit_token)
                        inc_file = IncludedFile(role_name, include_args, special_vars, new_task, is_role=True)
                    idx = 0
                    orig_inc_file = inc_file
                    while 1:
                        try:
                            pos = included_files[idx:].index(orig_inc_file)
                            inc_file = included_files[idx + pos]
                        except ValueError:
                            included_files.append(orig_inc_file)
                            inc_file = orig_inc_file
                        try:
                            inc_file.add_host(original_host)
                            inc_file._results.append(res)
                        except ValueError:
                            idx += pos + 1
                        else:
                            break
        return included_files