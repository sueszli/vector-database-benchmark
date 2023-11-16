from __future__ import annotations
import ansible.constants as C
from ansible.errors import AnsibleParserError
from ansible.playbook.block import Block
from ansible.playbook.task import Task
from ansible.utils.display import Display
from ansible.utils.sentinel import Sentinel
__all__ = ['TaskInclude']
display = Display()

class TaskInclude(Task):
    """
    A task include is derived from a regular task to handle the special
    circumstances related to the `- include_*: ...` task.
    """
    BASE = frozenset(('file', '_raw_params'))
    OTHER_ARGS = frozenset(('apply',))
    VALID_ARGS = BASE.union(OTHER_ARGS)
    VALID_INCLUDE_KEYWORDS = frozenset(('action', 'args', 'collections', 'debugger', 'ignore_errors', 'loop', 'loop_control', 'loop_with', 'name', 'no_log', 'register', 'run_once', 'tags', 'timeout', 'vars', 'when'))

    def __init__(self, block=None, role=None, task_include=None):
        if False:
            i = 10
            return i + 15
        super(TaskInclude, self).__init__(block=block, role=role, task_include=task_include)
        self.statically_loaded = False

    @staticmethod
    def load(data, block=None, role=None, task_include=None, variable_manager=None, loader=None):
        if False:
            return 10
        ti = TaskInclude(block=block, role=role, task_include=task_include)
        task = ti.check_options(ti.load_data(data, variable_manager=variable_manager, loader=loader), data)
        return task

    def check_options(self, task, data):
        if False:
            i = 10
            return i + 15
        '\n        Method for options validation to use in \'load_data\' for TaskInclude and HandlerTaskInclude\n        since they share the same validations. It is not named \'validate_options\' on purpose\n        to prevent confusion with \'_validate_*" methods. Note that the task passed might be changed\n        as a side-effect of this method.\n        '
        my_arg_names = frozenset(task.args.keys())
        bad_opts = my_arg_names.difference(self.VALID_ARGS)
        if bad_opts and task.action in C._ACTION_ALL_PROPER_INCLUDE_IMPORT_TASKS:
            raise AnsibleParserError('Invalid options for %s: %s' % (task.action, ','.join(list(bad_opts))), obj=data)
        if not task.args.get('_raw_params'):
            task.args['_raw_params'] = task.args.pop('file', None)
            if not task.args['_raw_params']:
                raise AnsibleParserError('No file specified for %s' % task.action)
        apply_attrs = task.args.get('apply', {})
        if apply_attrs and task.action not in C._ACTION_INCLUDE_TASKS:
            raise AnsibleParserError('Invalid options for %s: apply' % task.action, obj=data)
        elif not isinstance(apply_attrs, dict):
            raise AnsibleParserError('Expected a dict for apply but got %s instead' % type(apply_attrs), obj=data)
        return task

    def preprocess_data(self, ds):
        if False:
            return 10
        ds = super(TaskInclude, self).preprocess_data(ds)
        diff = set(ds.keys()).difference(self.VALID_INCLUDE_KEYWORDS)
        for k in diff:
            if ds[k] is not Sentinel and ds['action'] in C._ACTION_ALL_INCLUDE_ROLE_TASKS:
                if C.INVALID_TASK_ATTRIBUTE_FAILED:
                    raise AnsibleParserError("'%s' is not a valid attribute for a %s" % (k, self.__class__.__name__), obj=ds)
                else:
                    display.warning('Ignoring invalid attribute: %s' % k)
        return ds

    def copy(self, exclude_parent=False, exclude_tasks=False):
        if False:
            i = 10
            return i + 15
        new_me = super(TaskInclude, self).copy(exclude_parent=exclude_parent, exclude_tasks=exclude_tasks)
        new_me.statically_loaded = self.statically_loaded
        return new_me

    def build_parent_block(self):
        if False:
            return 10
        '\n        This method is used to create the parent block for the included tasks\n        when ``apply`` is specified\n        '
        apply_attrs = self.args.pop('apply', {})
        if apply_attrs:
            apply_attrs['block'] = []
            p_block = Block.load(apply_attrs, play=self._parent._play, task_include=self, role=self._role, variable_manager=self._variable_manager, loader=self._loader)
        else:
            p_block = self
        return p_block