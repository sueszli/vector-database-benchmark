from __future__ import annotations
from os.path import basename
import ansible.constants as C
from ansible.errors import AnsibleParserError
from ansible.playbook.attribute import NonInheritableFieldAttribute
from ansible.playbook.task_include import TaskInclude
from ansible.playbook.role import Role
from ansible.playbook.role.include import RoleInclude
from ansible.utils.display import Display
from ansible.module_utils.six import string_types
from ansible.template import Templar
__all__ = ['IncludeRole']
display = Display()

class IncludeRole(TaskInclude):
    """
    A Role include is derived from a regular role to handle the special
    circumstances related to the `- include_role: ...`
    """
    BASE = frozenset(('name', 'role'))
    FROM_ARGS = frozenset(('tasks_from', 'vars_from', 'defaults_from', 'handlers_from'))
    OTHER_ARGS = frozenset(('apply', 'public', 'allow_duplicates', 'rolespec_validate'))
    VALID_ARGS = BASE | FROM_ARGS | OTHER_ARGS
    public = NonInheritableFieldAttribute(isa='bool', default=None, private=False, always_post_validate=True)
    allow_duplicates = NonInheritableFieldAttribute(isa='bool', default=True, private=True, always_post_validate=True)
    rolespec_validate = NonInheritableFieldAttribute(isa='bool', default=True, private=True, always_post_validate=True)

    def __init__(self, block=None, role=None, task_include=None):
        if False:
            return 10
        super(IncludeRole, self).__init__(block=block, role=role, task_include=task_include)
        self._from_files = {}
        self._parent_role = role
        self._role_name = None
        self._role_path = None

    def get_name(self):
        if False:
            print('Hello World!')
        ' return the name of the task '
        return self.name or '%s : %s' % (self.action, self._role_name)

    def get_block_list(self, play=None, variable_manager=None, loader=None):
        if False:
            for i in range(10):
                print('nop')
        if play is None:
            myplay = self._parent._play
        else:
            myplay = play
        ri = RoleInclude.load(self._role_name, play=myplay, variable_manager=variable_manager, loader=loader, collection_list=self.collections)
        ri.vars |= self.vars
        if variable_manager is not None:
            available_variables = variable_manager.get_vars(play=myplay, task=self)
        else:
            available_variables = {}
        templar = Templar(loader=loader, variables=available_variables)
        from_files = templar.template(self._from_files)
        actual_role = Role.load(ri, myplay, parent_role=self._parent_role, from_files=from_files, from_include=True, validate=self.rolespec_validate, public=self.public)
        actual_role._metadata.allow_duplicates = self.allow_duplicates
        if self.statically_loaded or self.public:
            myplay.roles.append(actual_role)
        self._role_path = actual_role._role_path
        dep_chain = actual_role.get_dep_chain()
        p_block = self.build_parent_block()
        p_block.collections = actual_role.collections
        blocks = actual_role.compile(play=myplay, dep_chain=dep_chain)
        for b in blocks:
            b._parent = p_block
            b.collections = actual_role.collections
        handlers = actual_role.get_handler_blocks(play=myplay, dep_chain=dep_chain)
        for h in handlers:
            h._parent = p_block
        myplay.handlers = myplay.handlers + handlers
        return (blocks, handlers)

    @staticmethod
    def load(data, block=None, role=None, task_include=None, variable_manager=None, loader=None):
        if False:
            while True:
                i = 10
        ir = IncludeRole(block, role, task_include=task_include).load_data(data, variable_manager=variable_manager, loader=loader)
        if ir.action in C._ACTION_INCLUDE_ROLE:
            ir.static = False
        my_arg_names = frozenset(ir.args.keys())
        ir._role_name = ir.args.get('name', ir.args.get('role'))
        if ir._role_name is None:
            raise AnsibleParserError("'name' is a required field for %s." % ir.action, obj=data)
        bad_opts = my_arg_names.difference(IncludeRole.VALID_ARGS)
        if bad_opts:
            raise AnsibleParserError('Invalid options for %s: %s' % (ir.action, ','.join(list(bad_opts))), obj=data)
        for key in my_arg_names.intersection(IncludeRole.FROM_ARGS):
            from_key = key.removesuffix('_from')
            args_value = ir.args.get(key)
            if not isinstance(args_value, string_types):
                raise AnsibleParserError('Expected a string for %s but got %s instead' % (key, type(args_value)))
            ir._from_files[from_key] = basename(args_value)
        apply_attrs = ir.args.get('apply', {})
        if apply_attrs and ir.action not in C._ACTION_INCLUDE_ROLE:
            raise AnsibleParserError('Invalid options for %s: apply' % ir.action, obj=data)
        elif not isinstance(apply_attrs, dict):
            raise AnsibleParserError('Expected a dict for apply but got %s instead' % type(apply_attrs), obj=data)
        for option in my_arg_names.intersection(IncludeRole.OTHER_ARGS):
            setattr(ir, option, ir.args.get(option))
        return ir

    def copy(self, exclude_parent=False, exclude_tasks=False):
        if False:
            print('Hello World!')
        new_me = super(IncludeRole, self).copy(exclude_parent=exclude_parent, exclude_tasks=exclude_tasks)
        new_me.statically_loaded = self.statically_loaded
        new_me._from_files = self._from_files.copy()
        new_me._parent_role = self._parent_role
        new_me._role_name = self._role_name
        new_me._role_path = self._role_path
        return new_me

    def get_include_params(self):
        if False:
            while True:
                i = 10
        v = super(IncludeRole, self).get_include_params()
        if self._parent_role:
            v |= self._parent_role.get_role_params()
            v.setdefault('ansible_parent_role_names', []).insert(0, self._parent_role.get_name())
            v.setdefault('ansible_parent_role_paths', []).insert(0, self._parent_role._role_path)
        return v