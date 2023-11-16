from __future__ import annotations
import ansible.constants as C
from ansible.errors import AnsibleParserError, AnsibleError, AnsibleAssertionError
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text
from ansible.parsing.splitter import parse_kv, split_args
from ansible.plugins.loader import module_loader, action_loader
from ansible.template import Templar
from ansible.utils.fqcn import add_internal_fqcns
from ansible.utils.sentinel import Sentinel
FREEFORM_ACTIONS = frozenset(C.MODULE_REQUIRE_ARGS)
RAW_PARAM_MODULES = FREEFORM_ACTIONS.union(add_internal_fqcns(('include', 'include_vars', 'include_tasks', 'include_role', 'import_tasks', 'import_role', 'add_host', 'group_by', 'set_fact', 'meta')))
BUILTIN_TASKS = frozenset(add_internal_fqcns(('meta', 'include', 'include_tasks', 'include_role', 'import_tasks', 'import_role')))

class ModuleArgsParser:
    """
    There are several ways a module and argument set can be expressed:

    # legacy form (for a shell command)
    - action: shell echo hi

    # common shorthand for local actions vs delegate_to
    - local_action: shell echo hi

    # most commonly:
    - copy: src=a dest=b

    # legacy form
    - action: copy src=a dest=b

    # complex args form, for passing structured data
    - copy:
        src: a
        dest: b

    # gross, but technically legal
    - action:
        module: copy
        args:
          src: a
          dest: b

    # Standard YAML form for command-type modules. In this case, the args specified
    # will act as 'defaults' and will be overridden by any args specified
    # in one of the other formats (complex args under the action, or
    # parsed from the k=v string
    - command: 'pwd'
      args:
        chdir: '/tmp'


    This class has some of the logic to canonicalize these into the form

    - module: <module_name>
      delegate_to: <optional>
      args: <args>

    Args may also be munged for certain shell command parameters.
    """

    def __init__(self, task_ds=None, collection_list=None):
        if False:
            i = 10
            return i + 15
        task_ds = {} if task_ds is None else task_ds
        if not isinstance(task_ds, dict):
            raise AnsibleAssertionError("the type of 'task_ds' should be a dict, but is a %s" % type(task_ds))
        self._task_ds = task_ds
        self._collection_list = collection_list
        from ansible.playbook.task import Task
        from ansible.playbook.handler import Handler
        self._task_attrs = set(Task.fattributes)
        self._task_attrs.update(set(Handler.fattributes))
        self._task_attrs.update(['local_action', 'static'])
        self._task_attrs = frozenset(self._task_attrs)
        self.resolved_action = None

    def _split_module_string(self, module_string):
        if False:
            while True:
                i = 10
        '\n        when module names are expressed like:\n        action: copy src=a dest=b\n        the first part of the string is the name of the module\n        and the rest are strings pertaining to the arguments.\n        '
        tokens = split_args(module_string)
        if len(tokens) > 1:
            return (tokens[0].strip(), ' '.join(tokens[1:]))
        else:
            return (tokens[0].strip(), '')

    def _normalize_parameters(self, thing, action=None, additional_args=None):
        if False:
            return 10
        '\n        arguments can be fuzzy.  Deal with all the forms.\n        '
        additional_args = {} if additional_args is None else additional_args
        final_args = dict()
        if additional_args:
            if isinstance(additional_args, string_types):
                templar = Templar(loader=None)
                if templar.is_template(additional_args):
                    final_args['_variable_params'] = additional_args
                else:
                    raise AnsibleParserError("Complex args containing variables cannot use bare variables (without Jinja2 delimiters), and must use the full variable style ('{{var_name}}')")
            elif isinstance(additional_args, dict):
                final_args.update(additional_args)
            else:
                raise AnsibleParserError('Complex args must be a dictionary or variable string ("{{var}}").')
        if action is not None:
            args = self._normalize_new_style_args(thing, action)
        else:
            (action, args) = self._normalize_old_style_args(thing)
            if args and 'args' in args:
                tmp_args = args.pop('args')
                if isinstance(tmp_args, string_types):
                    tmp_args = parse_kv(tmp_args)
                args.update(tmp_args)
        if args and action not in FREEFORM_ACTIONS:
            for arg in args:
                arg = to_text(arg)
                if arg.startswith('_ansible_'):
                    raise AnsibleError("invalid parameter specified for action '%s': '%s'" % (action, arg))
        if args:
            final_args.update(args)
        return (action, final_args)

    def _normalize_new_style_args(self, thing, action):
        if False:
            for i in range(10):
                print('nop')
        "\n        deals with fuzziness in new style module invocations\n        accepting key=value pairs and dictionaries, and returns\n        a dictionary of arguments\n\n        possible example inputs:\n            'echo hi', 'shell'\n            {'region': 'xyz'}, 'ec2'\n        standardized outputs like:\n            { _raw_params: 'echo hi', _uses_shell: True }\n        "
        if isinstance(thing, dict):
            args = thing
        elif isinstance(thing, string_types):
            check_raw = action in FREEFORM_ACTIONS
            args = parse_kv(thing, check_raw=check_raw)
        elif thing is None:
            args = None
        else:
            raise AnsibleParserError('unexpected parameter type in action: %s' % type(thing), obj=self._task_ds)
        return args

    def _normalize_old_style_args(self, thing):
        if False:
            return 10
        "\n        deals with fuzziness in old-style (action/local_action) module invocations\n        returns tuple of (module_name, dictionary_args)\n\n        possible example inputs:\n           { 'shell' : 'echo hi' }\n           'shell echo hi'\n           {'module': 'ec2', 'x': 1 }\n        standardized outputs like:\n           ('ec2', { 'x': 1} )\n        "
        action = None
        args = None
        if isinstance(thing, dict):
            thing = thing.copy()
            if 'module' in thing:
                (action, module_args) = self._split_module_string(thing['module'])
                args = thing.copy()
                check_raw = action in FREEFORM_ACTIONS
                args.update(parse_kv(module_args, check_raw=check_raw))
                del args['module']
        elif isinstance(thing, string_types):
            (action, args) = self._split_module_string(thing)
            check_raw = action in FREEFORM_ACTIONS
            args = parse_kv(args, check_raw=check_raw)
        else:
            raise AnsibleParserError('unexpected parameter type in action: %s' % type(thing), obj=self._task_ds)
        return (action, args)

    def parse(self, skip_action_validation=False):
        if False:
            return 10
        '\n        Given a task in one of the supported forms, parses and returns\n        returns the action, arguments, and delegate_to values for the\n        task, dealing with all sorts of levels of fuzziness.\n        '
        thing = None
        action = None
        delegate_to = self._task_ds.get('delegate_to', Sentinel)
        args = dict()
        additional_args = self._task_ds.get('args', dict())
        if 'action' in self._task_ds:
            thing = self._task_ds['action']
            (action, args) = self._normalize_parameters(thing, action=action, additional_args=additional_args)
        if 'local_action' in self._task_ds:
            if action is not None:
                raise AnsibleParserError('action and local_action are mutually exclusive', obj=self._task_ds)
            thing = self._task_ds.get('local_action', '')
            delegate_to = 'localhost'
            (action, args) = self._normalize_parameters(thing, action=action, additional_args=additional_args)
        non_task_ds = dict(((k, v) for (k, v) in self._task_ds.items() if k not in self._task_attrs and (not k.startswith('with_'))))
        for (item, value) in non_task_ds.items():
            context = None
            is_action_candidate = False
            if item in BUILTIN_TASKS:
                is_action_candidate = True
            elif skip_action_validation:
                is_action_candidate = True
            else:
                context = action_loader.find_plugin_with_context(item, collection_list=self._collection_list)
                if not context.resolved:
                    context = module_loader.find_plugin_with_context(item, collection_list=self._collection_list)
                is_action_candidate = context.resolved and bool(context.redirect_list)
            if is_action_candidate:
                if action is not None:
                    raise AnsibleParserError('conflicting action statements: %s, %s' % (action, item), obj=self._task_ds)
                if context is not None and context.resolved:
                    self.resolved_action = context.resolved_fqcn
                action = item
                thing = value
                (action, args) = self._normalize_parameters(thing, action=action, additional_args=additional_args)
        if action is None:
            if non_task_ds:
                bad_action = list(non_task_ds.keys())[0]
                raise AnsibleParserError("couldn't resolve module/action '{0}'. This often indicates a misspelling, missing collection, or incorrect module path.".format(bad_action), obj=self._task_ds)
            else:
                raise AnsibleParserError('no module/action detected in task.', obj=self._task_ds)
        elif args.get('_raw_params', '') != '' and action not in RAW_PARAM_MODULES:
            templar = Templar(loader=None)
            raw_params = args.pop('_raw_params')
            if templar.is_template(raw_params):
                args['_variable_params'] = raw_params
            else:
                raise AnsibleParserError("this task '%s' has extra params, which is only allowed in the following modules: %s" % (action, ', '.join(RAW_PARAM_MODULES)), obj=self._task_ds)
        return (action, args, delegate_to)