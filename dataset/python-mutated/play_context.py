from __future__ import annotations
from ansible import constants as C
from ansible import context
from ansible.playbook.attribute import FieldAttribute
from ansible.playbook.base import Base
from ansible.utils.display import Display
display = Display()
__all__ = ['PlayContext']
TASK_ATTRIBUTE_OVERRIDES = ('become', 'become_user', 'become_pass', 'become_method', 'become_flags', 'connection', 'docker_extra_args', 'delegate_to', 'no_log', 'remote_user')
RESET_VARS = ('ansible_connection', 'ansible_user', 'ansible_host', 'ansible_port', 'ansible_docker_extra_args', 'ansible_ssh_host', 'ansible_ssh_pass', 'ansible_ssh_port', 'ansible_ssh_user', 'ansible_ssh_private_key_file', 'ansible_ssh_pipelining', 'ansible_ssh_executable')

class PlayContext(Base):
    """
    This class is used to consolidate the connection information for
    hosts in a play and child tasks, where the task may override some
    connection/authentication information.
    """
    module_compression = FieldAttribute(isa='string', default=C.DEFAULT_MODULE_COMPRESSION)
    shell = FieldAttribute(isa='string')
    executable = FieldAttribute(isa='string', default=C.DEFAULT_EXECUTABLE)
    remote_addr = FieldAttribute(isa='string')
    password = FieldAttribute(isa='string')
    timeout = FieldAttribute(isa='int', default=C.DEFAULT_TIMEOUT)
    connection_user = FieldAttribute(isa='string')
    private_key_file = FieldAttribute(isa='string', default=C.DEFAULT_PRIVATE_KEY_FILE)
    pipelining = FieldAttribute(isa='bool', default=C.ANSIBLE_PIPELINING)
    network_os = FieldAttribute(isa='string')
    docker_extra_args = FieldAttribute(isa='string')
    connection_lockfd = FieldAttribute(isa='int')
    become = FieldAttribute(isa='bool')
    become_method = FieldAttribute(isa='string')
    become_user = FieldAttribute(isa='string')
    become_pass = FieldAttribute(isa='string')
    become_exe = FieldAttribute(isa='string', default=C.DEFAULT_BECOME_EXE)
    become_flags = FieldAttribute(isa='string', default=C.DEFAULT_BECOME_FLAGS)
    prompt = FieldAttribute(isa='string')
    only_tags = FieldAttribute(isa='set', default=set)
    skip_tags = FieldAttribute(isa='set', default=set)
    start_at_task = FieldAttribute(isa='string')
    step = FieldAttribute(isa='bool', default=False)
    force_handlers = FieldAttribute(isa='bool', default=False)

    @property
    def verbosity(self):
        if False:
            print('Hello World!')
        display.deprecated('PlayContext.verbosity is deprecated, use ansible.utils.display.Display.verbosity instead.', version='2.18')
        return self._internal_verbosity

    @verbosity.setter
    def verbosity(self, value):
        if False:
            print('Hello World!')
        display.deprecated('PlayContext.verbosity is deprecated, use ansible.utils.display.Display.verbosity instead.', version='2.18')
        self._internal_verbosity = value

    def __init__(self, play=None, passwords=None, connection_lockfd=None):
        if False:
            while True:
                i = 10
        super(PlayContext, self).__init__()
        if passwords is None:
            passwords = {}
        self.password = passwords.get('conn_pass', '')
        self.become_pass = passwords.get('become_pass', '')
        self._become_plugin = None
        self.prompt = ''
        self.success_key = ''
        self.connection_lockfd = connection_lockfd
        if context.CLIARGS:
            self.set_attributes_from_cli()
        else:
            self._internal_verbosity = 0
        if play:
            self.set_attributes_from_play(play)

    def set_attributes_from_plugin(self, plugin):
        if False:
            for i in range(10):
                print('nop')
        options = C.config.get_configuration_definitions(plugin.plugin_type, plugin._load_name)
        for option in options:
            if option:
                flag = options[option].get('name')
                if flag:
                    setattr(self, flag, plugin.get_option(flag))

    def set_attributes_from_play(self, play):
        if False:
            for i in range(10):
                print('nop')
        self.force_handlers = play.force_handlers

    def set_attributes_from_cli(self):
        if False:
            return 10
        '\n        Configures this connection information instance with data from\n        options specified by the user on the command line. These have a\n        lower precedence than those set on the play or host.\n        '
        if context.CLIARGS.get('timeout', False):
            self.timeout = int(context.CLIARGS['timeout'])
        self.private_key_file = context.CLIARGS.get('private_key_file')
        self._internal_verbosity = context.CLIARGS.get('verbosity')
        self.start_at_task = context.CLIARGS.get('start_at_task', None)

    def set_task_and_variable_override(self, task, variables, templar):
        if False:
            return 10
        '\n        Sets attributes from the task if they are set, which will override\n        those from the play.\n\n        :arg task: the task object with the parameters that were set on it\n        :arg variables: variables from inventory\n        :arg templar: templar instance if templating variables is needed\n        '
        new_info = self.copy()
        for attr in TASK_ATTRIBUTE_OVERRIDES:
            if (attr_val := getattr(task, attr, None)) is not None:
                setattr(new_info, attr, attr_val)
        if task.delegate_to is not None:
            delegated_host_name = templar.template(task.delegate_to)
            delegated_vars = variables.get('ansible_delegated_vars', dict()).get(delegated_host_name, dict())
            delegated_transport = C.DEFAULT_TRANSPORT
            for transport_var in C.MAGIC_VARIABLE_MAPPING.get('connection'):
                if transport_var in delegated_vars:
                    delegated_transport = delegated_vars[transport_var]
                    break
            for address_var in ('ansible_%s_host' % delegated_transport,) + C.MAGIC_VARIABLE_MAPPING.get('remote_addr'):
                if address_var in delegated_vars:
                    break
            else:
                display.debug('no remote address found for delegated host %s\nusing its name, so success depends on DNS resolution' % delegated_host_name)
                delegated_vars['ansible_host'] = delegated_host_name
            for port_var in ('ansible_%s_port' % delegated_transport,) + C.MAGIC_VARIABLE_MAPPING.get('port'):
                if port_var in delegated_vars:
                    break
            else:
                if delegated_transport == 'winrm':
                    delegated_vars['ansible_port'] = 5986
                else:
                    delegated_vars['ansible_port'] = C.DEFAULT_REMOTE_PORT
            for user_var in ('ansible_%s_user' % delegated_transport,) + C.MAGIC_VARIABLE_MAPPING.get('remote_user'):
                if user_var in delegated_vars and delegated_vars[user_var]:
                    break
            else:
                delegated_vars['ansible_user'] = task.remote_user or self.remote_user
        else:
            delegated_vars = dict()
            for exe_var in C.MAGIC_VARIABLE_MAPPING.get('executable'):
                if exe_var in variables:
                    setattr(new_info, 'executable', variables.get(exe_var))
        attrs_considered = []
        for (attr, variable_names) in C.MAGIC_VARIABLE_MAPPING.items():
            for variable_name in variable_names:
                if attr in attrs_considered:
                    continue
                if task.delegate_to is not None:
                    if isinstance(delegated_vars, dict) and variable_name in delegated_vars:
                        setattr(new_info, attr, delegated_vars[variable_name])
                        attrs_considered.append(attr)
                elif variable_name in variables:
                    setattr(new_info, attr, variables[variable_name])
                    attrs_considered.append(attr)
        for become_pass_name in C.MAGIC_VARIABLE_MAPPING.get('become_pass'):
            if become_pass_name in variables:
                break
        if new_info.port is None and C.DEFAULT_REMOTE_PORT is not None:
            new_info.port = int(C.DEFAULT_REMOTE_PORT)
        if len(delegated_vars) > 0:
            for connection_type in C.MAGIC_VARIABLE_MAPPING.get('connection'):
                if connection_type in delegated_vars:
                    break
            else:
                remote_addr_local = new_info.remote_addr in C.LOCALHOST
                inv_hostname_local = delegated_vars.get('inventory_hostname') in C.LOCALHOST
                if remote_addr_local and inv_hostname_local:
                    setattr(new_info, 'connection', 'local')
                elif getattr(new_info, 'connection', None) == 'local' and (not remote_addr_local or not inv_hostname_local):
                    setattr(new_info, 'connection', C.DEFAULT_TRANSPORT)
        if new_info.connection == 'local':
            if not new_info.connection_user:
                new_info.connection_user = new_info.remote_user
        if new_info.remote_addr == 'inventory_hostname':
            new_info.remote_addr = variables.get('inventory_hostname')
            display.warning('The "%s" connection plugin has an improperly configured remote target value, forcing "inventory_hostname" templated value instead of the string' % new_info.connection)
        if new_info.no_log is None:
            new_info.no_log = C.DEFAULT_NO_LOG
        if task.check_mode is not None:
            new_info.check_mode = task.check_mode
        if task.diff is not None:
            new_info.diff = task.diff
        return new_info

    def set_become_plugin(self, plugin):
        if False:
            for i in range(10):
                print('nop')
        self._become_plugin = plugin

    def update_vars(self, variables):
        if False:
            return 10
        "\n        Adds 'magic' variables relating to connections to the variable dictionary provided.\n        In case users need to access from the play, this is a legacy from runner.\n        "
        for (prop, var_list) in C.MAGIC_VARIABLE_MAPPING.items():
            try:
                if 'become' in prop:
                    continue
                var_val = getattr(self, prop)
                for var_opt in var_list:
                    if var_opt not in variables and var_val is not None:
                        variables[var_opt] = var_val
            except AttributeError:
                continue