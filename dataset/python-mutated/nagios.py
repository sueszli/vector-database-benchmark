"""
Run nagios plugins/checks from salt and get the return as data.
"""
import logging
import os
import stat
log = logging.getLogger(__name__)
PLUGINDIR = '/usr/lib/nagios/plugins/'

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load if nagios-plugins are installed\n    '
    if os.path.isdir(PLUGINDIR):
        return 'nagios'
    return (False, 'The nagios execution module cannot be loaded: nagios-plugins are not installed.')

def _execute_cmd(plugin, args='', run_type='cmd.retcode'):
    if False:
        i = 10
        return i + 15
    "\n    Execute nagios plugin if it's in the directory with salt command specified in run_type\n    "
    data = {}
    all_plugins = list_plugins()
    if plugin in all_plugins:
        data = __salt__[run_type]('{}{} {}'.format(PLUGINDIR, plugin, args), python_shell=False)
    return data

def _execute_pillar(pillar_name, run_type):
    if False:
        print('Hello World!')
    '\n    Run one or more nagios plugins from pillar data and get the result of run_type\n    The pillar have to be in this format:\n    ------\n    webserver:\n        Ping_google:\n            - check_icmp: 8.8.8.8\n            - check_icmp: google.com\n        Load:\n            - check_load: -w 0.8 -c 1\n        APT:\n            - check_apt\n    -------\n    '
    groups = __salt__['pillar.get'](pillar_name)
    data = {}
    for group in groups:
        data[group] = {}
        commands = groups[group]
        for command in commands:
            if isinstance(command, dict):
                plugin = next(iter(command.keys()))
                args = command[plugin]
            else:
                plugin = command
                args = ''
            command_key = _format_dict_key(args, plugin)
            data[group][command_key] = run_type(plugin, args)
    return data

def _format_dict_key(args, plugin):
    if False:
        while True:
            i = 10
    key_name = plugin
    args_key = args.replace(' ', '')
    if args != '':
        args_key = '_' + args_key
        key_name = plugin + args_key
    return key_name

def run(plugin, args=''):
    if False:
        print('Hello World!')
    "\n    Run nagios plugin and return all the data execution with cmd.run\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' nagios.run check_apt\n        salt '*' nagios.run check_icmp '8.8.8.8'\n    "
    data = _execute_cmd(plugin, args, 'cmd.run')
    return data

def retcode(plugin, args='', key_name=None):
    if False:
        while True:
            i = 10
    '\n    Run one nagios plugin and return retcode of the execution\n    '
    data = {}
    if key_name is None:
        key_name = _format_dict_key(args, plugin)
    data[key_name] = {}
    status = _execute_cmd(plugin, args, 'cmd.retcode')
    data[key_name]['status'] = status
    return data

def run_all(plugin, args=''):
    if False:
        while True:
            i = 10
    '\n    Run nagios plugin and return all the data execution with cmd.run_all\n    '
    data = _execute_cmd(plugin, args, 'cmd.run_all')
    return data

def retcode_pillar(pillar_name):
    if False:
        while True:
            i = 10
    "\n    Run one or more nagios plugins from pillar data and get the result of cmd.retcode\n    The pillar have to be in this format::\n\n        ------\n        webserver:\n            Ping_google:\n                - check_icmp: 8.8.8.8\n                - check_icmp: google.com\n            Load:\n                - check_load: -w 0.8 -c 1\n            APT:\n                - check_apt\n        -------\n\n    webserver is the role to check, the next keys are the group and the items\n    the check with the arguments if needed\n\n    You must to group different checks(one o more) and always it will return\n    the highest value of all the checks\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' nagios.retcode webserver\n    "
    groups = __salt__['pillar.get'](pillar_name)
    check = {}
    data = {}
    for group in groups:
        commands = groups[group]
        for command in commands:
            if isinstance(command, dict):
                plugin = next(iter(command.keys()))
                args = command[plugin]
            else:
                plugin = command
                args = ''
            check.update(retcode(plugin, args, group))
            current_value = 0
            new_value = int(check[group]['status'])
            if group in data:
                current_value = int(data[group]['status'])
            if new_value > current_value or group not in data:
                if group not in data:
                    data[group] = {}
                data[group]['status'] = new_value
    return data

def run_pillar(pillar_name):
    if False:
        i = 10
        return i + 15
    "\n    Run one or more nagios plugins from pillar data and get the result of cmd.run\n    The pillar have to be in this format::\n\n        ------\n        webserver:\n            Ping_google:\n                - check_icmp: 8.8.8.8\n                - check_icmp: google.com\n            Load:\n                - check_load: -w 0.8 -c 1\n            APT:\n                - check_apt\n        -------\n\n    webserver is the role to check, the next keys are the group and the items\n    the check with the arguments if needed\n\n    You have to group different checks in a group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' nagios.run webserver\n    "
    data = _execute_pillar(pillar_name, run)
    return data

def run_all_pillar(pillar_name):
    if False:
        i = 10
        return i + 15
    "\n    Run one or more nagios plugins from pillar data and get the result of cmd.run_all\n    The pillar have to be in this format::\n\n        ------\n        webserver:\n            Ping_google:\n                - check_icmp: 8.8.8.8\n                - check_icmp: google.com\n            Load:\n                - check_load: -w 0.8 -c 1\n            APT:\n                - check_apt\n        -------\n\n    webserver is the role to check, the next keys are the group and the items\n    the check with the arguments if needed\n\n    You have to group different checks in a group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' nagios.run webserver\n    "
    data = _execute_pillar(pillar_name, run_all)
    return data

def list_plugins():
    if False:
        print('Hello World!')
    "\n    List all the nagios plugins\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' nagios.list_plugins\n    "
    plugin_list = os.listdir(PLUGINDIR)
    ret = []
    for plugin in plugin_list:
        stat_f = os.path.join(PLUGINDIR, plugin)
        execute_bit = stat.S_IXUSR & os.stat(stat_f)[stat.ST_MODE]
        if execute_bit:
            ret.append(plugin)
    return ret