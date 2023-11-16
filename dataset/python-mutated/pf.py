"""
Control the OpenBSD packet filter (PF).

:codeauthor: Jasper Lievisse Adriaanse <j@jasper.la>

.. versionadded:: 2019.2.0
"""
import logging
import re
import salt.utils.path
from salt.exceptions import CommandExecutionError, SaltInvocationError
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only works on OpenBSD and FreeBSD for now; other systems with pf (macOS,\n    FreeBSD, etc) need to be tested before enabling them.\n    '
    tested_oses = ['FreeBSD', 'OpenBSD']
    if __grains__['os'] in tested_oses and salt.utils.path.which('pfctl'):
        return True
    return (False, 'The pf execution module cannot be loaded: either the OS ({}) is not tested or the pfctl binary was not found'.format(__grains__['os']))

def enable():
    if False:
        return 10
    "\n    Enable the Packet Filter.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pf.enable\n    "
    ret = {}
    result = __salt__['cmd.run_all']('pfctl -e', output_loglevel='trace', python_shell=False)
    if result['retcode'] == 0:
        ret = {'comment': 'pf enabled', 'changes': True}
    elif result['stderr'] == 'pfctl: pf already enabled':
        ret = {'comment': 'pf already enabled', 'changes': False}
    else:
        raise CommandExecutionError('Could not enable pf', info={'errors': [result['stderr']], 'changes': False})
    return ret

def disable():
    if False:
        while True:
            i = 10
    "\n    Disable the Packet Filter.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pf.disable\n    "
    ret = {}
    result = __salt__['cmd.run_all']('pfctl -d', output_loglevel='trace', python_shell=False)
    if result['retcode'] == 0:
        ret = {'comment': 'pf disabled', 'changes': True}
    elif result['stderr'] == 'pfctl: pf not enabled':
        ret = {'comment': 'pf already disabled', 'changes': False}
    else:
        raise CommandExecutionError('Could not disable pf', info={'errors': [result['stderr']], 'changes': False})
    return ret

def loglevel(level):
    if False:
        while True:
            i = 10
    "\n    Set the debug level which limits the severity of log messages printed by ``pf(4)``.\n\n    level:\n        Log level. Should be one of the following: emerg, alert, crit, err, warning, notice,\n        info or debug (OpenBSD); or none, urgent, misc, loud (FreeBSD).\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pf.loglevel emerg\n    "
    ret = {'changes': True}
    myos = __grains__['os']
    if myos == 'FreeBSD':
        all_levels = ['none', 'urgent', 'misc', 'loud']
    else:
        all_levels = ['emerg', 'alert', 'crit', 'err', 'warning', 'notice', 'info', 'debug']
    if level not in all_levels:
        raise SaltInvocationError('Unknown loglevel: {}'.format(level))
    result = __salt__['cmd.run_all']('pfctl -x {}'.format(level), output_loglevel='trace', python_shell=False)
    if result['retcode'] != 0:
        raise CommandExecutionError('Problem encountered setting loglevel', info={'errors': [result['stderr']], 'changes': False})
    return ret

def load(file='/etc/pf.conf', noop=False):
    if False:
        i = 10
        return i + 15
    "\n    Load a ruleset from the specific file, overwriting the currently loaded ruleset.\n\n    file:\n        Full path to the file containing the ruleset.\n\n    noop:\n        Don't actually load the rules, just parse them.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pf.load /etc/pf.conf.d/lockdown.conf\n    "
    ret = {'changes': True}
    cmd = ['pfctl', '-f', file]
    if noop:
        ret['changes'] = False
        cmd.append('-n')
    result = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
    if result['retcode'] != 0:
        raise CommandExecutionError('Problem loading the ruleset from {}'.format(file), info={'errors': [result['stderr']], 'changes': False})
    return ret

def flush(modifier):
    if False:
        while True:
            i = 10
    "\n    Flush the specified packet filter parameters.\n\n    modifier:\n        Should be one of the following:\n\n        - all\n        - info\n        - osfp\n        - rules\n        - sources\n        - states\n        - tables\n\n        Please refer to the OpenBSD `pfctl(8) <https://man.openbsd.org/pfctl#T>`_\n        documentation for a detailed explanation of each command.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pf.flush states\n    "
    ret = {}
    all_modifiers = ['rules', 'states', 'info', 'osfp', 'all', 'sources', 'tables']
    capital_modifiers = ['Sources', 'Tables']
    all_modifiers += capital_modifiers
    if modifier.title() in capital_modifiers:
        modifier = modifier.title()
    if modifier not in all_modifiers:
        raise SaltInvocationError('Unknown modifier: {}'.format(modifier))
    cmd = 'pfctl -v -F {}'.format(modifier)
    result = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
    if result['retcode'] == 0:
        if re.match('^0.*', result['stderr']):
            ret['changes'] = False
        else:
            ret['changes'] = True
        ret['comment'] = result['stderr']
    else:
        raise CommandExecutionError('Could not flush {}'.format(modifier), info={'errors': [result['stderr']], 'changes': False})
    return ret

def table(command, table, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Apply a command on the specified table.\n\n    table:\n        Name of the table.\n\n    command:\n        Command to apply to the table. Supported commands are:\n\n        - add\n        - delete\n        - expire\n        - flush\n        - kill\n        - replace\n        - show\n        - test\n        - zero\n\n        Please refer to the OpenBSD `pfctl(8) <https://man.openbsd.org/pfctl#T>`_\n        documentation for a detailed explanation of each command.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pf.table expire table=spam_hosts number=300\n        salt \'*\' pf.table add table=local_hosts addresses=\'["127.0.0.1", "::1"]\'\n    '
    ret = {}
    all_commands = ['kill', 'flush', 'add', 'delete', 'expire', 'replace', 'show', 'test', 'zero']
    if command not in all_commands:
        raise SaltInvocationError('Unknown table command: {}'.format(command))
    cmd = ['pfctl', '-t', table, '-T', command]
    if command in ['add', 'delete', 'replace', 'test']:
        cmd += kwargs.get('addresses', [])
    elif command == 'expire':
        number = kwargs.get('number', None)
        if not number:
            raise SaltInvocationError('need expire_number argument for expire command')
        else:
            cmd.append(number)
    result = __salt__['cmd.run_all'](cmd, output_level='trace', python_shell=False)
    if result['retcode'] == 0:
        if command == 'show':
            ret = {'comment': result['stdout'].split()}
        elif command == 'test':
            ret = {'comment': result['stderr'], 'matches': True}
        else:
            if re.match('^(0.*|no changes)', result['stderr']):
                ret['changes'] = False
            else:
                ret['changes'] = True
            ret['comment'] = result['stderr']
    elif command == 'test' and re.match('^\\d+/\\d+ addresses match.$', result['stderr']):
        ret = {'comment': result['stderr'], 'matches': False}
    else:
        raise CommandExecutionError('Could not apply {} on table {}'.format(command, table), info={'errors': [result['stderr']], 'changes': False})
    return ret

def show(modifier):
    if False:
        while True:
            i = 10
    "\n    Show filter parameters.\n\n    modifier:\n        Modifier to apply for filtering. Only a useful subset of what pfctl supports\n        can be used with Salt.\n\n        - rules\n        - states\n        - tables\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pf.show rules\n    "
    ret = {'changes': False}
    capital_modifiers = ['Tables']
    all_modifiers = ['rules', 'states', 'tables']
    all_modifiers += capital_modifiers
    if modifier.title() in capital_modifiers:
        modifier = modifier.title()
    if modifier not in all_modifiers:
        raise SaltInvocationError('Unknown modifier: {}'.format(modifier))
    cmd = 'pfctl -s {}'.format(modifier)
    result = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
    if result['retcode'] == 0:
        ret['comment'] = result['stdout'].split('\n')
    else:
        raise CommandExecutionError('Could not show {}'.format(modifier), info={'errors': [result['stderr']], 'changes': False})
    return ret