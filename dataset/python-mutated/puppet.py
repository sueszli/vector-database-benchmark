"""
Execute puppet routines
"""
import datetime
import logging
import os
import salt.utils.args
import salt.utils.files
import salt.utils.path
import salt.utils.platform
import salt.utils.stringutils
import salt.utils.yaml
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        return 10
    '\n    Only load if puppet is installed\n    '
    unavailable_exes = ', '.join((exe for exe in ('facter', 'puppet') if salt.utils.path.which(exe) is None))
    if unavailable_exes:
        return (False, 'The puppet execution module cannot be loaded: {} unavailable.'.format(unavailable_exes))
    else:
        return 'puppet'

def _format_fact(output):
    if False:
        return 10
    try:
        (fact, value) = output.split(' => ', 1)
        value = value.strip()
    except ValueError:
        fact = None
        value = None
    return (fact, value)

class _Puppet:
    """
    Puppet helper class. Used to format command for execution.
    """

    def __init__(self):
        if False:
            return 10
        "\n        Setup a puppet instance, based on the premis that default usage is to\n        run 'puppet agent --test'. Configuration and run states are stored in\n        the default locations.\n        "
        self.subcmd = 'agent'
        self.subcmd_args = []
        self.kwargs = {'color': 'false'}
        self.args = []
        puppet_config = __salt__['cmd.run']('puppet config print --render-as yaml vardir rundir confdir')
        conf = salt.utils.yaml.safe_load(puppet_config)
        self.vardir = conf['vardir']
        self.rundir = conf['rundir']
        self.confdir = conf['confdir']
        self.disabled_lockfile = self.vardir + '/state/agent_disabled.lock'
        self.run_lockfile = self.vardir + '/state/agent_catalog_run.lock'
        self.agent_pidfile = self.rundir + '/agent.pid'
        self.lastrunfile = self.vardir + '/state/last_run_summary.yaml'

    def __repr__(self):
        if False:
            while True:
                i = 10
        '\n        Format the command string to executed using cmd.run_all.\n        '
        cmd = 'puppet {subcmd} --vardir {vardir} --confdir {confdir}'.format(**self.__dict__)
        args = ' '.join(self.subcmd_args)
        args += ''.join([' --{}'.format(k) for k in self.args])
        args += ''.join([' --{} {}'.format(k, v) for (k, v) in self.kwargs.items()])
        if salt.utils.platform.is_windows():
            return 'cmd /V:ON /c {} {} ^& if !ERRORLEVEL! EQU 2 (EXIT 0) ELSE (EXIT /B)'.format(cmd, args)
        return '({} {}) || test $? -eq 2'.format(cmd, args)

    def arguments(self, args=None):
        if False:
            print('Hello World!')
        "\n        Read in arguments for the current subcommand. These are added to the\n        cmd line without '--' appended. Any others are redirected as standard\n        options with the double hyphen prefixed.\n        "
        args = args and list(args) or []
        if self.subcmd == 'apply':
            self.subcmd_args = [args[0]]
            del args[0]
        if self.subcmd == 'agent':
            args.extend(['test'])
        self.args = args

def run(*args, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Execute a puppet run and return a dict with the stderr, stdout,\n    return code, etc. The first positional argument given is checked as a\n    subcommand. Following positional arguments should be ordered with arguments\n    required by the subcommand first, followed by non-keyword arguments.\n    Tags are specified by a tag keyword and comma separated list of values. --\n    http://docs.puppetlabs.com/puppet/latest/reference/lang_tags.html\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' puppet.run\n        salt '*' puppet.run tags=basefiles::edit,apache::server\n        salt '*' puppet.run agent onetime no-daemonize no-usecacheonfailure no-splay ignorecache\n        salt '*' puppet.run debug\n        salt '*' puppet.run apply /a/b/manifest.pp modulepath=/a/b/modules tags=basefiles::edit,apache::server\n    "
    puppet = _Puppet()
    buildargs = ()
    for arg in args:
        if arg in ['agent', 'apply']:
            puppet.subcmd = arg
        else:
            buildargs += (arg,)
    puppet.arguments(buildargs)
    puppet.kwargs.update(salt.utils.args.clean_kwargs(**kwargs))
    ret = __salt__['cmd.run_all'](repr(puppet), python_shell=True)
    return ret

def noop(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Execute a puppet noop run and return a dict with the stderr, stdout,\n    return code, etc. Usage is the same as for puppet.run.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' puppet.noop\n        salt '*' puppet.noop tags=basefiles::edit,apache::server\n        salt '*' puppet.noop debug\n        salt '*' puppet.noop apply /a/b/manifest.pp modulepath=/a/b/modules tags=basefiles::edit,apache::server\n    "
    args += ('noop',)
    return run(*args, **kwargs)

def enable():
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2014.7.0\n\n    Enable the puppet agent\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' puppet.enable\n    "
    puppet = _Puppet()
    if os.path.isfile(puppet.disabled_lockfile):
        try:
            os.remove(puppet.disabled_lockfile)
        except OSError as exc:
            msg = 'Failed to enable: {}'.format(exc)
            log.error(msg)
            raise CommandExecutionError(msg)
        else:
            return True
    return False

def disable(message=None):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2014.7.0\n\n    Disable the puppet agent\n\n    message\n        .. versionadded:: 2015.5.2\n\n        Disable message to send to puppet\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' puppet.disable\n        salt '*' puppet.disable 'Disabled, contact XYZ before enabling'\n    "
    puppet = _Puppet()
    if os.path.isfile(puppet.disabled_lockfile):
        return False
    else:
        with salt.utils.files.fopen(puppet.disabled_lockfile, 'w') as lockfile:
            try:
                msg = '{{"disabled_message":"{0}"}}'.format(message) if message is not None else '{}'
                lockfile.write(salt.utils.stringutils.to_str(msg))
                lockfile.close()
                return True
            except OSError as exc:
                msg = 'Failed to disable: {}'.format(exc)
                log.error(msg)
                raise CommandExecutionError(msg)

def status():
    if False:
        return 10
    "\n    .. versionadded:: 2014.7.0\n\n    Display puppet agent status\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' puppet.status\n    "
    puppet = _Puppet()
    if os.path.isfile(puppet.disabled_lockfile):
        return 'Administratively disabled'
    if os.path.isfile(puppet.run_lockfile):
        try:
            with salt.utils.files.fopen(puppet.run_lockfile, 'r') as fp_:
                pid = int(salt.utils.stringutils.to_unicode(fp_.read()))
                os.kill(pid, 0)
        except (OSError, ValueError):
            return 'Stale lockfile'
        else:
            return 'Applying a catalog'
    if os.path.isfile(puppet.agent_pidfile):
        try:
            with salt.utils.files.fopen(puppet.agent_pidfile, 'r') as fp_:
                pid = int(salt.utils.stringutils.to_unicode(fp_.read()))
                os.kill(pid, 0)
        except (OSError, ValueError):
            return 'Stale pidfile'
        else:
            return 'Idle daemon'
    return 'Stopped'

def summary():
    if False:
        return 10
    "\n    .. versionadded:: 2014.7.0\n\n    Show a summary of the last puppet agent run\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' puppet.summary\n    "
    puppet = _Puppet()
    try:
        with salt.utils.files.fopen(puppet.lastrunfile, 'r') as fp_:
            report = salt.utils.yaml.safe_load(fp_)
        result = {}
        if 'time' in report:
            try:
                result['last_run'] = datetime.datetime.fromtimestamp(int(report['time']['last_run'])).isoformat()
            except (TypeError, ValueError, KeyError):
                result['last_run'] = 'invalid or missing timestamp'
            result['time'] = {}
            for key in ('total', 'config_retrieval'):
                if key in report['time']:
                    result['time'][key] = report['time'][key]
        if 'resources' in report:
            result['resources'] = report['resources']
    except salt.utils.yaml.YAMLError as exc:
        raise CommandExecutionError('YAML error parsing puppet run summary: {}'.format(exc))
    except OSError as exc:
        raise CommandExecutionError('Unable to read puppet run summary: {}'.format(exc))
    return result

def plugin_sync():
    if False:
        return 10
    "\n    Runs a plugin sync between the puppet master and agent\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' puppet.plugin_sync\n    "
    ret = __salt__['cmd.run']('puppet plugin download')
    if not ret:
        return ''
    return ret

def facts(puppet=False):
    if False:
        while True:
            i = 10
    "\n    Run facter and return the results\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' puppet.facts\n    "
    ret = {}
    opt_puppet = '--puppet' if puppet else ''
    cmd_ret = __salt__['cmd.run_all']('facter {}'.format(opt_puppet))
    if cmd_ret['retcode'] != 0:
        raise CommandExecutionError(cmd_ret['stderr'])
    output = cmd_ret['stdout']
    for line in output.splitlines():
        if not line:
            continue
        (fact, value) = _format_fact(line)
        if not fact:
            continue
        ret[fact] = value
    return ret

def fact(name, puppet=False):
    if False:
        print('Hello World!')
    "\n    Run facter for a specific fact\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' puppet.fact kernel\n    "
    opt_puppet = '--puppet' if puppet else ''
    ret = __salt__['cmd.run_all']('facter {} {}'.format(opt_puppet, name), python_shell=False)
    if ret['retcode'] != 0:
        raise CommandExecutionError(ret['stderr'])
    if not ret['stdout']:
        return ''
    return ret['stdout']