"""
Module for managing Solaris logadm based log rotations.
"""
import logging
import shlex
import salt.utils.args
import salt.utils.decorators as decorators
import salt.utils.files
import salt.utils.stringutils
log = logging.getLogger(__name__)
default_conf = '/etc/logadm.conf'
option_toggles = {'-c': 'copy', '-l': 'localtime', '-N': 'skip_missing'}
option_flags = {'-A': 'age', '-C': 'count', '-a': 'post_command', '-b': 'pre_command', '-e': 'mail_addr', '-E': 'expire_command', '-g': 'group', '-m': 'mode', '-M': 'rename_command', '-o': 'owner', '-p': 'period', '-P': 'timestmp', '-R': 'old_created_command', '-s': 'size', '-S': 'max_size', '-t': 'template', '-T': 'old_pattern', '-w': 'entryname', '-z': 'compress_count'}

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only work on Solaris based systems\n    '
    if 'Solaris' in __grains__['os_family']:
        return True
    return (False, 'The logadm execution module cannot be loaded: only available on Solaris.')

def _arg2opt(arg):
    if False:
        print('Hello World!')
    '\n    Turn a pass argument into the correct option\n    '
    res = [o for (o, a) in option_toggles.items() if a == arg]
    res += [o for (o, a) in option_flags.items() if a == arg]
    return res[0] if res else None

def _parse_conf(conf_file=default_conf):
    if False:
        print('Hello World!')
    '\n    Parse a logadm configuration file.\n    '
    ret = {}
    with salt.utils.files.fopen(conf_file, 'r') as ifile:
        for line in ifile:
            line = salt.utils.stringutils.to_unicode(line).strip()
            if not line:
                continue
            if line.startswith('#'):
                continue
            splitline = line.split(' ', 1)
            ret[splitline[0]] = splitline[1]
    return ret

def _parse_options(entry, options, include_unset=True):
    if False:
        while True:
            i = 10
    '\n    Parse a logadm options string\n    '
    log_cfg = {}
    options = shlex.split(options)
    if not options:
        return None
    if entry.startswith('/'):
        log_cfg['log_file'] = entry
    else:
        log_cfg['entryname'] = entry
    index = 0
    while index < len(options):
        if index in [0, len(options) - 1] and options[index].startswith('/'):
            log_cfg['log_file'] = options[index]
        elif options[index] in option_toggles:
            log_cfg[option_toggles[options[index]]] = True
        elif options[index] in option_flags and index + 1 <= len(options):
            log_cfg[option_flags[options[index]]] = int(options[index + 1]) if options[index + 1].isdigit() else options[index + 1]
            index += 1
        else:
            if 'additional_options' not in log_cfg:
                log_cfg['additional_options'] = []
            if ' ' in options[index]:
                log_cfg['dditional_options'] = "'{}'".format(options[index])
            else:
                log_cfg['additional_options'].append(options[index])
        index += 1
    if 'additional_options' in log_cfg:
        log_cfg['additional_options'] = ' '.join(log_cfg['additional_options'])
    if 'log_file' not in log_cfg and 'entryname' in log_cfg:
        log_cfg['log_file'] = log_cfg['entryname']
        del log_cfg['entryname']
    if include_unset:
        for name in option_toggles.values():
            if name not in log_cfg:
                log_cfg[name] = False
        for name in option_flags.values():
            if name not in log_cfg:
                log_cfg[name] = None
    return log_cfg

def show_conf(conf_file=default_conf, name=None):
    if False:
        print('Hello World!')
    "\n    Show configuration\n\n    conf_file : string\n        path to logadm.conf, defaults to /etc/logadm.conf\n    name : string\n        optional show only a single entry\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' logadm.show_conf\n        salt '*' logadm.show_conf name=/var/log/syslog\n    "
    cfg = _parse_conf(conf_file)
    if name and name in cfg:
        return {name: cfg[name]}
    elif name:
        return {name: 'not found in {}'.format(conf_file)}
    else:
        return cfg

def list_conf(conf_file=default_conf, log_file=None, include_unset=False):
    if False:
        print('Hello World!')
    "\n    Show parsed configuration\n\n    .. versionadded:: 2018.3.0\n\n    conf_file : string\n        path to logadm.conf, defaults to /etc/logadm.conf\n    log_file : string\n        optional show only one log file\n    include_unset : boolean\n        include unset flags in output\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' logadm.list_conf\n        salt '*' logadm.list_conf log=/var/log/syslog\n        salt '*' logadm.list_conf include_unset=False\n    "
    cfg = _parse_conf(conf_file)
    cfg_parsed = {}
    for entry in cfg:
        log_cfg = _parse_options(entry, cfg[entry], include_unset)
        cfg_parsed[log_cfg['log_file'] if 'log_file' in log_cfg else log_cfg['entryname']] = log_cfg
    if log_file and log_file in cfg_parsed:
        return {log_file: cfg_parsed[log_file]}
    elif log_file:
        return {log_file: 'not found in {}'.format(conf_file)}
    else:
        return cfg_parsed

@decorators.memoize
def show_args():
    if False:
        while True:
            i = 10
    "\n    Show which arguments map to which flags and options.\n\n    .. versionadded:: 2018.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' logadm.show_args\n    "
    mapping = {'flags': {}, 'options': {}}
    for (flag, arg) in option_toggles.items():
        mapping['flags'][flag] = arg
    for (option, arg) in option_flags.items():
        mapping['options'][option] = arg
    return mapping

def rotate(name, pattern=None, conf_file=default_conf, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Set up pattern for logging.\n\n    name : string\n        alias for entryname\n    pattern : string\n        alias for log_file\n    conf_file : string\n        optional path to alternative configuration file\n    kwargs : boolean|string|int\n        optional additional flags and parameters\n\n    .. note::\n        ``name`` and ``pattern`` were kept for backwards compatibility reasons.\n\n        ``name`` is an alias for the ``entryname`` argument, ``pattern`` is an alias\n        for ``log_file``. These aliases will only be used if the ``entryname`` and\n        ``log_file`` arguments are not passed.\n\n        For a full list of arguments see ```logadm.show_args```.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' logadm.rotate myapplog pattern='/var/log/myapp/*.log' count=7\n        salt '*' logadm.rotate myapplog log_file='/var/log/myapp/*.log' count=4 owner=myappd mode='0700'\n\n    "
    kwargs = salt.utils.args.clean_kwargs(**kwargs)
    if 'entryname' not in kwargs and name and (not name.startswith('/')):
        kwargs['entryname'] = name
    if 'log_file' not in kwargs:
        if pattern and pattern.startswith('/'):
            kwargs['log_file'] = pattern
        elif name and name.startswith('/'):
            kwargs['log_file'] = name
    log.debug('logadm.rotate - kwargs: %s', kwargs)
    command = 'logadm -f {}'.format(conf_file)
    for (arg, val) in kwargs.items():
        if arg in option_toggles.values() and val:
            command = '{} {}'.format(command, _arg2opt(arg))
        elif arg in option_flags.values():
            command = '{} {} {}'.format(command, _arg2opt(arg), shlex.quote(str(val)))
        elif arg != 'log_file':
            log.warning("Unknown argument %s, don't know how to map this!", arg)
    if 'log_file' in kwargs:
        if 'entryname' not in kwargs:
            command = '{} -w {}'.format(command, shlex.quote(kwargs['log_file']))
        else:
            command = '{} {}'.format(command, shlex.quote(kwargs['log_file']))
    log.debug('logadm.rotate - command: %s', command)
    result = __salt__['cmd.run_all'](command, python_shell=False)
    if result['retcode'] != 0:
        return dict(Error='Failed in adding log', Output=result['stderr'])
    return dict(Result='Success')

def remove(name, conf_file=default_conf):
    if False:
        for i in range(10):
            print('nop')
    "\n    Remove log pattern from logadm\n\n    CLI Example:\n\n    .. code-block:: bash\n\n      salt '*' logadm.remove myapplog\n    "
    command = 'logadm -f {} -r {}'.format(conf_file, name)
    result = __salt__['cmd.run_all'](command, python_shell=False)
    if result['retcode'] != 0:
        return dict(Error='Failure in removing log. Possibly already removed?', Output=result['stderr'])
    return dict(Result='Success')