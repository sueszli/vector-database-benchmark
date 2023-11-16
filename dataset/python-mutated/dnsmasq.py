"""
Module for managing dnsmasq
"""
import logging
import os
import salt.utils.files
import salt.utils.platform
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only work on POSIX-like systems.\n    '
    if salt.utils.platform.is_windows():
        return (False, 'dnsmasq execution module cannot be loaded: only works on non-Windows systems.')
    return True

def version():
    if False:
        print('Hello World!')
    "\n    Shows installed version of dnsmasq.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dnsmasq.version\n    "
    cmd = 'dnsmasq -v'
    out = __salt__['cmd.run'](cmd).splitlines()
    comps = out[0].split()
    return comps[2]

def fullversion():
    if False:
        i = 10
        return i + 15
    "\n    Shows installed version of dnsmasq and compile options.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dnsmasq.fullversion\n    "
    cmd = 'dnsmasq -v'
    out = __salt__['cmd.run'](cmd).splitlines()
    comps = out[0].split()
    version_num = comps[2]
    comps = out[1].split()
    return {'version': version_num, 'compile options': comps[3:]}

def set_config(config_file='/etc/dnsmasq.conf', follow=True, **kwargs):
    if False:
        print('Hello World!')
    "\n    Sets a value or a set of values in the specified file. By default, if\n    conf-dir is configured in this file, salt will attempt to set the option\n    in any file inside the conf-dir where it has already been enabled. If it\n    does not find it inside any files, it will append it to the main config\n    file. Setting follow to False will turn off this behavior.\n\n    If a config option currently appears multiple times (such as dhcp-host,\n    which is specified at least once per host), the new option will be added\n    to the end of the main config file (and not to any includes). If you need\n    an option added to a specific include file, specify it as the config_file.\n\n    :param string config_file: config file where settings should be updated / added.\n    :param bool follow: attempt to set the config option inside any file within\n        the ``conf-dir`` where it has already been enabled.\n    :param kwargs: key value pairs that contain the configuration settings that you\n        want set.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' dnsmasq.set_config domain=mydomain.com\n        salt '*' dnsmasq.set_config follow=False domain=mydomain.com\n        salt '*' dnsmasq.set_config config_file=/etc/dnsmasq.conf domain=mydomain.com\n    "
    dnsopts = get_config(config_file)
    includes = [config_file]
    if follow is True and 'conf-dir' in dnsopts:
        for filename in os.listdir(dnsopts['conf-dir']):
            if filename.startswith('.'):
                continue
            if filename.endswith('~'):
                continue
            if filename.endswith('bak'):
                continue
            if filename.endswith('#') and filename.endswith('#'):
                continue
            includes.append('{}/{}'.format(dnsopts['conf-dir'], filename))
    ret_kwargs = {}
    for key in kwargs:
        if key.startswith('__'):
            continue
        ret_kwargs[key] = kwargs[key]
        if key in dnsopts:
            if isinstance(dnsopts[key], str):
                for config in includes:
                    __salt__['file.sed'](path=config, before='^{}=.*'.format(key), after='{}={}'.format(key, kwargs[key]))
            else:
                __salt__['file.append'](config_file, '{}={}'.format(key, kwargs[key]))
        else:
            __salt__['file.append'](config_file, '{}={}'.format(key, kwargs[key]))
    return ret_kwargs

def get_config(config_file='/etc/dnsmasq.conf'):
    if False:
        return 10
    "\n    Dumps all options from the config file.\n\n    config_file\n        The location of the config file from which to obtain contents.\n        Defaults to ``/etc/dnsmasq.conf``.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' dnsmasq.get_config\n        salt '*' dnsmasq.get_config config_file=/etc/dnsmasq.conf\n    "
    dnsopts = _parse_dnamasq(config_file)
    if 'conf-dir' in dnsopts:
        for filename in os.listdir(dnsopts['conf-dir']):
            if filename.startswith('.'):
                continue
            if filename.endswith('~'):
                continue
            if filename.endswith('#') and filename.endswith('#'):
                continue
            dnsopts.update(_parse_dnamasq('{}/{}'.format(dnsopts['conf-dir'], filename)))
    return dnsopts

def _parse_dnamasq(filename):
    if False:
        return 10
    '\n    Generic function for parsing dnsmasq files including includes.\n    '
    fileopts = {}
    if not os.path.isfile(filename):
        raise CommandExecutionError("Error: No such file '{}'".format(filename))
    with salt.utils.files.fopen(filename, 'r') as fp_:
        for line in fp_:
            line = salt.utils.stringutils.to_unicode(line)
            if not line.strip():
                continue
            if line.startswith('#'):
                continue
            if '=' in line:
                comps = line.split('=')
                if comps[0] in fileopts:
                    if isinstance(fileopts[comps[0]], str):
                        temp = fileopts[comps[0]]
                        fileopts[comps[0]] = [temp]
                    fileopts[comps[0]].append(comps[1].strip())
                else:
                    fileopts[comps[0]] = comps[1].strip()
            else:
                if 'unparsed' not in fileopts:
                    fileopts['unparsed'] = []
                fileopts['unparsed'].append(line)
    return fileopts