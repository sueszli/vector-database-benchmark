"""
Module for managing logrotate.
"""
import logging
import os
import salt.utils.files
import salt.utils.platform
import salt.utils.stringutils
from salt.exceptions import SaltInvocationError
_LOG = logging.getLogger(__name__)
_DEFAULT_CONF = '/etc/logrotate.conf'
__func_alias__ = {'set_': 'set'}

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only work on POSIX-like systems\n    '
    if salt.utils.platform.is_windows():
        return (False, 'The logrotate execution module cannot be loaded: only available on non-Windows systems.')
    return True

def _convert_if_int(value):
    if False:
        print('Hello World!')
    '\n    Convert to an int if necessary.\n\n    :param str value: The value to check/convert.\n\n    :return: The converted or passed value.\n    :rtype: bool|int|str\n    '
    try:
        value = int(str(value))
    except ValueError:
        pass
    return value

def _parse_conf(conf_file=_DEFAULT_CONF):
    if False:
        for i in range(10):
            print('nop')
    "\n    Parse a logrotate configuration file.\n\n    Includes will also be parsed, and their configuration will be stored in the\n    return dict, as if they were part of the main config file. A dict of which\n    configs came from which includes will be stored in the 'include files' dict\n    inside the return dict, for later reference by the user or module.\n    "
    ret = {}
    mode = 'single'
    multi_names = []
    multi = {}
    prev_comps = None
    with salt.utils.files.fopen(conf_file, 'r') as ifile:
        for line in ifile:
            line = salt.utils.stringutils.to_unicode(line).strip()
            if not line:
                continue
            if line.startswith('#'):
                continue
            comps = line.split()
            if '{' in line and '}' not in line:
                mode = 'multi'
                if len(comps) == 1 and prev_comps:
                    multi_names = prev_comps
                else:
                    multi_names = comps
                    multi_names.pop()
                continue
            if '}' in line:
                mode = 'single'
                for multi_name in multi_names:
                    ret[multi_name] = multi
                multi_names = []
                multi = {}
                continue
            if mode == 'single':
                key = ret
            else:
                key = multi
            if comps[0] == 'include':
                if 'include files' not in ret:
                    ret['include files'] = {}
                for include in os.listdir(comps[1]):
                    if include not in ret['include files']:
                        ret['include files'][include] = []
                    include_path = os.path.join(comps[1], include)
                    include_conf = _parse_conf(include_path)
                    for file_key in include_conf:
                        ret[file_key] = include_conf[file_key]
                        ret['include files'][include].append(file_key)
            prev_comps = comps
            if len(comps) > 2:
                key[comps[0]] = ' '.join(comps[1:])
            elif len(comps) > 1:
                key[comps[0]] = _convert_if_int(comps[1])
            else:
                key[comps[0]] = True
    return ret

def show_conf(conf_file=_DEFAULT_CONF):
    if False:
        return 10
    "\n    Show parsed configuration\n\n    :param str conf_file: The logrotate configuration file.\n\n    :return: The parsed configuration.\n    :rtype: dict\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' logrotate.show_conf\n    "
    return _parse_conf(conf_file)

def get(key, value=None, conf_file=_DEFAULT_CONF):
    if False:
        while True:
            i = 10
    "\n    Get the value for a specific configuration line.\n\n    :param str key: The command or stanza block to configure.\n    :param str value: The command value or command of the block specified by the key parameter.\n    :param str conf_file: The logrotate configuration file.\n\n    :return: The value for a specific configuration line.\n    :rtype: bool|int|str\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' logrotate.get rotate\n\n        salt '*' logrotate.get /var/log/wtmp rotate /etc/logrotate.conf\n    "
    current_conf = _parse_conf(conf_file)
    stanza = current_conf.get(key, False)
    if value:
        if stanza:
            return stanza.get(value, False)
        _LOG.debug("Block '%s' not present or empty.", key)
    return stanza

def set_(key, value, setting=None, conf_file=_DEFAULT_CONF):
    if False:
        for i in range(10):
            print('nop')
    "\n    Set a new value for a specific configuration line.\n\n    :param str key: The command or block to configure.\n    :param str value: The command value or command of the block specified by the key parameter.\n    :param str setting: The command value for the command specified by the value parameter.\n    :param str conf_file: The logrotate configuration file.\n\n    :return: A boolean representing whether all changes succeeded.\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' logrotate.set rotate 2\n\n    Can also be used to set a single value inside a multiline configuration\n    block. For instance, to change rotate in the following block:\n\n    .. code-block:: text\n\n        /var/log/wtmp {\n            monthly\n            create 0664 root root\n            rotate 1\n        }\n\n    Use the following command:\n\n    .. code-block:: bash\n\n        salt '*' logrotate.set /var/log/wtmp rotate 2\n\n    This module also has the ability to scan files inside an include directory,\n    and make changes in the appropriate file.\n    "
    conf = _parse_conf(conf_file)
    for include in conf['include files']:
        if key in conf['include files'][include]:
            conf_file = os.path.join(conf['include'], include)
    new_line = ''
    kwargs = {'flags': 8, 'backup': False, 'path': conf_file, 'pattern': '^{}.*'.format(key), 'show_changes': False}
    if setting is None:
        current_value = conf.get(key, False)
        if isinstance(current_value, dict):
            raise SaltInvocationError('Error: {} includes a dict, and a specific setting inside the dict was not declared'.format(key))
        if value == current_value:
            _LOG.debug("Command '%s' already has: %s", key, value)
            return True
        if value is True:
            new_line = key
        elif value:
            new_line = '{} {}'.format(key, value)
        kwargs.update({'prepend_if_not_found': True})
    else:
        stanza = conf.get(key, dict())
        if stanza and (not isinstance(stanza, dict)):
            error_msg = 'Error: A setting for a dict was declared, but the configuration line given is not a dict'
            raise SaltInvocationError(error_msg)
        if setting == stanza.get(value, False):
            _LOG.debug("Command '%s' already has: %s", value, setting)
            return True
        if setting:
            stanza[value] = setting
        else:
            del stanza[value]
        new_line = _dict_to_stanza(key, stanza)
        kwargs.update({'pattern': '^{0}.*?{{.*?}}'.format(key), 'flags': 24, 'append_if_not_found': True})
    kwargs.update({'repl': new_line})
    _LOG.debug("Setting file '%s' line: %s", conf_file, new_line)
    return __salt__['file.replace'](**kwargs)

def _dict_to_stanza(key, stanza):
    if False:
        return 10
    '\n    Convert a dict to a multi-line stanza\n    '
    ret = ''
    for skey in stanza:
        if stanza[skey] is True:
            stanza[skey] = ''
        ret += '    {} {}\n'.format(skey, stanza[skey])
    return '{0} {{\n{1}}}'.format(key, ret)