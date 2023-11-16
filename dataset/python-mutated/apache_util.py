""" Utility functions for certbot-apache plugin """
import atexit
import binascii
import fnmatch
import logging
import re
import subprocess
import sys
from contextlib import ExitStack
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from certbot import errors
from certbot import util
from certbot.compat import os
if sys.version_info >= (3, 9):
    import importlib.resources as importlib_resources
else:
    import importlib_resources
logger = logging.getLogger(__name__)

def get_mod_deps(mod_name: str) -> List[str]:
    if False:
        print('Hello World!')
    "Get known module dependencies.\n\n    .. note:: This does not need to be accurate in order for the client to\n        run.  This simply keeps things clean if the user decides to revert\n        changes.\n    .. warning:: If all deps are not included, it may cause incorrect parsing\n        behavior, due to enable_mod's shortcut for updating the parser's\n        currently defined modules (`.ApacheParser.add_mod`)\n        This would only present a major problem in extremely atypical\n        configs that use ifmod for the missing deps.\n\n    "
    deps = {'ssl': ['setenvif', 'mime']}
    return deps.get(mod_name, [])

def get_file_path(vhost_path: str) -> Optional[str]:
    if False:
        while True:
            i = 10
    'Get file path from augeas_vhost_path.\n\n    Takes in Augeas path and returns the file name\n\n    :param str vhost_path: Augeas virtual host path\n\n    :returns: filename of vhost\n    :rtype: str\n\n    '
    if not vhost_path or not vhost_path.startswith('/files/'):
        return None
    return _split_aug_path(vhost_path)[0]

def get_internal_aug_path(vhost_path: str) -> str:
    if False:
        print('Hello World!')
    'Get the Augeas path for a vhost with the file path removed.\n\n    :param str vhost_path: Augeas virtual host path\n\n    :returns: Augeas path to vhost relative to the containing file\n    :rtype: str\n\n    '
    return _split_aug_path(vhost_path)[1]

def _split_aug_path(vhost_path: str) -> Tuple[str, str]:
    if False:
        print('Hello World!')
    'Splits an Augeas path into a file path and an internal path.\n\n    After removing "/files", this function splits vhost_path into the\n    file path and the remaining Augeas path.\n\n    :param str vhost_path: Augeas virtual host path\n\n    :returns: file path and internal Augeas path\n    :rtype: `tuple` of `str`\n\n    '
    file_path = vhost_path[6:]
    internal_path: List[str] = []
    while not os.path.exists(file_path):
        (file_path, _, internal_path_part) = file_path.rpartition('/')
        internal_path.append(internal_path_part)
    return (file_path, '/'.join(reversed(internal_path)))

def parse_define_file(filepath: str, varname: str) -> Dict[str, str]:
    if False:
        for i in range(10):
            print('nop')
    ' Parses Defines from a variable in configuration file\n\n    :param str filepath: Path of file to parse\n    :param str varname: Name of the variable\n\n    :returns: Dict of Define:Value pairs\n    :rtype: `dict`\n\n    '
    return_vars: Dict[str, str] = {}
    a_opts = util.get_var_from_file(varname, filepath).split()
    for (i, v) in enumerate(a_opts):
        if v == '-D' and len(a_opts) >= i + 2:
            var_parts = a_opts[i + 1].partition('=')
            return_vars[var_parts[0]] = var_parts[2]
        elif len(v) > 2 and v.startswith('-D'):
            var_parts = v[2:].partition('=')
            return_vars[var_parts[0]] = var_parts[2]
    return return_vars

def unique_id() -> str:
    if False:
        i = 10
        return i + 15
    ' Returns an unique id to be used as a VirtualHost identifier'
    return binascii.hexlify(os.urandom(16)).decode('utf-8')

def included_in_paths(filepath: str, paths: Iterable[str]) -> bool:
    if False:
        i = 10
        return i + 15
    '\n    Returns true if the filepath is included in the list of paths\n    that may contain full paths or wildcard paths that need to be\n    expanded.\n\n    :param str filepath: Filepath to check\n    :param list paths: List of paths to check against\n\n    :returns: True if included\n    :rtype: bool\n    '
    return any((fnmatch.fnmatch(filepath, path) for path in paths))

def parse_defines(define_cmd: List[str]) -> Dict[str, str]:
    if False:
        return 10
    '\n    Gets Defines from httpd process and returns a dictionary of\n    the defined variables.\n\n    :param list define_cmd: httpd command to dump defines\n\n    :returns: dictionary of defined variables\n    :rtype: dict\n    '
    variables: Dict[str, str] = {}
    matches = parse_from_subprocess(define_cmd, 'Define: ([^ \\n]*)')
    try:
        matches.remove('DUMP_RUN_CFG')
    except ValueError:
        return {}
    for match in matches:
        parts = match.split('=', 1)
        value = parts[1] if len(parts) == 2 else ''
        variables[parts[0]] = value
    return variables

def parse_includes(inc_cmd: List[str]) -> List[str]:
    if False:
        print('Hello World!')
    '\n    Gets Include directives from httpd process and returns a list of\n    their values.\n\n    :param list inc_cmd: httpd command to dump includes\n\n    :returns: list of found Include directive values\n    :rtype: list of str\n    '
    return parse_from_subprocess(inc_cmd, '\\(.*\\) (.*)')

def parse_modules(mod_cmd: List[str]) -> List[str]:
    if False:
        i = 10
        return i + 15
    '\n    Get loaded modules from httpd process, and return the list\n    of loaded module names.\n\n    :param list mod_cmd: httpd command to dump loaded modules\n\n    :returns: list of found LoadModule module names\n    :rtype: list of str\n    '
    return parse_from_subprocess(mod_cmd, '(.*)_module')

def parse_from_subprocess(command: List[str], regexp: str) -> List[str]:
    if False:
        while True:
            i = 10
    'Get values from stdout of subprocess command\n\n    :param list command: Command to run\n    :param str regexp: Regexp for parsing\n\n    :returns: list parsed from command output\n    :rtype: list\n\n    '
    stdout = _get_runtime_cfg(command)
    return re.compile(regexp).findall(stdout)

def _get_runtime_cfg(command: List[str]) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Get runtime configuration info.\n\n    :param command: Command to run\n\n    :returns: stdout from command\n\n    '
    try:
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=False, env=util.env_no_snap_for_external_calls())
        (stdout, stderr) = (proc.stdout, proc.stderr)
    except (OSError, ValueError):
        logger.error('Error running command %s for runtime parameters!%s', command, os.linesep)
        raise errors.MisconfigurationError('Error accessing loaded Apache parameters: {0}'.format(command))
    if proc.returncode != 0:
        logger.warning('Error in checking parameter list: %s', stderr)
        raise errors.MisconfigurationError('Apache is unable to check whether or not the module is loaded because Apache is misconfigured.')
    return stdout

def find_ssl_apache_conf(prefix: str) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Find a TLS Apache config file in the dedicated storage.\n    :param str prefix: prefix of the TLS Apache config file to find\n    :return: the path the TLS Apache config file\n    :rtype: str\n    '
    file_manager = ExitStack()
    atexit.register(file_manager.close)
    ref = importlib_resources.files('certbot_apache').joinpath('_internal', 'tls_configs', '{0}-options-ssl-apache.conf'.format(prefix))
    return str(file_manager.enter_context(importlib_resources.as_file(ref)))