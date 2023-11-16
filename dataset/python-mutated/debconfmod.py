"""
Support for Debconf
"""
import logging
import os
import re
import salt.utils.files
import salt.utils.path
import salt.utils.stringutils
import salt.utils.versions
log = logging.getLogger(__name__)
__func_alias__ = {'set_': 'set'}
__virtualname__ = 'debconf'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Confirm this module is on a Debian based system and that debconf-utils\n    is installed.\n    '
    if __grains__['os_family'] != 'Debian':
        return (False, 'The debconfmod module could not be loaded: unsupported OS family')
    if salt.utils.path.which('debconf-get-selections') is None:
        return (False, 'The debconfmod module could not be loaded: debconf-utils is not installed.')
    return __virtualname__

def _unpack_lines(out):
    if False:
        i = 10
        return i + 15
    '\n    Unpack the debconf lines\n    '
    rexp = '(?ms)^(?P<package>[^#]\\S+)[\t ]+(?P<question>\\S+)[\t ]+(?P<type>\\S+)[\t ]+(?P<value>[^\n]*)$'
    lines = re.findall(rexp, out)
    return lines

def get_selections(fetchempty=True):
    if False:
        print('Hello World!')
    "\n    Answers to debconf questions for all packages in the following format::\n\n        {'package': [['question', 'type', 'value'], ...]}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' debconf.get_selections\n    "
    selections = {}
    cmd = 'debconf-get-selections'
    out = __salt__['cmd.run_stdout'](cmd)
    lines = _unpack_lines(out)
    for line in lines:
        (package, question, type_, value) = line
        if fetchempty or value:
            selections.setdefault(package, []).append([question, type_, value])
    return selections

def show(name):
    if False:
        i = 10
        return i + 15
    "\n    Answers to debconf questions for a package in the following format::\n\n        [['question', 'type', 'value'], ...]\n\n    If debconf doesn't know about a package, we return None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' debconf.show <package name>\n    "
    selections = get_selections()
    result = selections.get(name)
    return result

def _set_file(path):
    if False:
        return 10
    '\n    Execute the set selections command for debconf\n    '
    cmd = 'debconf-set-selections {}'.format(path)
    __salt__['cmd.run_stdout'](cmd, python_shell=False)

def set_(package, question, type, value, *extra):
    if False:
        print('Hello World!')
    "\n    Set answers to debconf questions for a package.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' debconf.set <package> <question> <type> <value> [<value> ...]\n    "
    if extra:
        value = ' '.join((value,) + tuple(extra))
    (fd_, fname) = salt.utils.files.mkstemp(prefix='salt-', close_fd=False)
    line = '{} {} {} {}'.format(package, question, type, value)
    os.write(fd_, salt.utils.stringutils.to_bytes(line))
    os.close(fd_)
    _set_file(fname)
    os.unlink(fname)
    return True

def set_template(path, template, context, defaults, saltenv='base', **kwargs):
    if False:
        print('Hello World!')
    "\n    Set answers to debconf questions from a template.\n\n    path\n        location of the file containing the package selections\n\n    template\n        template format\n\n    context\n        variables to add to the template environment\n\n    default\n        default values for the template environment\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' debconf.set_template salt://pathto/pkg.selections.jinja jinja None None\n\n    "
    path = __salt__['cp.get_template'](path=path, dest=None, template=template, saltenv=saltenv, context=context, defaults=defaults, **kwargs)
    return set_file(path, saltenv, **kwargs)

def set_file(path, saltenv='base', **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Set answers to debconf questions from a file.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' debconf.set_file salt://pathto/pkg.selections\n    "
    if '__env__' in kwargs:
        kwargs.pop('__env__')
    path = __salt__['cp.cache_file'](path, saltenv)
    if path:
        _set_file(path)
        return True
    return False