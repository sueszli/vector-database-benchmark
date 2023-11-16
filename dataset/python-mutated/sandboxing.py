"""
Utility functions for our sandboxing model which is implemented on top of separate processes and
virtual environments.
"""
from __future__ import absolute_import
import fnmatch
import os
import sys
from sysconfig import get_path
from oslo_config import cfg
from st2common.constants.action import LIBS_DIR as ACTION_LIBS_DIR
from st2common.constants.pack import SYSTEM_PACK_NAMES
from st2common.content.utils import get_pack_base_path

def get_python_lib():
    if False:
        for i in range(10):
            print('nop')
    'Replacement for distutil.sysconfig.get_python_lib, returns a string with the python platform lib path (to site-packages)'
    return get_path('platlib')
__all__ = ['get_sandbox_python_binary_path', 'get_sandbox_python_path', 'get_sandbox_python_path_for_python_action', 'get_sandbox_path', 'get_sandbox_virtualenv_path']

def get_sandbox_python_binary_path(pack=None):
    if False:
        while True:
            i = 10
    '\n    Return path to the Python binary for the provided pack.\n    :param pack: Pack name.\n    :type pack: ``str``\n    '
    system_base_path = cfg.CONF.system.base_path
    virtualenv_path = os.path.join(system_base_path, 'virtualenvs', pack)
    if pack in SYSTEM_PACK_NAMES:
        python_path = sys.executable
    else:
        python_path = os.path.join(virtualenv_path, 'bin/python')
    return python_path

def get_sandbox_path(virtualenv_path):
    if False:
        return 10
    '\n    Return PATH environment variable value for the sandboxed environment.\n    This function makes sure that virtualenv/bin directory is in the path and has precedence over\n    the global PATH values.\n    Note: This function needs to be called from the parent process (one which is spawning a\n    sandboxed process).\n    '
    sandbox_path = []
    parent_path = os.environ.get('PATH', '')
    if not virtualenv_path:
        return parent_path
    parent_path = parent_path.split(':')
    parent_path = [path for path in parent_path if path]
    virtualenv_bin_path = os.path.join(virtualenv_path, 'bin/')
    sandbox_path.append(virtualenv_bin_path)
    sandbox_path.extend(parent_path)
    sandbox_path = ':'.join(sandbox_path)
    return sandbox_path

def get_sandbox_python_path(inherit_from_parent=True, inherit_parent_virtualenv=True):
    if False:
        return 10
    '\n    Return PYTHONPATH environment variable value for the new sandboxed environment.\n    This function takes into account if the current (parent) process is running under virtualenv\n    and other things like that.\n    Note: This function needs to be called from the parent process (one which is spawning a\n    sandboxed process).\n    :param inherit_from_parent: True to inheir PYTHONPATH from the current process.\n    :type inherit_from_parent: ``str``\n    :param inherit_parent_virtualenv: True to inherit virtualenv path if the current process is\n                                      running inside virtual environment.\n    :type inherit_parent_virtualenv: ``str``\n    '
    sandbox_python_path = []
    parent_python_path = os.environ.get('PYTHONPATH', '')
    parent_python_path = parent_python_path.split(':')
    parent_python_path = [path for path in parent_python_path if path]
    if inherit_from_parent:
        sandbox_python_path.extend(parent_python_path)
    if inherit_parent_virtualenv and is_in_virtualenv():
        site_packages_dir = get_python_lib()
        sys_prefix = os.path.abspath(sys.prefix)
        if sys_prefix not in site_packages_dir:
            raise ValueError(f'The file with "{sys_prefix}" is not found in "{site_packages_dir}".')
        sandbox_python_path.append(site_packages_dir)
    sandbox_python_path = ':'.join(sandbox_python_path)
    sandbox_python_path = ':' + sandbox_python_path
    return sandbox_python_path

def get_sandbox_python_path_for_python_action(pack, inherit_from_parent=True, inherit_parent_virtualenv=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return sandbox PYTHONPATH for a particular Python runner action.\n    Same as get_sandbox_python_path() function, but it's intended to be used for Python runner\n    actions.\n    "
    sandbox_python_path = get_sandbox_python_path(inherit_from_parent=inherit_from_parent, inherit_parent_virtualenv=inherit_parent_virtualenv)
    pack_base_path = get_pack_base_path(pack_name=pack)
    virtualenv_path = get_sandbox_virtualenv_path(pack=pack)
    if virtualenv_path and os.path.isdir(virtualenv_path):
        pack_virtualenv_lib_path = os.path.join(virtualenv_path, 'lib')
        virtualenv_directories = os.listdir(pack_virtualenv_lib_path)
        virtualenv_directories = [dir_name for dir_name in virtualenv_directories if fnmatch.fnmatch(dir_name, 'python*')]
        pack_actions_lib_paths = os.path.join(pack_base_path, 'actions', ACTION_LIBS_DIR)
        pack_virtualenv_lib_path = os.path.join(virtualenv_path, 'lib')
        pack_venv_lib_directory = os.path.join(pack_virtualenv_lib_path, virtualenv_directories[0])
        pack_venv_site_packages_directory = os.path.join(pack_virtualenv_lib_path, virtualenv_directories[0], 'site-packages')
        full_sandbox_python_path = [pack_venv_lib_directory, pack_venv_site_packages_directory, pack_actions_lib_paths, sandbox_python_path]
        sandbox_python_path = ':'.join(full_sandbox_python_path)
    return sandbox_python_path

def get_sandbox_virtualenv_path(pack):
    if False:
        i = 10
        return i + 15
    '\n    Return a path to the virtual environment for the provided pack.\n    '
    if pack in SYSTEM_PACK_NAMES:
        virtualenv_path = None
    else:
        system_base_path = cfg.CONF.system.base_path
        virtualenv_path = os.path.join(system_base_path, 'virtualenvs', pack)
    return virtualenv_path

def is_in_virtualenv():
    if False:
        i = 10
        return i + 15
    '\n    :return: True if we are currently in a virtualenv, else False\n    :rtype: ``Boolean``\n    '
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def get_virtualenv_prefix():
    if False:
        i = 10
        return i + 15
    '\n    :return: Returns a tuple where the first element is the name of the attribute\n             where we retrieved the virtualenv prefix from. The second element is\n             the virtualenv prefix.\n    '
    if hasattr(sys, 'real_prefix'):
        return ('sys.real_prefix', sys.real_prefix)
    elif hasattr(sys, 'base_prefix'):
        return ('sys.base_prefix', sys.base_prefix)
    return (None, None)

def set_virtualenv_prefix(prefix_tuple):
    if False:
        return 10
    '\n    :return: Sets the virtualenv prefix given a tuple returned from get_virtualenv_prefix()\n    '
    if prefix_tuple[0] == 'sys.real_prefix' and hasattr(sys, 'real_prefix'):
        sys.real_prefix = prefix_tuple[1]
    elif prefix_tuple[0] == 'sys.base_prefix' and hasattr(sys, 'base_prefix'):
        sys.base_prefix = prefix_tuple[1]

def clear_virtualenv_prefix():
    if False:
        i = 10
        return i + 15
    '\n    :return: Unsets / removes / resets the virtualenv prefix\n    '
    if hasattr(sys, 'real_prefix'):
        del sys.real_prefix
    if hasattr(sys, 'base_prefix'):
        sys.base_prefix = sys.prefix