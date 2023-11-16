"""
Module to manage Linux kernel modules
"""
import logging
import os
import re
import salt.utils.files
import salt.utils.path
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only runs on Linux systems\n    '
    return __grains__['kernel'] == 'Linux'

def _new_mods(pre_mods, post_mods):
    if False:
        while True:
            i = 10
    '\n    Return a list of the new modules, pass an lsmod dict before running\n    modprobe and one after modprobe has run\n    '
    pre = set()
    post = set()
    for mod in pre_mods:
        pre.add(mod['module'])
    for mod in post_mods:
        post.add(mod['module'])
    return post - pre

def _rm_mods(pre_mods, post_mods):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a list of the new modules, pass an lsmod dict before running\n    modprobe and one after modprobe has run\n    '
    pre = set()
    post = set()
    for mod in pre_mods:
        pre.add(mod['module'])
    for mod in post_mods:
        post.add(mod['module'])
    return pre - post

def _get_modules_conf():
    if False:
        while True:
            i = 10
    '\n    Return location of modules config file.\n    Default: /etc/modules\n    '
    if 'systemd' in __grains__:
        return '/etc/modules-load.d/salt_managed.conf'
    return '/etc/modules'

def _strip_module_name(mod):
    if False:
        i = 10
        return i + 15
    "\n    Return module name and strip configuration. It is possible insert modules\n    in this format:\n        bonding mode=4 miimon=1000\n    This method return only 'bonding'\n    "
    if mod.strip() == '':
        return False
    return mod.split()[0]

def _set_persistent_module(mod):
    if False:
        return 10
    '\n    Add module to configuration file to make it persistent. If module is\n    commented uncomment it.\n    '
    conf = _get_modules_conf()
    if not os.path.exists(conf):
        __salt__['file.touch'](conf)
    mod_name = _strip_module_name(mod)
    if not mod_name or mod_name in mod_list(True) or mod_name not in available():
        return set()
    escape_mod = re.escape(mod)
    if __salt__['file.search'](conf, '^#[\t ]*{}[\t ]*$'.format(escape_mod), multiline=True):
        __salt__['file.uncomment'](conf, escape_mod)
    else:
        __salt__['file.append'](conf, mod)
    return {mod_name}

def _remove_persistent_module(mod, comment):
    if False:
        i = 10
        return i + 15
    '\n    Remove module from configuration file. If comment is true only comment line\n    where module is.\n    '
    conf = _get_modules_conf()
    mod_name = _strip_module_name(mod)
    if not mod_name or mod_name not in mod_list(True):
        return set()
    escape_mod = re.escape(mod)
    if comment:
        __salt__['file.comment'](conf, '^[\t ]*{}[\t ]?'.format(escape_mod))
    else:
        __salt__['file.sed'](conf, '^[\t ]*{}[\t ]?'.format(escape_mod), '')
    return {mod_name}

def _which(cmd):
    if False:
        print('Hello World!')
    '\n    Utility function wrapper to error out early if a command is not found\n    '
    _cmd = salt.utils.path.which(cmd)
    if not _cmd:
        raise CommandExecutionError("Command '{}' cannot be found".format(cmd))
    return _cmd

def available():
    if False:
        while True:
            i = 10
    "\n    Return a list of all available kernel modules\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kmod.available\n    "
    ret = []
    mod_dir = os.path.join('/lib/modules/', os.uname()[2])
    built_in_file = os.path.join(mod_dir, 'modules.builtin')
    if os.path.exists(built_in_file):
        with salt.utils.files.fopen(built_in_file, 'r') as f:
            for line in f:
                ret.append(os.path.basename(line)[:-4])
    for (root, dirs, files) in salt.utils.path.os_walk(mod_dir):
        for fn_ in files:
            if '.ko' in fn_:
                ret.append(fn_[:fn_.index('.ko')].replace('-', '_'))
    if 'Arch' in __grains__['os_family']:
        mod_dir_arch = '/lib/modules/extramodules-' + os.uname()[2][0:3] + '-ARCH'
        for (root, dirs, files) in salt.utils.path.os_walk(mod_dir_arch):
            for fn_ in files:
                if '.ko' in fn_:
                    ret.append(fn_[:fn_.index('.ko')].replace('-', '_'))
    return sorted(list(ret))

def check_available(mod):
    if False:
        i = 10
        return i + 15
    "\n    Check to see if the specified kernel module is available\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kmod.check_available kvm\n    "
    return mod in available()

def lsmod():
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a dict containing information about currently loaded modules\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kmod.lsmod\n    "
    ret = []
    for line in __salt__['cmd.run'](_which('lsmod')).splitlines():
        comps = line.split()
        if not len(comps) > 2:
            continue
        if comps[0] == 'Module':
            continue
        mdat = {'size': comps[1], 'module': comps[0], 'depcount': comps[2]}
        if len(comps) > 3:
            mdat['deps'] = comps[3].split(',')
        else:
            mdat['deps'] = []
        ret.append(mdat)
    return ret

def mod_list(only_persist=False):
    if False:
        while True:
            i = 10
    "\n    Return a list of the loaded module names\n\n    only_persist\n        Only return the list of loaded persistent modules\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kmod.mod_list\n    "
    mods = set()
    if only_persist:
        conf = _get_modules_conf()
        if os.path.exists(conf):
            try:
                with salt.utils.files.fopen(conf, 'r') as modules_file:
                    for line in modules_file:
                        line = line.strip()
                        mod_name = _strip_module_name(line)
                        if not line.startswith('#') and mod_name:
                            mods.add(mod_name)
            except OSError:
                log.error('kmod module could not open modules file at %s', conf)
    else:
        for mod in lsmod():
            mods.add(mod['module'])
    return sorted(list(mods))

def load(mod, persist=False):
    if False:
        while True:
            i = 10
    "\n    Load the specified kernel module\n\n    mod\n        Name of module to add\n\n    persist\n        Write module to /etc/modules to make it load on system reboot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kmod.load kvm\n    "
    pre_mods = lsmod()
    res = __salt__['cmd.run_all']('{} {}'.format(_which('modprobe'), mod), python_shell=False)
    if res['retcode'] == 0:
        post_mods = lsmod()
        mods = _new_mods(pre_mods, post_mods)
        persist_mods = set()
        if persist:
            persist_mods = _set_persistent_module(mod)
        return sorted(list(mods | persist_mods))
    else:
        return 'Error loading module {}: {}'.format(mod, res['stderr'])

def is_loaded(mod):
    if False:
        i = 10
        return i + 15
    "\n    Check to see if the specified kernel module is loaded\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kmod.is_loaded kvm\n    "
    return mod in mod_list()

def remove(mod, persist=False, comment=True):
    if False:
        i = 10
        return i + 15
    "\n    Remove the specified kernel module\n\n    mod\n        Name of module to remove\n\n    persist\n        Also remove module from /etc/modules\n\n    comment\n        If persist is set don't remove line from /etc/modules but only\n        comment it\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kmod.remove kvm\n    "
    pre_mods = lsmod()
    res = __salt__['cmd.run_all']('{} {}'.format(_which('rmmod'), mod), python_shell=False)
    if res['retcode'] == 0:
        post_mods = lsmod()
        mods = _rm_mods(pre_mods, post_mods)
        persist_mods = set()
        if persist:
            persist_mods = _remove_persistent_module(mod, comment)
        return sorted(list(mods | persist_mods))
    else:
        return 'Error removing module {}: {}'.format(mod, res['stderr'])