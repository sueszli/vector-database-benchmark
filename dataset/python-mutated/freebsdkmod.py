"""
Module to manage FreeBSD kernel modules
"""
import os
import re
import salt.utils.files
__virtualname__ = 'kmod'
_LOAD_MODULE = '{0}_load="YES"'
_LOADER_CONF = '/boot/loader.conf'
_MODULE_RE = '^{0}_load="YES"'
_MODULES_RE = '^(\\w+)_load="YES"'

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only runs on FreeBSD systems\n    '
    if __grains__['kernel'] == 'FreeBSD':
        return __virtualname__
    return (False, 'The freebsdkmod execution module cannot be loaded: only available on FreeBSD systems.')

def _new_mods(pre_mods, post_mods):
    if False:
        print('Hello World!')
    '\n    Return a list of the new modules, pass an kldstat dict before running\n    modprobe and one after modprobe has run\n    '
    pre = set()
    post = set()
    for mod in pre_mods:
        pre.add(mod['module'])
    for mod in post_mods:
        post.add(mod['module'])
    return post - pre

def _rm_mods(pre_mods, post_mods):
    if False:
        print('Hello World!')
    '\n    Return a list of the new modules, pass an kldstat dict before running\n    modprobe and one after modprobe has run\n    '
    pre = set()
    post = set()
    for mod in pre_mods:
        pre.add(mod['module'])
    for mod in post_mods:
        post.add(mod['module'])
    return pre - post

def _get_module_name(line):
    if False:
        print('Hello World!')
    match = re.search(_MODULES_RE, line)
    if match:
        return match.group(1)
    return None

def _get_persistent_modules():
    if False:
        i = 10
        return i + 15
    '\n    Returns a list of modules in loader.conf that load on boot.\n    '
    mods = set()
    with salt.utils.files.fopen(_LOADER_CONF, 'r') as loader_conf:
        for line in loader_conf:
            line = salt.utils.stringutils.to_unicode(line)
            line = line.strip()
            mod_name = _get_module_name(line)
            if mod_name:
                mods.add(mod_name)
    return mods

def _set_persistent_module(mod):
    if False:
        return 10
    '\n    Add a module to loader.conf to make it persistent.\n    '
    if not mod or mod in mod_list(True) or mod not in available():
        return set()
    __salt__['file.append'](_LOADER_CONF, _LOAD_MODULE.format(mod))
    return {mod}

def _remove_persistent_module(mod, comment):
    if False:
        print('Hello World!')
    '\n    Remove module from loader.conf. If comment is true only comment line where\n    module is.\n    '
    if not mod or mod not in mod_list(True):
        return set()
    if comment:
        __salt__['file.comment'](_LOADER_CONF, _MODULE_RE.format(mod))
    else:
        __salt__['file.sed'](_LOADER_CONF, _MODULE_RE.format(mod), '')
    return {mod}

def available():
    if False:
        while True:
            i = 10
    "\n    Return a list of all available kernel modules\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kmod.available\n    "
    ret = []
    for path in __salt__['file.find']('/boot/kernel', name='*.ko$'):
        bpath = os.path.basename(path)
        comps = bpath.split('.')
        if 'ko' in comps:
            ret.append('.'.join(comps[:comps.index('ko')]))
    return ret

def check_available(mod):
    if False:
        while True:
            i = 10
    "\n    Check to see if the specified kernel module is available\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kmod.check_available vmm\n    "
    return mod in available()

def lsmod():
    if False:
        print('Hello World!')
    "\n    Return a dict containing information about currently loaded modules\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kmod.lsmod\n    "
    ret = []
    for line in __salt__['cmd.run']('kldstat').splitlines():
        comps = line.split()
        if not len(comps) > 2:
            continue
        if comps[0] == 'Id':
            continue
        if comps[4] == 'kernel':
            continue
        ret.append({'module': comps[4][:-3], 'size': comps[3], 'depcount': comps[1]})
    return ret

def mod_list(only_persist=False):
    if False:
        while True:
            i = 10
    "\n    Return a list of the loaded module names\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kmod.mod_list\n    "
    mods = set()
    if only_persist:
        if not _get_persistent_modules():
            return mods
        for mod in _get_persistent_modules():
            mods.add(mod)
    else:
        for mod in lsmod():
            mods.add(mod['module'])
    return sorted(list(mods))

def load(mod, persist=False):
    if False:
        print('Hello World!')
    "\n    Load the specified kernel module\n\n    mod\n        Name of the module to add\n\n    persist\n        Write the module to sysrc kld_modules to make it load on system reboot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kmod.load bhyve\n    "
    pre_mods = lsmod()
    response = __salt__['cmd.run_all']('kldload {}'.format(mod), python_shell=False)
    if response['retcode'] == 0:
        post_mods = lsmod()
        mods = _new_mods(pre_mods, post_mods)
        persist_mods = set()
        if persist:
            persist_mods = _set_persistent_module(mod)
        return sorted(list(mods | persist_mods))
    elif 'module already loaded or in kernel' in response['stderr']:
        if persist and mod not in _get_persistent_modules():
            persist_mods = _set_persistent_module(mod)
            return sorted(list(persist_mods))
        else:
            return [None]
    else:
        return 'Module {} not found'.format(mod)

def is_loaded(mod):
    if False:
        i = 10
        return i + 15
    "\n    Check to see if the specified kernel module is loaded\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kmod.is_loaded vmm\n    "
    return mod in mod_list()

def remove(mod, persist=False, comment=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Remove the specified kernel module\n\n    mod\n        Name of module to remove\n\n    persist\n        Also remove module from /boot/loader.conf\n\n    comment\n        If persist is set don't remove line from /boot/loader.conf but only\n        comment it\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kmod.remove vmm\n    "
    pre_mods = lsmod()
    res = __salt__['cmd.run_all']('kldunload {}'.format(mod), python_shell=False)
    if res['retcode'] == 0:
        post_mods = lsmod()
        mods = _rm_mods(pre_mods, post_mods)
        persist_mods = set()
        if persist:
            persist_mods = _remove_persistent_module(mod, comment)
        return sorted(list(mods | persist_mods))
    else:
        return 'Error removing module {}: {}'.format(mod, res['stderr'])