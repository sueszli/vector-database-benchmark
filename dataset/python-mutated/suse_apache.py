"""
Support for Apache

Please note: The functions in here are SUSE-specific. Placing them in this
separate file will allow them to load only on SUSE systems, while still
loading under the ``apache`` namespace.
"""
import logging
import salt.utils.path
log = logging.getLogger(__name__)
__virtualname__ = 'apache'
__deprecated__ = (3009, 'apache', 'https://github.com/salt-extensions/saltext-apache')

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only load the module if apache is installed.\n    '
    if salt.utils.path.which('apache2ctl') and __grains__['os_family'] == 'Suse':
        return __virtualname__
    return (False, 'apache execution module not loaded: apache not installed.')

def check_mod_enabled(mod):
    if False:
        return 10
    "\n    Checks to see if the specific apache mod is enabled.\n\n    This will only be functional on operating systems that support\n    `a2enmod -l` to list the enabled mods.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' apache.check_mod_enabled status\n    "
    if mod.endswith('.load') or mod.endswith('.conf'):
        mod_name = mod[:-5]
    else:
        mod_name = mod
    cmd = 'a2enmod -l'
    try:
        active_mods = __salt__['cmd.run'](cmd, python_shell=False).split(' ')
    except Exception as e:
        return e
    return mod_name in active_mods

def a2enmod(mod):
    if False:
        print('Hello World!')
    "\n    Runs a2enmod for the given mod.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' apache.a2enmod vhost_alias\n    "
    ret = {}
    command = ['a2enmod', mod]
    try:
        status = __salt__['cmd.retcode'](command, python_shell=False)
    except Exception as e:
        return e
    ret['Name'] = 'Apache2 Enable Mod'
    ret['Mod'] = mod
    if status == 1:
        ret['Status'] = f'Mod {mod} Not found'
    elif status == 0:
        ret['Status'] = f'Mod {mod} enabled'
    else:
        ret['Status'] = status
    return ret

def a2dismod(mod):
    if False:
        return 10
    "\n    Runs a2dismod for the given mod.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' apache.a2dismod vhost_alias\n    "
    ret = {}
    command = ['a2dismod', mod]
    try:
        status = __salt__['cmd.retcode'](command, python_shell=False)
    except Exception as e:
        return e
    ret['Name'] = 'Apache2 Disable Mod'
    ret['Mod'] = mod
    if status == 256:
        ret['Status'] = f'Mod {mod} Not found'
    elif status == 0:
        ret['Status'] = f'Mod {mod} disabled'
    else:
        ret['Status'] = status
    return ret