"""

Code for locating the game assets.

All access to game assets should happen through objects obtained from get().
"""
import os
import pathlib
import sys
LINUX_DIRS = {'config_home': ('XDG_CONFIG_HOME', ('{HOME}/.config', {'HOME'})), 'data_home': ('XDG_DATA_HOME', ('{HOME}/.local/share', {'HOME'})), 'data_dirs': ('XDG_DATA_DIRS', ('/usr/local/share/:/usr/share/', {})), 'config_dirs': ('XDG_CONFIG_DIRS', ('/etc/xdg', {})), 'cache_home': ('XDG_CACHE_HOME', ('{HOME}/.cache', {'HOME'})), 'runtime_dir': ('XDG_RUNTIME_DIR', '/run/user/$UID')}
WINDOWS_DIRS = {'config_home': ('APPDATA', (False, None)), 'data_home': ('APPDATA', (False, None)), 'config_dirs': ('ALLUSERSPROFILE', (False, None))}

def get_dir(which):
    if False:
        i = 10
        return i + 15
    '\n    Returns directories used for data and config storage.\n    returns pathlib.Path\n    '
    platform_table = None
    if sys.platform.startswith('linux'):
        platform_table = LINUX_DIRS
    elif sys.platform.startswith('darwin'):
        raise RuntimeError('macOS not really supported')
    elif sys.platform.startswith('win32'):
        platform_table = WINDOWS_DIRS
    else:
        raise RuntimeError(f"unsupported platform: '{sys.platform}'")
    if which not in platform_table:
        raise ValueError(f"unknown directory requested: '{which}'")
    (env_var, (default_template, required_envs)) = platform_table[which]
    env_val = os.environ.get(env_var)
    if env_val:
        path = env_val
    elif default_template:
        env_vars = {var: os.environ.get(var) for var in required_envs}
        if not all(env_vars.values()):
            env_var_str = ', '.join([var for (var, val) in env_vars.items() if val is None])
            raise RuntimeError(f"could not reconstruct {which}, missing env variables: '{env_var_str}'")
        path = default_template.format(**env_vars)
    else:
        raise RuntimeError(f"could not find '{which}' in environment")
    return pathlib.Path(path)