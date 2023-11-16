"""
Disable user-controlled config for single-user servers

Applies patches to prevent loading configuration from the user's home directory.

Only used when launching a single-user server with disable_user_config=True.

This is where we still have some monkeypatches,
because we want to prevent loading configuration from user directories,
and `jupyter_core` functions don't allow that.

Due to extensions, we aren't able to apply patches in one place on the ServerApp,
we have to insert the patches at the lowest-level
on function objects themselves,
to ensure we modify calls to e.g. `jupyter_core.jupyter_path`
that may have been imported already!

We should perhaps ask for the necessary hooks to modify this in jupyter_core,
rather than keeing these monkey patches around.
"""
import os
from jupyter_core import paths

def _exclude_home(path_list):
    if False:
        return 10
    'Filter out any entries in a path list that are in my home directory.\n\n    Used to disable per-user configuration.\n    '
    home = os.path.expanduser('~/')
    for p in path_list:
        if not p.startswith(home):
            yield p
_original_jupyter_paths = None
_jupyter_paths_without_home = None

def _disable_user_config(serverapp):
    if False:
        while True:
            i = 10
    '\n    disable user-controlled sources of configuration\n    by excluding directories in their home from paths.\n\n    This _does not_ disable frontend config,\n    such as UI settings persistence.\n\n    1. Python config file paths\n    2. Search paths for extensions, etc.\n    3. import path\n    '
    original_jupyter_path = paths.jupyter_path()
    jupyter_path_without_home = list(_exclude_home(original_jupyter_path))
    default_config_file_paths = serverapp.config_file_paths
    config_file_paths = list(_exclude_home(default_config_file_paths))
    serverapp.__class__.config_file_paths = property(lambda self: config_file_paths)
    assert serverapp.config_file_paths == config_file_paths
    global _original_jupyter_paths, _jupyter_paths_without_home, _original_jupyter_config_dir
    _original_jupyter_paths = paths.jupyter_path()
    _jupyter_paths_without_home = list(_exclude_home(_original_jupyter_paths))

    def get_jupyter_path_without_home(*subdirs):
        if False:
            return 10
        from jupyterhub.singleuser._disable_user_config import _jupyter_paths_without_home
        paths = list(_jupyter_paths_without_home)
        if subdirs:
            paths = [os.path.join(p, *subdirs) for p in paths]
        return paths
    paths.jupyter_path.__code__ = get_jupyter_path_without_home.__code__
    if not os.getenv('JUPYTER_CONFIG_DIR') and (not list(_exclude_home([paths.jupyter_config_dir()]))):
        from jupyter_core.application import JupyterApp

        def get_env_config_dir(obj, cls=None):
            if False:
                while True:
                    i = 10
            return paths.ENV_CONFIG_PATH[0]
        JupyterApp.config_dir.get = get_env_config_dir
    serverapp.disable_user_config = True