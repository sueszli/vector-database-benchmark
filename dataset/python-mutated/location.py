"""
Determine the config file location and set up mounts.
"""
import os
import pathlib
from .. import config, default_dirs
from ..util.fslike.directory import Directory
from ..util.fslike.union import Union
from ..util.fslike.wrapper import WriteBlocker

def get_config_path(custom_cfg_dir: str=None) -> Directory:
    if False:
        return 10
    '\n    Locates the main configuration file by name in some searchpaths.\n    Optionally, mount a custom directory with highest priority.\n    '
    if config.DEVMODE:
        return Directory(os.path.join(config.BUILD_SRC_DIR, 'cfg')).root
    result = Union().root
    global_configs = pathlib.Path(config.GLOBAL_CONFIG_DIR)
    if global_configs.is_dir():
        result.mount(WriteBlocker(Directory(global_configs).root).root)
    home_cfg = default_dirs.get_dir('config_home') / 'openage'
    result.mount(Directory(home_cfg, create_if_missing=True).root)
    if custom_cfg_dir:
        result.mount(Directory(custom_cfg_dir).root)
    return result