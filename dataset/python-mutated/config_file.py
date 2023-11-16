"""
Load and save the configuration : file <-> console var system
"""
from __future__ import annotations
import typing
from ..log import info, spam
if typing.TYPE_CHECKING:
    from openage.util.fslike.path import Path

def load_config_file(path: Path, set_cvar_func: typing.Callable, loaded_files: set=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Load a config file, with possible subfile, into the cvar system.\n\n    set_cvar is a function that accepts (key, value) to actually\n    add the data.\n    '
    if not loaded_files:
        loaded_files = set()
    if not path.is_file():
        info(f'config file {path} not found.')
        return
    if repr(path) in loaded_files:
        return
    info(f'loading config file {path}...')
    loaded_files.add(repr(path))
    with path.open() as config:
        for line in config:
            spam(f'Reading config line: {line}')
            lstrip = line.lstrip()
            if not lstrip or lstrip.startswith('#'):
                continue
            strip = lstrip.rstrip()
            split = strip.split()
            if split[0] == 'set' and len(split) >= 3:
                set_cvar_func(split[1], ' '.join(split[2:]))
            elif split[0] == 'load' and len(split) >= 2:
                for sub_path in split[1:]:
                    new_path = path.parent / sub_path
                    load_config_file(new_path, set_cvar_func, loaded_files)