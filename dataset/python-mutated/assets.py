"""
Code for locating the game assets.
"""
from __future__ import annotations
import typing
import os
from pathlib import Path
from .util.fslike.directory import Directory
from .util.fslike.union import Union
from .util.fslike.wrapper import WriteBlocker
from . import config
from . import default_dirs
if typing.TYPE_CHECKING:
    from openage.util.fslike.union import UnionPath

def get_asset_path(custom_asset_dir: str=None) -> UnionPath:
    if False:
        while True:
            i = 10
    '\n    Returns a Path object for the game assets.\n\n    `custom_asset_dir` can a custom asset directory, which is mounted at the\n    top of the union filesystem (i.e. has highest priority).\n\n    This function is used by the both the conversion process\n    and the game startup. The conversion uses it for its output,\n    the game as its data source(s).\n    '
    result = Union().root
    if not custom_asset_dir and config.DEVMODE:
        result.mount(Directory(os.path.join(config.BUILD_SRC_DIR, 'assets')).root)
        return result
    global_data = Path(config.GLOBAL_ASSET_DIR)
    if global_data.is_dir():
        result.mount(WriteBlocker(Directory(global_data).root).root)
    home_data = default_dirs.get_dir('data_home') / 'openage'
    result.mount(Directory(home_data, create_if_missing=True).root / 'assets')
    if custom_asset_dir:
        result.mount(Directory(custom_asset_dir).root)
    return result

def test():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests whether a specific asset exists.\n    '
    from .testing.testing import assert_value
    import argparse
    fakecli = argparse.ArgumentParser()
    fakecli.add_argument('--asset-dir', default=None)
    args = fakecli.parse_args([])
    assert_value(get_asset_path(args.asset_dir)['test']['textures']['missing.png'].filesize, 580)