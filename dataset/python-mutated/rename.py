"""
Renaming interface assets and splitting into directories.
"""
from __future__ import annotations
import typing
from ....value_object.read.media.hardcoded.interface import ASSETS
from .cutter import ingame_hud_background_index
if typing.TYPE_CHECKING:
    from openage.util.fslike.path import Path

def hud_rename(filepath: Path) -> Path:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a human-usable name according to the original\n    and hardcoded metadata.\n    '
    try:
        return filepath.parent[f'hud{str(ingame_hud_background_index(int(filepath.stem))).zfill(4)}{filepath.suffix}']
    except ValueError:
        return asset_rename(filepath)

def asset_rename(filepath: Path) -> Path:
    if False:
        i = 10
        return i + 15
    '\n    Rename a slp asset path by the lookup map above.\n    '
    try:
        return filepath.parent[ASSETS[filepath.stem] + filepath.suffix]
    except (KeyError, AttributeError):
        return filepath