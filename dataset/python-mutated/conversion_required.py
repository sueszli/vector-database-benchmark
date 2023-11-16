"""
Test whether there already are converted modpacks present.
"""
from __future__ import annotations
import typing
from ....log import info
from .modpack_search import enumerate_modpacks
if typing.TYPE_CHECKING:
    from openage.util.fslike.union import UnionPath

def conversion_required(asset_dir: UnionPath) -> bool:
    if False:
        while True:
            i = 10
    '\n    Check if an asset conversion is required to run the game.\n\n    Asset conversions are required if:\n        - the modpack folder does not exist\n        - no modpacks inside the modpack folder exist\n        - the converted assets are outdated\n\n    :param asset_dir: The asset directory to check.\n    :type asset_dir: UnionPath\n    :return: True if an asset conversion is required, else False.\n    '
    try:
        modpacks = enumerate_modpacks(asset_dir / 'converted')
    except FileNotFoundError:
        modpacks = set()
    if not modpacks or (len(modpacks) == 1 and 'engine' in modpacks):
        info('No converted assets have been found')
        return True
    return False