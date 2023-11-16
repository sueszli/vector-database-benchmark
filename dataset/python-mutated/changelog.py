"""
Asset version change log

used to determine whether assets that were converted by an earlier version of
openage are still up to date.
"""
from __future__ import annotations
import typing
from ....log import warn
from ....testing.testing import TestError
ASSET_VERSION_FILENAME = 'asset_version'
GAMESPEC_VERSION_FILENAME = 'gamespec_version'
COMPONENTS = {'graphics', 'sounds', 'metadata', 'interface'}
CHANGES = ({'graphics', 'sounds'}, {'sounds'}, {'graphics'}, {'interface'}, {'interface'}, {'metadata'}, {'metadata'}, {'graphics'})
ASSET_VERSION = len(CHANGES) - 1

def changes(asset_version: int) -> set:
    if False:
        print('Hello World!')
    '\n    return all changed components since the passed version number.\n    '
    if asset_version >= len(CHANGES):
        warn('asset version from the future: %d', asset_version)
        warn('current version is: %d', ASSET_VERSION)
        warn('leaving assets as they are.')
        return set()
    changed_components = set()
    return changed_components

def test() -> typing.NoReturn:
    if False:
        i = 10
        return i + 15
    '\n    verify only allowed versions are stored in the changes\n    '
    for entry in CHANGES:
        if entry > COMPONENTS:
            invalid = entry - COMPONENTS
            raise TestError(f"'{invalid}': invalid changelog entry")