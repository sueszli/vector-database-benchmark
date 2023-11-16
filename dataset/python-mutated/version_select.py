"""
Initial version detection based on user input.

TODO: Version selection.
"""
from __future__ import annotations
import typing
from ....log import warn, info
from ...service.init.version_detect import iterate_game_versions
from ...value_object.init.game_version import Support
from ...value_object.init.game_version import GameVersion
if typing.TYPE_CHECKING:
    from openage.convert.value_object.init.game_version import GameEdition, GameExpansion, GameVersion
    from openage.util.fslike.directory import Directory

def get_game_version(srcdir: Directory, avail_game_eds: list[GameEdition], avail_game_exps: list[GameExpansion]) -> GameVersion:
    if False:
        while True:
            i = 10
    '\n    Mount the input folders for conversion.\n    '
    info('Looking for compatible games to convert...')
    game_version = iterate_game_versions(srcdir, avail_game_eds, avail_game_exps)
    no_support = False
    if not game_version.edition or game_version.edition.support == Support.NOPE:
        warn(f'No valid game version(s) could not be detected in {srcdir.resolve_native_path()}')
        no_support = True
    else:
        broken_edition = game_version.edition.support == Support.BREAKS
        if broken_edition:
            warn('You have installed an incompatible game edition:')
            warn(' * \x1b[31;1m%s\x1b[m', game_version.edition)
            no_support = True
        broken_expansions = []
        for expansion in game_version.expansions:
            if expansion.support == Support.BREAKS:
                broken_expansions.append(expansion)
        if broken_expansions:
            warn('You have installed incompatible game expansions:')
            for expansion in broken_expansions:
                warn(' * \x1b[31;1m%s\x1b[m', expansion)
    if no_support:
        warn('You need at least one of:')
        for edition in avail_game_eds:
            if edition.support == Support.YES:
                warn(' * \x1b[34m%s\x1b[m', edition)
        return GameVersion(edition=None)
    info('Compatible game edition detected:')
    info(' * %s', game_version.edition.edition_name)
    if game_version.expansions:
        info('Compatible expansions detected:')
        for expansion in game_version.expansions:
            info(' * %s', expansion.expansion_name)
    return game_version