"""
Receives cleaned-up srcdir and targetdir objects from .main, and drives the
actual conversion process.
"""
from __future__ import annotations
import typing
import timeit
from ...log import info, dbg
from ..processor.export.modpack_exporter import ModpackExporter
from ..service.debug_info import debug_gamedata_format
from ..service.debug_info import debug_string_resources, debug_registered_graphics, debug_modpack, debug_execution_time
from ..service.init.changelog import ASSET_VERSION
from ..service.read.gamedata import get_gamespec
from ..service.read.palette import get_palettes
from ..service.read.register_media import get_existing_graphics
from ..service.read.string_resource import get_string_resources
if typing.TYPE_CHECKING:
    from argparse import Namespace
    from openage.convert.value_object.init.game_version import GameVersion

def convert(args: Namespace) -> None:
    if False:
        print('Hello World!')
    '\n    args must hold srcdir and targetdir (FS-like objects),\n    plus any additional configuration options.\n    '
    convert_metadata(args)
    del args.palettes
    info(f'asset conversion complete; asset version: {ASSET_VERSION}')

def convert_metadata(args: Namespace) -> None:
    if False:
        while True:
            i = 10
    '\n    Converts the metadata part.\n    '
    if not args.flag('no_metadata'):
        info('converting metadata')
    args.converter = get_converter(args.game_version)
    palettes = get_palettes(args.srcdir, args.game_version)
    args.palettes = palettes
    if args.flag('no_metadata'):
        return
    gamedata_path = args.targetdir.joinpath('gamedata')
    if gamedata_path.exists():
        gamedata_path.removerecursive()
    read_start = timeit.default_timer()
    debug_gamedata_format(args.debugdir, args.debug_info, args.game_version)
    gamespec = get_gamespec(args.srcdir, args.game_version, not args.flag('no_pickle_cache'))
    if args.game_version.edition.game_id == 'SWGB':
        args.blend_mode_count = gamespec[0]['blend_mode_count_swgb'].value
    else:
        args.blend_mode_count = None
    string_resources = get_string_resources(args)
    debug_string_resources(args.debugdir, args.debug_info, string_resources)
    existing_graphics = get_existing_graphics(args)
    debug_registered_graphics(args.debugdir, args.debug_info, existing_graphics)
    read_end = timeit.default_timer()
    conversion_start = timeit.default_timer()
    modpacks = args.converter.convert(gamespec, args, string_resources, existing_graphics)
    conversion_end = timeit.default_timer()
    export_start = timeit.default_timer()
    for modpack in modpacks:
        ModpackExporter.export(modpack, args)
        debug_modpack(args.debugdir, args.debug_info, modpack)
    export_end = timeit.default_timer()
    stages_time = {'read': read_end - read_start, 'convert': conversion_end - conversion_start, 'export': export_end - export_start}
    debug_execution_time(args.debugdir, args.debug_info, stages_time)
    if args.flag('gen_extra_files'):
        dbg('generating extra files for visualization')

def get_converter(game_version: GameVersion):
    if False:
        i = 10
        return i + 15
    '\n    Returns the converter for the specified game version.\n    '
    game_edition = game_version.edition
    game_expansions = game_version.expansions
    if game_edition.game_id == 'ROR':
        from ..processor.conversion.ror.processor import RoRProcessor
        return RoRProcessor
    if game_edition.game_id == 'AOE1DE':
        from ..processor.conversion.de1.processor import DE1Processor
        return DE1Processor
    if game_edition.game_id == 'AOC':
        from ..processor.conversion.aoc.processor import AoCProcessor
        return AoCProcessor
    if game_edition.game_id == 'AOCDEMO':
        game_edition.game_id = 'AOC'
        from ..processor.conversion.aoc_demo.processor import DemoProcessor
        return DemoProcessor
    if game_edition.game_id == 'HDEDITION':
        from ..processor.conversion.hd.processor import HDProcessor
        return HDProcessor
    if game_edition.game_id == 'AOE2DE':
        from ..processor.conversion.de2.processor import DE2Processor
        return DE2Processor
    if game_edition.game_id == 'SWGB':
        if 'SWGB_CC' in [expansion.game_id for expansion in game_expansions]:
            from ..processor.conversion.swgbcc.processor import SWGBCCProcessor
            return SWGBCCProcessor
    raise RuntimeError(f'no valid converter found for game edition {game_edition.edition_name}')