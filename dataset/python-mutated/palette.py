"""
Module for reading palette files.
"""
from __future__ import annotations
import typing
from ...value_object.read.media.colortable import ColorTable
from ...value_object.read.media_types import MediaType
if typing.TYPE_CHECKING:
    from openage.convert.value_object.init.game_version import GameVersion
    from openage.util.fslike.directory import Directory

def get_palettes(srcdir: Directory, game_version: GameVersion, index: int=None) -> dict[int, ColorTable]:
    if False:
        print('Hello World!')
    '\n    Read and create the color palettes.\n    '
    game_edition = game_version.edition
    palettes = {}
    if game_edition.game_id in ('ROR', 'AOC', 'SWGB', 'HDEDITION'):
        if index:
            palette_path = f'{MediaType.PALETTES.value}/{str(index)}.bina'
            palette_file = srcdir[palette_path]
            palette = ColorTable(palette_file.open('rb').read())
            palette_id = int(palette_file.stem)
            palettes[palette_id] = palette
        else:
            palette_dir = srcdir[MediaType.PALETTES.value]
            for palette_file in palette_dir.iterdir():
                if palette_file.stem.startswith('505'):
                    palette = ColorTable(palette_file.open('rb').read())
                    palette_id = int(palette_file.stem)
                    palettes[palette_id] = palette
            if game_edition.game_id == 'HDEDITION':
                pass
    elif game_edition.game_id in ('AOE1DE', 'AOE2DE'):
        conf_filepath = f'{MediaType.PALETTES.value}/palettes.conf'
        conf_file = srcdir[conf_filepath].open('rb')
        palette_paths = {}
        for line in conf_file.read().decode('utf-8').split('\n'):
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            (palette_id, filepath) = line.split(',')
            palette_id = int(palette_id)
            palette_paths[palette_id] = filepath
        if index:
            palette_path = f'{MediaType.PALETTES.value}/{palette_paths[index]}'
            palette = ColorTable(srcdir[palette_path].open('rb').read())
            palettes[index] = palette
        else:
            for (palette_id, filepath) in palette_paths.items():
                palette_path = f'{MediaType.PALETTES.value}/{filepath}'
                palette_file = srcdir[palette_path]
                palette = ColorTable(palette_file.open('rb').read())
                palettes[palette_id] = palette
    else:
        raise RuntimeError(f'no valid palette converter found for game edition{game_edition.edition_name}')
    return palettes