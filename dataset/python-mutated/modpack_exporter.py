"""
Export data from a modpack to files.
"""
from __future__ import annotations
import typing
from ....log import info
from .data_exporter import DataExporter
from .generate_manifest_hashes import generate_hashes
from .media_exporter import MediaExporter
if typing.TYPE_CHECKING:
    from argparse import Namespace
    from openage.convert.entity_object.conversion.modpack import Modpack

class ModpackExporter:
    """
    Writes the contents of a created modpack into a targetdir.
    """

    @staticmethod
    def export(modpack: Modpack, args: Namespace) -> None:
        if False:
            print('Hello World!')
        '\n        Export a modpack to a directory.\n\n        :param modpack: Modpack that is going to be exported.\n        :param args: Converter arguments.\n        :type modpack: ..dataformats.modpack.Modpack\n        :type args: Namespace\n        '
        sourcedir = args.srcdir
        exportdir = args.targetdir
        modpack_dir = exportdir.joinpath(f'{modpack.info.packagename}')
        info('Starting export...')
        info('Dumping info file...')
        DataExporter.export([modpack.info], modpack_dir)
        info('Dumping data files...')
        DataExporter.export(modpack.get_data_files(), modpack_dir)
        if args.flag('no_media'):
            info('Skipping media file export...')
            return
        info('Exporting media files...')
        MediaExporter.export(modpack.get_media_files(), sourcedir, modpack_dir, args)
        info('Dumping metadata files...')
        DataExporter.export(modpack.get_metadata_files(), modpack_dir)
        generate_hashes(modpack, modpack_dir)
        DataExporter.export([modpack.manifest], modpack_dir)