"""
Converts media requested by export requests to files.
"""
from __future__ import annotations
import typing
import logging
import os
from openage.convert.entity_object.export.texture import Texture
from openage.convert.service import debug_info
from openage.convert.service.export.load_media_cache import load_media_cache
from openage.convert.value_object.read.media.blendomatic import Blendomatic
from openage.convert.value_object.read.media_types import MediaType
from openage.log import dbg, info, get_loglevel
from openage.util.strings import format_progress
if typing.TYPE_CHECKING:
    from argparse import Namespace
    from openage.convert.entity_object.export.media_export_request import MediaExportRequest
    from openage.convert.value_object.read.media.colortable import ColorTable
    from openage.convert.value_object.init.game_version import GameVersion
    from openage.util.fslike.path import Path

class MediaExporter:
    """
    Provides functions for converting media files and writing them to a targetdir.
    """

    @staticmethod
    def export(export_requests: dict[MediaType, list[MediaExportRequest]], sourcedir: Path, exportdir: Path, args: Namespace) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts files requested by MediaExportRequests.\n\n        :param export_requests: Export requests for media files. This is a dict of export requests\n                                by their media type.\n        :param sourcedir: Directory where all media assets are mounted. Source subfolder and\n                          source filename should be stored in the export request.\n        :param exportdir: Directory the resulting file(s) will be exported to. Target subfolder\n                          and target filename should be stored in the export request.\n        :param args: Converter arguments.\n        :type export_requests: dict\n        :type sourcedir: Path\n        :type exportdir: Path\n        :type args: Namespace\n        '
        cache_info = {}
        if args.game_version.edition.media_cache:
            cache_info = load_media_cache(args.game_version.edition.media_cache)
        for media_type in export_requests.keys():
            cur_export_requests = export_requests[media_type]
            export_func = None
            kwargs = {}
            if media_type is MediaType.TERRAIN:
                kwargs['game_version'] = args.game_version
                kwargs['palettes'] = args.palettes
                kwargs['compression_level'] = args.compression_level
                export_func = MediaExporter._export_terrain
                info('-- Exporting terrain files...')
            elif media_type is MediaType.GRAPHICS:
                kwargs['palettes'] = args.palettes
                kwargs['compression_level'] = args.compression_level
                kwargs['cache_info'] = cache_info
                export_func = MediaExporter._export_graphics
                info('-- Exporting graphics files...')
            elif media_type is MediaType.SOUNDS:
                kwargs['loglevel'] = args.debug_info
                kwargs['debugdir'] = args.debugdir
                export_func = MediaExporter._export_sound
                info('-- Exporting sound files...')
            elif media_type is MediaType.BLEND:
                kwargs['blend_mode_count'] = args.blend_mode_count
                export_func = MediaExporter._export_blend
                info('-- Exporting blend files...')
            total_count = len(cur_export_requests)
            for (count, request) in enumerate(cur_export_requests, start=1):
                export_func(request, sourcedir, exportdir, **kwargs)
                print(f'-- Files done: {format_progress(count, total_count)}', end='\r', flush=True)
        if args.debug_info > 5:
            cachedata = {}
            for request in export_requests[MediaType.GRAPHICS]:
                kwargs = {}
                kwargs['palettes'] = args.palettes
                kwargs['compression_level'] = args.compression_level
                cache = MediaExporter._get_media_cache(request, sourcedir, args.palettes, compression_level=2)
                cachedata[request] = cache
            debug_info.debug_media_cache(args.debugdir, args.debug_info, sourcedir, cachedata, args.game_version)

    @staticmethod
    def _export_blend(export_request: MediaExportRequest, sourcedir: Path, exportdir: Path, blend_mode_count: int=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert and export a blending mode.\n\n        :param export_request: Export request for a blending mask.\n        :param sourcedir: Directory where all media assets are mounted. Source subfolder and\n                          source filename should be stored in the export request.\n        :param exportdir: Directory the resulting file(s) will be exported to. Target subfolder\n                          and target filename should be stored in the export request.\n        :param blend_mode_count: Number of blending modes extracted from the source file.\n        :type export_request: MediaExportRequest\n        :type sourcedir: Path\n        :type exportdir: Path\n        :type blend_mode_count: int\n        '
        source_file = sourcedir.joinpath(export_request.source_filename)
        media_file = source_file.open('rb')
        blend_data = Blendomatic(media_file, blend_mode_count)
        from .texture_merge import merge_frames
        textures = blend_data.get_textures()
        for (idx, texture) in enumerate(textures):
            merge_frames(texture)
            MediaExporter.save_png(texture, exportdir[export_request.targetdir], f'{export_request.target_filename}{idx}.png')
            if get_loglevel() <= logging.DEBUG:
                MediaExporter.log_fileinfo(source_file, exportdir[export_request.targetdir, f'{export_request.target_filename}{idx}.png'])

    @staticmethod
    def _export_graphics(export_request: MediaExportRequest, sourcedir: Path, exportdir: Path, palettes: dict[int, ColorTable], compression_level: int, cache_info: dict=None) -> None:
        if False:
            while True:
                i = 10
        '\n        Convert and export a graphics file.\n\n        :param export_request: Export request for a graphics file.\n        :param sourcedir: Directory where all media assets are mounted. Source subfolder and\n                          source filename should be stored in the export request.\n        :param exportdir: Directory the resulting file(s) will be exported to. Target subfolder\n                          and target filename should be stored in the export request.\n        :param palettes: Palettes used by the game.\n        :param compression_level: PNG compression level for the resulting image file.\n        :param cache_info: Media cache information with compression parameters from a previous run.\n        :type export_request: MediaExportRequest\n        :type sourcedir: Path\n        :type exportdir: Path\n        :type palettes: dict\n        :type compression_level: int\n        :type cache_info: tuple\n        '
        source_file = sourcedir[export_request.get_type().value, export_request.source_filename]
        try:
            media_file = source_file.open('rb')
        except FileNotFoundError:
            if source_file.suffix.lower() in ('.smx', '.sld'):
                other_filename = export_request.source_filename[:-3] + 'smp'
                source_file = sourcedir[export_request.get_type().value, other_filename]
                export_request.set_source_filename(other_filename)
            media_file = source_file.open('rb')
        if source_file.suffix.lower() == '.slp':
            from ...value_object.read.media.slp import SLP
            image = SLP(media_file.read())
        elif source_file.suffix.lower() == '.smp':
            from ...value_object.read.media.smp import SMP
            image = SMP(media_file.read())
        elif source_file.suffix.lower() == '.smx':
            from ...value_object.read.media.smx import SMX
            image = SMX(media_file.read())
        elif source_file.suffix.lower() == '.sld':
            from ...value_object.read.media.sld import SLD
            image = SLD(media_file.read())
        else:
            raise SyntaxError(f'Source file {source_file.name} has an unrecognized extension: {source_file.suffix.lower()}')
        packer_cache = None
        compr_cache = None
        if cache_info:
            cache_params = cache_info.get(export_request.source_filename, None)
            if cache_params:
                packer_cache = cache_params['packer_settings']
                compression_level = cache_params['compr_settings'][0]
                compr_cache = cache_params['compr_settings'][1:]
        from .texture_merge import merge_frames
        texture = Texture(image, palettes)
        merge_frames(texture, cache=packer_cache)
        MediaExporter.save_png(texture, exportdir[export_request.targetdir], export_request.target_filename, compression_level=compression_level, cache=compr_cache)
        metadata = {export_request.target_filename: texture.get_metadata()}
        export_request.set_changed()
        export_request.notify_observers(metadata)
        export_request.clear_changed()
        if get_loglevel() <= logging.DEBUG:
            MediaExporter.log_fileinfo(source_file, exportdir[export_request.targetdir, export_request.target_filename])

    @staticmethod
    def _export_interface(export_request: MediaExportRequest, sourcedir: Path, **kwargs) -> None:
        if False:
            print('Hello World!')
        '\n        Convert and export a sprite file.\n        '

    @staticmethod
    def _export_palette(export_request: MediaExportRequest, sourcedir: Path, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Convert and export a palette file.\n        '

    @staticmethod
    def _export_sound(export_request: MediaExportRequest, sourcedir: Path, exportdir: Path, **kwargs) -> None:
        if False:
            print('Hello World!')
        '\n        Convert and export a sound file.\n\n        :param export_request: Export request for a sound file.\n        :param sourcedir: Directory where all media assets are mounted. Source subfolder and\n                          source filename should be stored in the export request.\n        :param exportdir: Directory the resulting file(s) will be exported to. Target subfolder\n                          and target filename should be stored in the export request.\n        :type export_request: MediaExportRequest\n        :type sourcedir: Path\n        :type exportdir: Path\n        '
        source_file = sourcedir[export_request.get_type().value, export_request.source_filename]
        if source_file.is_file():
            with source_file.open_r() as infile:
                media_file = infile.read()
        else:
            debug_info.debug_not_found_sounds(kwargs['debugdir'], kwargs['loglevel'], source_file)
            return
        from ...service.export.opus.opusenc import encode
        soundata = encode(media_file)
        if isinstance(soundata, (str, int)):
            raise RuntimeError(f'opusenc failed: {soundata}')
        export_file = exportdir[export_request.targetdir, export_request.target_filename]
        with export_file.open_w() as outfile:
            outfile.write(soundata)
        if get_loglevel() <= logging.DEBUG:
            MediaExporter.log_fileinfo(source_file, exportdir[export_request.targetdir, export_request.target_filename])

    @staticmethod
    def _export_terrain(export_request: MediaExportRequest, sourcedir: Path, exportdir: Path, palettes: dict[int, ColorTable], game_version: GameVersion, compression_level: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert and export a terrain graphics file.\n\n        :param export_request: Export request for a terrain graphics file.\n        :param sourcedir: Directory where all media assets are mounted. Source subfolder and\n                          source filename should be stored in the export request.\n        :param exportdir: Directory the resulting file(s) will be exported to. Target subfolder\n                          and target filename should be stored in the export request.\n        :param game_version: Game edition and expansion info.\n        :param palettes: Palettes used by the game.\n        :param compression_level: PNG compression level for the resulting image file.\n        :type export_request: MediaExportRequest\n        :type sourcedir: Directory\n        :type exportdir: Directory\n        :type palettes: dict\n        :type game_version: GameVersion\n        :type compression_level: int\n        '
        source_file = sourcedir[export_request.get_type().value, export_request.source_filename]
        if source_file.suffix.lower() == '.slp':
            from ...value_object.read.media.slp import SLP
            media_file = source_file.open('rb')
            image = SLP(media_file.read())
        elif source_file.suffix.lower() == '.dds':
            pass
        elif source_file.suffix.lower() == '.png':
            from shutil import copyfileobj
            src_path = source_file.open('rb')
            dst_path = exportdir[export_request.targetdir, export_request.target_filename].open('wb')
            copyfileobj(src_path, dst_path)
            return
        else:
            raise SyntaxError(f'Source file {source_file.name} has an unrecognized extension: {source_file.suffix.lower()}')
        if game_version.edition.game_id in ('AOC', 'SWGB'):
            from .terrain_merge import merge_terrain
            texture = Texture(image, palettes)
            merge_terrain(texture)
        else:
            from .texture_merge import merge_frames
            texture = Texture(image, palettes)
            merge_frames(texture)
        MediaExporter.save_png(texture, exportdir[export_request.targetdir], export_request.target_filename, compression_level)
        if get_loglevel() <= logging.DEBUG:
            MediaExporter.log_fileinfo(source_file, exportdir[export_request.targetdir, export_request.target_filename])

    @staticmethod
    def _get_media_cache(export_request: MediaExportRequest, sourcedir: Path, palettes: dict[int, ColorTable], compression_level: int) -> None:
        if False:
            print('Hello World!')
        '\n        Convert a media file and return the used settings. This performs\n        a dry run, i.e. the graphics media is not saved on the filesystem.\n\n        :param export_request: Export request for a graphics file.\n        :param sourcedir: Directory where all media assets are mounted. Source subfolder and\n                          source filename should be stored in the export request.\n        :param exportdir: Directory the resulting file(s) will be exported to. Target subfolder\n                          and target filename should be stored in the export request.\n        :param palettes: Palettes used by the game.\n        :param compression_level: PNG compression level for the resulting image file.\n        :type export_request: MediaExportRequest\n        :type sourcedir: Path\n        :type exportdir: Path\n        :type palettes: dict\n        :type compression_level: int\n        '
        source_file = sourcedir[export_request.get_type().value, export_request.source_filename]
        try:
            media_file = source_file.open('rb')
        except FileNotFoundError:
            if source_file.suffix.lower() == '.smx':
                other_filename = export_request.source_filename[:-1] + 'p'
                source_file = sourcedir[export_request.get_type().value, other_filename]
            media_file = source_file.open('rb')
        if source_file.suffix.lower() == '.slp':
            from ...value_object.read.media.slp import SLP
            image = SLP(media_file.read())
        elif source_file.suffix.lower() == '.smp':
            from ...value_object.read.media.smp import SMP
            image = SMP(media_file.read())
        elif source_file.suffix.lower() == '.smx':
            from ...value_object.read.media.smx import SMX
            image = SMX(media_file.read())
        elif source_file.suffix.lower() == '.sld':
            from ...value_object.read.media.sld import SLD
            image = SLD(media_file.read())
        else:
            raise SyntaxError(f'Source file {source_file.name} has an unrecognized extension: {source_file.suffix.lower()}')
        from .texture_merge import merge_frames
        texture = Texture(image, palettes)
        merge_frames(texture)
        MediaExporter.save_png(texture, None, None, compression_level=compression_level, cache=None, dry_run=True)
        return texture.get_cache_params()

    @staticmethod
    def save_png(texture: Texture, targetdir: Path, filename: str, compression_level: int=1, cache: dict=None, dry_run: bool=False) -> None:
        if False:
            while True:
                i = 10
        '\n        Store the image data into the target directory path,\n        with given filename="dir/out.png".\n\n        :param texture: Texture with an image atlas.\n        :param targetdir: Directory where the image file is created.\n        :param filename: Name of the resulting image file.\n        :param compression_level: PNG compression level used for the resulting image file.\n        :param dry_run: If True, create the PNG but don\'t save it as a file.\n        :type texture: Texture\n        :type targetdir: Directory\n        :type filename: str\n        :type compression_level: int\n        :type dry_run: bool\n        '
        from ...service.export.png import png_create
        compression_levels = {0: png_create.CompressionMethod.COMPR_NONE, 1: png_create.CompressionMethod.COMPR_DEFAULT, 2: png_create.CompressionMethod.COMPR_OPTI, 3: png_create.CompressionMethod.COMPR_GREEDY, 4: png_create.CompressionMethod.COMPR_AGGRESSIVE}
        if not dry_run:
            (_, ext) = os.path.splitext(filename)
            if ext != '.png':
                raise ValueError(f"Filename invalid, a texture must be savedas '*.png', not '*.{ext}'")
        compression_method = compression_levels.get(compression_level, png_create.CompressionMethod.COMPR_DEFAULT)
        (png_data, compr_params) = png_create.save(texture.image_data.data, compression_method, cache)
        if not dry_run:
            with targetdir[filename].open('wb') as imagefile:
                imagefile.write(png_data)
        if compr_params:
            texture.best_compr = (compression_level, *compr_params)

    @staticmethod
    def log_fileinfo(source_file: Path, target_file: Path) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Log source and target file information to the shell.\n        '
        source_format = source_file.suffix[1:].upper()
        target_format = target_file.suffix[1:].upper()
        source_path = source_file.resolve_native_path()
        if source_path:
            source_size = os.path.getsize(source_path)
        else:
            with source_file.open('r') as src:
                src.seek(0, os.SEEK_END)
                source_size = src.tell()
        target_path = target_file.resolve_native_path()
        target_size = os.path.getsize(target_path)
        log = f'Converted: {source_file.name} ({source_format}, {source_size}B) -> {target_file.name} ({target_format}, {target_size}B | {target_size / source_size * 100 - 100:+.1f}%)'
        dbg(log)