"""Create freedesktop.org-compliant thumbnails for album folders

This plugin is POSIX-only.
Spec: standards.freedesktop.org/thumbnail-spec/latest/index.html
"""
import ctypes
import ctypes.util
import os
import shutil
from hashlib import md5
from pathlib import PurePosixPath
from xdg import BaseDirectory
from beets import util
from beets.plugins import BeetsPlugin
from beets.ui import Subcommand, decargs
from beets.util import bytestring_path, displayable_path, syspath
from beets.util.artresizer import ArtResizer
BASE_DIR = os.path.join(BaseDirectory.xdg_cache_home, 'thumbnails')
NORMAL_DIR = bytestring_path(os.path.join(BASE_DIR, 'normal'))
LARGE_DIR = bytestring_path(os.path.join(BASE_DIR, 'large'))

class ThumbnailsPlugin(BeetsPlugin):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.config.add({'auto': True, 'force': False, 'dolphin': False})
        if self.config['auto'] and self._check_local_ok():
            self.register_listener('art_set', self.process_album)

    def commands(self):
        if False:
            print('Hello World!')
        thumbnails_command = Subcommand('thumbnails', help='Create album thumbnails')
        thumbnails_command.parser.add_option('-f', '--force', dest='force', action='store_true', default=False, help='force regeneration of thumbnails deemed fine (existing & recent enough)')
        thumbnails_command.parser.add_option('--dolphin', dest='dolphin', action='store_true', default=False, help='create Dolphin-compatible thumbnail information (for KDE)')
        thumbnails_command.func = self.process_query
        return [thumbnails_command]

    def process_query(self, lib, opts, args):
        if False:
            i = 10
            return i + 15
        self.config.set_args(opts)
        if self._check_local_ok():
            for album in lib.albums(decargs(args)):
                self.process_album(album)

    def _check_local_ok(self):
        if False:
            i = 10
            return i + 15
        "Check that everything is ready:\n        - local capability to resize images\n        - thumbnail dirs exist (create them if needed)\n        - detect whether we'll use PIL or IM\n        - detect whether we'll use GIO or Python to get URIs\n        "
        if not ArtResizer.shared.local:
            self._log.warning('No local image resizing capabilities, cannot generate thumbnails')
            return False
        for dir in (NORMAL_DIR, LARGE_DIR):
            if not os.path.exists(syspath(dir)):
                os.makedirs(syspath(dir))
        if not ArtResizer.shared.can_write_metadata:
            raise RuntimeError(f'Thumbnails: ArtResizer backend {ArtResizer.shared.method} unexpectedly cannot write image metadata.')
        self._log.debug(f'using {ArtResizer.shared.method} to write metadata')
        uri_getter = GioURI()
        if not uri_getter.available:
            uri_getter = PathlibURI()
        self._log.debug('using {0.name} to compute URIs', uri_getter)
        self.get_uri = uri_getter.uri
        return True

    def process_album(self, album):
        if False:
            while True:
                i = 10
        'Produce thumbnails for the album folder.'
        self._log.debug('generating thumbnail for {0}', album)
        if not album.artpath:
            self._log.info('album {0} has no art', album)
            return
        if self.config['dolphin']:
            self.make_dolphin_cover_thumbnail(album)
        size = ArtResizer.shared.get_size(album.artpath)
        if not size:
            self._log.warning('problem getting the picture size for {0}', album.artpath)
            return
        wrote = True
        if max(size) >= 256:
            wrote &= self.make_cover_thumbnail(album, 256, LARGE_DIR)
        wrote &= self.make_cover_thumbnail(album, 128, NORMAL_DIR)
        if wrote:
            self._log.info('wrote thumbnail for {0}', album)
        else:
            self._log.info('nothing to do for {0}', album)

    def make_cover_thumbnail(self, album, size, target_dir):
        if False:
            return 10
        'Make a thumbnail of given size for `album` and put it in\n        `target_dir`.\n        '
        target = os.path.join(target_dir, self.thumbnail_file_name(album.path))
        if os.path.exists(syspath(target)) and os.stat(syspath(target)).st_mtime > os.stat(syspath(album.artpath)).st_mtime:
            if self.config['force']:
                self._log.debug('found a suitable {1}x{1} thumbnail for {0}, forcing regeneration', album, size)
            else:
                self._log.debug('{1}x{1} thumbnail for {0} exists and is recent enough', album, size)
                return False
        resized = ArtResizer.shared.resize(size, album.artpath, target)
        self.add_tags(album, resized)
        shutil.move(syspath(resized), syspath(target))
        return True

    def thumbnail_file_name(self, path):
        if False:
            i = 10
            return i + 15
        'Compute the thumbnail file name\n        See https://standards.freedesktop.org/thumbnail-spec/latest/x227.html\n        '
        uri = self.get_uri(path)
        hash = md5(uri.encode('utf-8')).hexdigest()
        return bytestring_path(f'{hash}.png')

    def add_tags(self, album, image_path):
        if False:
            for i in range(10):
                print('nop')
        'Write required metadata to the thumbnail\n        See https://standards.freedesktop.org/thumbnail-spec/latest/x142.html\n        '
        mtime = os.stat(syspath(album.artpath)).st_mtime
        metadata = {'Thumb::URI': self.get_uri(album.artpath), 'Thumb::MTime': str(mtime)}
        try:
            ArtResizer.shared.write_metadata(image_path, metadata)
        except Exception:
            self._log.exception('could not write metadata to {0}', displayable_path(image_path))

    def make_dolphin_cover_thumbnail(self, album):
        if False:
            i = 10
            return i + 15
        outfilename = os.path.join(album.path, b'.directory')
        if os.path.exists(syspath(outfilename)):
            return
        artfile = os.path.split(album.artpath)[1]
        with open(syspath(outfilename), 'w') as f:
            f.write('[Desktop Entry]\n')
            f.write('Icon=./{}'.format(artfile.decode('utf-8')))
            f.close()
        self._log.debug('Wrote file {0}', displayable_path(outfilename))

class URIGetter:
    available = False
    name = 'Abstract base'

    def uri(self, path):
        if False:
            return 10
        raise NotImplementedError()

class PathlibURI(URIGetter):
    available = True
    name = 'Python Pathlib'

    def uri(self, path):
        if False:
            return 10
        return PurePosixPath(util.py3_path(path)).as_uri()

def copy_c_string(c_string):
    if False:
        print('Hello World!')
    'Copy a `ctypes.POINTER(ctypes.c_char)` value into a new Python\n    string and return it. The old memory is then safe to free.\n    '
    s = ctypes.cast(c_string, ctypes.c_char_p).value
    return b'' + s

class GioURI(URIGetter):
    """Use gio URI function g_file_get_uri. Paths must be utf-8 encoded."""
    name = 'GIO'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.libgio = self.get_library()
        self.available = bool(self.libgio)
        if self.available:
            self.libgio.g_type_init()
            self.libgio.g_file_get_uri.argtypes = [ctypes.c_char_p]
            self.libgio.g_file_new_for_path.restype = ctypes.c_void_p
            self.libgio.g_file_get_uri.argtypes = [ctypes.c_void_p]
            self.libgio.g_file_get_uri.restype = ctypes.POINTER(ctypes.c_char)
            self.libgio.g_object_unref.argtypes = [ctypes.c_void_p]

    def get_library(self):
        if False:
            i = 10
            return i + 15
        lib_name = ctypes.util.find_library('gio-2')
        try:
            if not lib_name:
                return False
            return ctypes.cdll.LoadLibrary(lib_name)
        except OSError:
            return False

    def uri(self, path):
        if False:
            return 10
        g_file_ptr = self.libgio.g_file_new_for_path(path)
        if not g_file_ptr:
            raise RuntimeError('No gfile pointer received for {}'.format(displayable_path(path)))
        try:
            uri_ptr = self.libgio.g_file_get_uri(g_file_ptr)
        finally:
            self.libgio.g_object_unref(g_file_ptr)
        if not uri_ptr:
            self.libgio.g_free(uri_ptr)
            raise RuntimeError('No URI received from the gfile pointer for {}'.format(displayable_path(path)))
        try:
            uri = copy_c_string(uri_ptr)
        finally:
            self.libgio.g_free(uri_ptr)
        try:
            return uri.decode(util._fsencoding())
        except UnicodeDecodeError:
            raise RuntimeError(f'Could not decode filename from GIO: {uri!r}')