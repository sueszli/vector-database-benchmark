"""Provides utilities to read, write and manipulate m3u playlist files."""
import traceback
from beets.util import FilesystemError, mkdirall, normpath, syspath

class EmptyPlaylistError(Exception):
    """Raised when a playlist file without media files is saved or loaded."""
    pass

class M3UFile:
    """Reads and writes m3u or m3u8 playlist files."""

    def __init__(self, path):
        if False:
            i = 10
            return i + 15
        '``path`` is the absolute path to the playlist file.\n\n        The playlist file type, m3u or m3u8 is determined by 1) the ending\n        being m3u8 and 2) the file paths contained in the list being utf-8\n        encoded. Since the list is passed from the outside, this is currently\n        out of control of this class.\n        '
        self.path = path
        self.extm3u = False
        self.media_list = []

    def load(self):
        if False:
            return 10
        "Reads the m3u file from disk and sets the object's attributes."
        pl_normpath = normpath(self.path)
        try:
            with open(syspath(pl_normpath), 'rb') as pl_file:
                raw_contents = pl_file.readlines()
        except OSError as exc:
            raise FilesystemError(exc, 'read', (pl_normpath,), traceback.format_exc())
        self.extm3u = True if raw_contents[0].rstrip() == b'#EXTM3U' else False
        for line in raw_contents[1:]:
            if line.startswith(b'#'):
                continue
            self.media_list.append(normpath(line.rstrip()))
        if not self.media_list:
            raise EmptyPlaylistError

    def set_contents(self, media_list, extm3u=True):
        if False:
            while True:
                i = 10
        'Sets self.media_list to a list of media file paths.\n\n        Also sets additional flags, changing the final m3u-file\'s format.\n\n        ``media_list`` is a list of paths to media files that should be added\n        to the playlist (relative or absolute paths, that\'s the responsibility\n        of the caller). By default the ``extm3u`` flag is set, to ensure a\n        save-operation writes an m3u-extended playlist (comment "#EXTM3U" at\n        the top of the file).\n        '
        self.media_list = media_list
        self.extm3u = extm3u

    def write(self):
        if False:
            while True:
                i = 10
        'Writes the m3u file to disk.\n\n        Handles the creation of potential parent directories.\n        '
        header = [b'#EXTM3U'] if self.extm3u else []
        if not self.media_list:
            raise EmptyPlaylistError
        contents = header + self.media_list
        pl_normpath = normpath(self.path)
        mkdirall(pl_normpath)
        try:
            with open(syspath(pl_normpath), 'wb') as pl_file:
                for line in contents:
                    pl_file.write(line + b'\n')
                pl_file.write(b'\n')
        except OSError as exc:
            raise FilesystemError(exc, 'create', (pl_normpath,), traceback.format_exc())