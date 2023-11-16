from enum import IntEnum
import math
import os
import re
import shutil
import struct
import sys
import unicodedata
from PyQt6.QtCore import QStandardPaths
from picard import log
from picard.const.sys import IS_LINUX, IS_MACOS, IS_WIN
from picard.util import WIN_MAX_DIRPATH_LEN, WIN_MAX_FILEPATH_LEN, WIN_MAX_NODE_LEN, _io_encoding, decode_filename, encode_filename, samefile
win32api = None
if IS_WIN:
    try:
        import pywintypes
        import win32api
    except ImportError as e:
        log.warning('pywin32 not available: %s', e)

def _get_utf16_length(text):
    if False:
        while True:
            i = 10
    'Returns the number of code points used by a unicode object in its\n    UTF-16 representation.\n    '
    if isinstance(text, bytes):
        return len(text)
    if sys.maxunicode == 65535:
        return len(text)
    return len(text.encode('utf-16%ce' % sys.byteorder[0])) // 2

def _shorten_to_utf16_length(text, length):
    if False:
        while True:
            i = 10
    'Truncates a str object to the given number of UTF-16 code points.\n    '
    assert isinstance(text, str), 'This function only works on unicode'
    if sys.maxunicode == 65535:
        shortened = text[:length]
        last = shortened[-1:]
        if last and 55296 <= ord(last) <= 56319:
            return shortened[:-1]
        return shortened
    enc = 'utf-16%ce' % sys.byteorder[0]
    shortened = text.encode(enc)[:length * 2]
    last = shortened[-2:]
    if last and 55296 <= struct.unpack('=H', last)[0] <= 56319:
        shortened = shortened[:-2]
    return shortened.decode(enc)

def _shorten_to_utf16_nfd_length(text, length):
    if False:
        while True:
            i = 10
    text = unicodedata.normalize('NFD', text)
    newtext = _shorten_to_utf16_length(text, length)
    try:
        if unicodedata.combining(text[len(newtext)]):
            newtext = newtext[:-1]
    except IndexError:
        pass
    return unicodedata.normalize('NFC', newtext)
_re_utf8 = re.compile('^utf([-_]?8)$', re.IGNORECASE)

def _shorten_to_bytes_length(text, length):
    if False:
        return 10
    'Truncates a unicode object to the given number of bytes it would take\n    when encoded in the "filesystem encoding".\n    '
    assert isinstance(text, str), 'This function only works on unicode'
    raw = encode_filename(text)
    if len(raw) <= length:
        return text
    if len(raw) == len(text):
        return text[:length]
    if _re_utf8.match(_io_encoding):
        i = length
        while i > 0 and raw[i] & 192 == 128:
            i -= 1
        return decode_filename(raw[:i])
    i = length
    while i > 0:
        try:
            return decode_filename(raw[:i])
        except UnicodeDecodeError:
            pass
        i -= 1
    return ''

class ShortenMode(IntEnum):
    BYTES = 0
    UTF16 = 1
    UTF16_NFD = 2

def shorten_filename(filename, length, mode):
    if False:
        for i in range(10):
            print('nop')
    'Truncates a filename to the given number of thingies,\n    as implied by `mode`.\n    '
    if isinstance(filename, bytes):
        return filename[:length]
    if mode == ShortenMode.BYTES:
        return _shorten_to_bytes_length(filename, length)
    if mode == ShortenMode.UTF16:
        return _shorten_to_utf16_length(filename, length)
    if mode == ShortenMode.UTF16_NFD:
        return _shorten_to_utf16_nfd_length(filename, length)

def shorten_path(path, length, mode):
    if False:
        for i in range(10):
            print('nop')
    "Reduce path nodes' length to given limit(s).\n\n    path: Absolute or relative path to shorten.\n    length: Maximum number of code points / bytes allowed in a node.\n    mode: One of the enum values from ShortenMode.\n    "

    def shorten(name, length):
        if False:
            return 10
        return name and shorten_filename(name, length, mode).strip() or ''
    (dirpath, filename) = os.path.split(path)
    (fileroot, ext) = os.path.splitext(filename)
    return os.path.join(os.path.join(*[shorten(node, length) for node in dirpath.split(os.path.sep)]), shorten(fileroot, length - len(ext)) + ext)

def _shorten_to_utf16_ratio(text, ratio):
    if False:
        i = 10
        return i + 15
    'Shortens the string to the given ratio (and strips it).'
    length = _get_utf16_length(text)
    limit = max(1, int(math.floor(length / ratio)))
    if isinstance(text, bytes):
        return text[:limit].strip()
    else:
        return _shorten_to_utf16_length(text, limit).strip()

class WinPathTooLong(OSError):
    pass

def _make_win_short_filename(relpath, reserved=0):
    if False:
        while True:
            i = 10
    'Shorten a relative file path according to WinAPI quirks.\n\n    relpath: The file\'s path.\n    reserved: Number of characters reserved for the parent path to be joined with,\n              e.g. 3 if it will be joined with "X:\\", respectively 5 for "X:\\y\\".\n              (note the inclusion of the final backslash)\n    '
    MAX_NODE_LENGTH = WIN_MAX_NODE_LEN - 29
    remaining = WIN_MAX_DIRPATH_LEN - reserved

    def shorten(path, length):
        if False:
            for i in range(10):
                print('nop')
        return shorten_path(path, length, mode=ShortenMode.UTF16)
    xlength = _get_utf16_length
    relpath = shorten(relpath, MAX_NODE_LENGTH)
    (dirpath, filename) = os.path.split(relpath)
    dplen = xlength(dirpath)
    if dplen <= remaining:
        filename_max = WIN_MAX_FILEPATH_LEN - (reserved + dplen + 1)
        filename = shorten(filename, filename_max)
        return os.path.join(dirpath, filename)
    try:
        computed = _make_win_short_filename._computed
    except AttributeError:
        computed = _make_win_short_filename._computed = {}
    try:
        (finaldirpath, filename_max) = computed[dirpath, reserved]
    except KeyError:
        dirnames = dirpath.split(os.path.sep)
        remaining -= len(dirnames) - 1
        average = float(remaining) / len(dirnames)
        if average < 1:
            raise WinPathTooLong('Path too long. You need to move renamed files to a different directory.')
        shortdirnames = [dn for dn in dirnames if len(dn) <= average]
        totalchars = sum(map(xlength, dirnames))
        shortdirchars = sum(map(xlength, shortdirnames))
        if remaining > shortdirchars + len(dirnames) - len(shortdirnames):
            ratio = float(totalchars - shortdirchars) / (remaining - shortdirchars)
            for (i, dn) in enumerate(dirnames):
                if len(dn) > average:
                    dirnames[i] = _shorten_to_utf16_ratio(dn, ratio)
        else:
            ratio = float(totalchars) / remaining
            dirnames = [_shorten_to_utf16_ratio(dn, ratio) for dn in dirnames]
        finaldirpath = os.path.join(*dirnames)
        recovered = remaining - sum(map(xlength, dirnames))
        filename_max = WIN_MAX_FILEPATH_LEN - WIN_MAX_DIRPATH_LEN - 1 + recovered
        computed[dirpath, reserved] = (finaldirpath, filename_max)
    filename = shorten(filename, filename_max)
    return os.path.join(finaldirpath, filename)

def _get_mount_point(target):
    if False:
        for i in range(10):
            print('nop')
    "Finds the target's mountpoint."
    try:
        mounts = _get_mount_point._mounts
    except AttributeError:
        mounts = _get_mount_point._mounts = {}
    try:
        mount = mounts[target]
    except KeyError:
        mount = target
        while mount and (not os.path.ismount(mount)):
            mount = os.path.dirname(mount)
        mounts[target] = mount
    return mount

def _get_filename_limit(target):
    if False:
        print('Hello World!')
    'Finds the maximum filename length under the given directory.'
    try:
        limits = _get_filename_limit._limits
    except AttributeError:
        limits = _get_filename_limit._limits = {}
    try:
        limit = limits[target]
    except KeyError:
        d = target
        while not os.path.exists(d):
            d = os.path.dirname(d)
        try:
            limit = os.statvfs(d).f_namemax
        except UnicodeEncodeError:
            limit = os.statvfs(d.encode(_io_encoding)).f_namemax
        limits[target] = limit
    return limit

def make_short_filename(basedir, relpath, win_shorten_path=False, relative_to=''):
    if False:
        return 10
    "Shorten a filename's path to proper limits.\n\n    basedir: Absolute path of the base directory where files will be moved.\n    relpath: File path, relative from the base directory.\n    win_shorten_path: Enforce 259 character limit for the path for Windows compatibility.\n    relative_to: An ancestor directory of basedir, against which win_shorten_path\n                 will be applied.\n    "
    try:
        basedir = os.path.abspath(basedir)
    except FileNotFoundError:
        basedir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.MusicLocation)
    relpath = os.path.normpath(relpath)
    if win_shorten_path and relative_to:
        relative_to = os.path.abspath(relative_to)
        assert basedir.startswith(relative_to) and basedir.split(relative_to)[1][:1] in (os.path.sep, ''), '`relative_to` must be an ancestor of `basedir`'
    relpath = os.path.join(*[part.strip() for part in relpath.split(os.path.sep)])
    if IS_WIN:
        if win_shorten_path:
            reserved = len(basedir)
            if not basedir.endswith(os.path.sep):
                reserved += 1
            return _make_win_short_filename(relpath, reserved)
        else:
            return shorten_path(relpath, WIN_MAX_NODE_LEN, mode=ShortenMode.UTF16)
    elif win_shorten_path:
        if not relative_to:
            relative_to = _get_mount_point(basedir)
            if relative_to == os.path.sep:
                relative_to = os.path.dirname(basedir)
        reserved = len(basedir) - len(relative_to) + 3 + 1
        relpath = _make_win_short_filename(relpath, reserved)
    if IS_MACOS:
        relpath = shorten_path(relpath, 255, mode=ShortenMode.UTF16_NFD)
    else:
        limit = _get_filename_limit(basedir)
        relpath = shorten_path(relpath, limit, mode=ShortenMode.BYTES)
    return relpath

def samefile_different_casing(path1, path2):
    if False:
        i = 10
        return i + 15
    'Returns True if path1 and path2 refer to the same file, but differ in casing of the filename.\n    Returns False if path1 and path2 refer to different files or there case is identical.\n    '
    path1 = os.path.normpath(path1)
    path2 = os.path.normpath(path2)
    if path1 == path2 or not os.path.exists(path1) or (not os.path.exists(path2)):
        return False
    dir1 = os.path.normcase(os.path.dirname(path1))
    dir2 = os.path.normcase(os.path.dirname(path2))
    try:
        dir1 = os.path.realpath(dir1)
        dir2 = os.path.realpath(dir2)
    except OSError:
        pass
    if dir1 != dir2 or not samefile(path1, path2):
        return False
    file1 = os.path.basename(path1)
    file2 = os.path.basename(path2)
    return file1 != file2 and file1.lower() == file2.lower()

def _make_unique_temp_name(target_path):
    if False:
        i = 10
        return i + 15
    i = 0
    target_dir = os.path.dirname(target_path)
    target_filename = os.path.basename(target_path)
    while True:
        temp_filename = '.%s%02d' % (target_filename[:-3], i)
        temp_path = os.path.join(target_dir, temp_filename)
        if not os.path.exists(temp_path):
            return temp_path
        i += 1

def _move_force_rename(source_path, target_path):
    if False:
        print('Hello World!')
    "Moves a file by renaming it first to a temporary name.\n    Ensure file casing changes on system's not natively supporting this.\n    "
    temp_path = _make_unique_temp_name(target_path)
    shutil.move(source_path, temp_path)
    os.rename(temp_path, target_path)

def move_ensure_casing(source_path, target_path):
    if False:
        for i in range(10):
            print('nop')
    'Moves a file from source_path to target_path.\n    If the move would result just in the name changing the case apply workarounds\n    for Linux and Windows to ensure the case change is applied on case-insensitive\n    file systems. Otherwise use shutil.move to move the file.\n    '
    source_path = os.path.normpath(source_path)
    target_path = os.path.normpath(target_path)
    if source_path == target_path:
        return
    if not IS_MACOS and samefile_different_casing(source_path, target_path):
        if IS_LINUX:
            _move_force_rename(source_path, target_path)
            return
        elif IS_WIN and win32api:
            shutil.move(source_path, target_path)
            try:
                actual_path = win32api.GetLongPathNameW(win32api.GetShortPathName(target_path))
                if samefile_different_casing(target_path, actual_path):
                    _move_force_rename(source_path, target_path)
            except pywintypes.error:
                pass
            return
    try:
        shutil.move(source_path, target_path)
    except shutil.SameFileError:
        pass

def make_save_path(path, win_compat=False, mac_compat=False):
    if False:
        return 10
    'Performs a couple of cleanups on a path to avoid side effects and incompatibilities.\n\n    - If win_compat is True, trailing dots in file and directory names will\n      be removed, as they are unsupported on Windows (dot is a delimiter for the file extension)\n    - Leading dots in file and directory names will be removed. These files cannot be properly\n      handled by Windows Explorer and on Unix like systems they count as hidden\n    - If mac_compat is True, normalize precomposed Unicode characters on macOS\n    - Remove unicode zero-width space (\\u200B) from path\n\n    Args:\n        path: filename or path to clean\n        win_compat: Set to True, if Windows compatibility is required\n        mac_compat: Set to True, if macOS compatibility is required\n\n    Returns: sanitized path\n    '
    if win_compat:
        path = path.replace('./', '_/').replace('.\\', '_\\')
        if path.endswith('.'):
            path = path[:-1] + '_'
    path = path.replace('/.', '/_').replace('\\.', '\\_')
    if path.startswith('.'):
        path = '_' + path[1:]
    if mac_compat:
        path = unicodedata.normalize('NFD', path)
    path = path.replace('\u200b', '')
    return path

def get_available_filename(new_path, old_path=None):
    if False:
        for i in range(10):
            print('nop')
    'Returns an available file name.\n\n    If new_path does already exist it appends " (N)" to the file name, where\n    N is an integer counted upwards.\n\n    If `old_path` is given the `new_path` is only changed if it does not point\n    to the same file location.\n\n    Args:\n      new_path: The requested file name for the file\n      old_path: The previous name of the file\n\n    Returns: A unique available file name.\n    '
    (tmp_filename, ext) = os.path.splitext(new_path)
    i = 1
    compare_old_path = old_path and os.path.exists(old_path)
    while os.path.exists(new_path) and (not compare_old_path or not samefile(old_path, new_path)):
        new_path = '%s (%d)%s' % (tmp_filename, i, ext)
        i += 1
    return new_path

def replace_extension(filename, new_ext):
    if False:
        print('Hello World!')
    'Replaces the extension in filename with new_ext.\n\n    If the file has no extension the extension is added.\n\n    Args:\n        filename: A file name\n        new_ext: New file extension\n\n    Returns: filename with replaced file extension\n    '
    (name, ext) = os.path.splitext(filename)
    return name + '.' + new_ext.lstrip('.')