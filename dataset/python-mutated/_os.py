import os
import tempfile
from os.path import abspath, dirname, join, normcase, sep
from pathlib import Path
from django.core.exceptions import SuspiciousFileOperation

def safe_join(base, *paths):
    if False:
        print('Hello World!')
    "\n    Join one or more path components to the base path component intelligently.\n    Return a normalized, absolute version of the final path.\n\n    Raise SuspiciousFileOperation if the final path isn't located inside of the\n    base path component.\n    "
    final_path = abspath(join(base, *paths))
    base_path = abspath(base)
    if not normcase(final_path).startswith(normcase(base_path + sep)) and normcase(final_path) != normcase(base_path) and (dirname(normcase(base_path)) != normcase(base_path)):
        raise SuspiciousFileOperation('The joined path ({}) is located outside of the base path component ({})'.format(final_path, base_path))
    return final_path

def symlinks_supported():
    if False:
        return 10
    '\n    Return whether or not creating symlinks are supported in the host platform\n    and/or if they are allowed to be created (e.g. on Windows it requires admin\n    permissions).\n    '
    with tempfile.TemporaryDirectory() as temp_dir:
        original_path = os.path.join(temp_dir, 'original')
        symlink_path = os.path.join(temp_dir, 'symlink')
        os.makedirs(original_path)
        try:
            os.symlink(original_path, symlink_path)
            supported = True
        except (OSError, NotImplementedError):
            supported = False
        return supported

def to_path(value):
    if False:
        i = 10
        return i + 15
    'Convert value to a pathlib.Path instance, if not already a Path.'
    if isinstance(value, Path):
        return value
    elif not isinstance(value, str):
        raise TypeError('Invalid path type: %s' % type(value).__name__)
    return Path(value)