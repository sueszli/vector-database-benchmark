import os
import os.path
import shutil
from PyQt6.QtCore import QStandardPaths
JUNK_FILES = {'.DS_Store', 'desktop.ini', 'Desktop.ini', 'Thumbs.db'}
PROTECTED_DIRECTORIES = set()
for location in QStandardPaths.StandardLocation:
    for path in QStandardPaths.standardLocations(location):
        try:
            PROTECTED_DIRECTORIES.add(os.path.realpath(path))
        except OSError:
            pass

class SkipRemoveDir(Exception):
    pass

def is_empty_dir(path, ignored_files=None):
    if False:
        return 10
    '\n    Checks if a directory is considered empty.\n\n    Args:\n        path: Path to directory to check.\n        ignored_files: List of files to ignore. I only some of those files is\n                       inside the directory it is still considered empty.\n\n    Returns:\n        True if path is considered an empty directory\n        False if path is not considered an empty directory\n\n    Raises:\n        NotADirectoryError: path is not a directory\n    '
    if ignored_files is None:
        ignored_files = JUNK_FILES
    return not set(os.listdir(path)) - set(ignored_files)

def rm_empty_dir(path):
    if False:
        while True:
            i = 10
    '\n    Delete a directory if it is considered empty by is_empty_dir and if it\n    is not considered a special directory (e.g. the users home dir or ~/Desktop).\n\n    Args:\n        path: Path to directory to remove.\n\n    Raises:\n        NotADirectoryError: path is not a directory\n        SkipRemoveDir: path was not deleted because it is either not empty\n                       or considered a special directory.\n    '
    if os.path.realpath(path) in PROTECTED_DIRECTORIES:
        raise SkipRemoveDir('%s is a protected directory' % path)
    elif not is_empty_dir(path):
        raise SkipRemoveDir('%s is not empty' % path)
    else:
        shutil.rmtree(path)