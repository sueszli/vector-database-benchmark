"""Utilities for working with pytype source files."""
import os
import re
from pytype.platform_utils import path_utils

class NoSuchDirectory(Exception):
    pass

def _pytype_source_dir():
    if False:
        print('Hello World!')
    'The base directory of the pytype source tree.'
    res = path_utils.dirname(__file__)
    if path_utils.basename(res) == '__pycache__':
        res = path_utils.dirname(res)
    return res

def get_full_path(path):
    if False:
        print('Hello World!')
    'Full path to a file or directory within the pytype source tree.\n\n  Arguments:\n    path: An absolute or relative path.\n\n  Returns:\n    path for absolute paths.\n    full path resolved relative to pytype/ for relative paths.\n  '
    if path_utils.isabs(path):
        return path
    else:
        return path_utils.join(_pytype_source_dir(), path)

def load_text_file(filename):
    if False:
        print('Hello World!')
    return _load_data_file(filename, text=True)

def load_binary_file(filename):
    if False:
        while True:
            i = 10
    return _load_data_file(filename, text=False)

def _load_data_file(filename, text):
    if False:
        i = 10
        return i + 15
    'Get the contents of a data file from the pytype installation.\n\n  Arguments:\n    filename: the path, relative to "pytype/"\n    text: whether to load the file as text or bytes.\n  Returns:\n    The contents of the file as a bytestring\n  Raises:\n    IOError: if file not found\n  '
    path = filename if path_utils.isabs(filename) else get_full_path(filename)
    loader = globals().get('__loader__', None)
    if loader:
        data = loader.get_data(path)
        if text:
            return re.sub('\r\n?', '\n', data.decode('utf-8'))
        return data
    with open(path, 'r' if text else 'rb') as fi:
        return fi.read()

def list_files(basedir):
    if False:
        while True:
            i = 10
    'List files in the directory rooted at |basedir|.'
    if not path_utils.isdir(basedir):
        raise NoSuchDirectory(basedir)
    directories = ['']
    while directories:
        d = directories.pop()
        for basename in os.listdir(path_utils.join(basedir, d)):
            filename = path_utils.join(d, basename)
            if path_utils.isdir(path_utils.join(basedir, filename)):
                directories.append(filename)
            elif path_utils.exists(path_utils.join(basedir, filename)):
                yield filename

def list_pytype_files(suffix):
    if False:
        return 10
    'Recursively get the contents of a directory in the pytype installation.\n\n  This reports files in said directory as well as all subdirectories of it.\n\n  Arguments:\n    suffix: the path, relative to "pytype/"\n  Yields:\n    The filenames, relative to pytype/{suffix}\n  Raises:\n    NoSuchDirectory: if the directory doesn\'t exist.\n  '
    assert not suffix.endswith('/')
    loader = globals().get('__loader__', None)
    try:
        filenames = loader.get_zipfile().namelist()
    except AttributeError:
        yield from list_files(get_full_path(suffix))
    else:
        for filename in filenames:
            directory = 'pytype/' + suffix + '/'
            try:
                i = filename.rindex(directory)
            except ValueError:
                pass
            else:
                yield filename[i + len(directory):]