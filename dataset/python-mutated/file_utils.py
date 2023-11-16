"""Mycroft file utils.

This module contains functions handling mycroft resource files and things like
accessing and curating mycroft's cache.
"""
import os
import psutil
from stat import S_ISREG, ST_MTIME, ST_MODE, ST_SIZE
import tempfile
import xdg.BaseDirectory
import mycroft.configuration
from .log import LOG

def resolve_resource_file(res_name):
    if False:
        for i in range(10):
            print('nop')
    "Convert a resource into an absolute filename.\n\n    Resource names are in the form: 'filename.ext'\n    or 'path/filename.ext'\n\n    The system wil look for $XDG_DATA_DIRS/mycroft/res_name first\n    (defaults to ~/.local/share/mycroft/res_name), and if not found will\n    look at /opt/mycroft/res_name, then finally it will look for res_name\n    in the 'mycroft/res' folder of the source code package.\n\n    Example:\n        With mycroft running as the user 'bob', if you called\n        ``resolve_resource_file('snd/beep.wav')``\n        it would return either:\n        '$XDG_DATA_DIRS/mycroft/beep.wav',\n        '/home/bob/.mycroft/snd/beep.wav' or\n        '/opt/mycroft/snd/beep.wav' or\n        '.../mycroft/res/snd/beep.wav'\n        where the '...' is replaced by the path\n        where the package has been installed.\n\n    Args:\n        res_name (str): a resource path/name\n\n    Returns:\n        (str) path to resource or None if no resource found\n    "
    config = mycroft.configuration.Configuration.get()
    if os.path.isfile(res_name):
        return res_name
    for conf_dir in xdg.BaseDirectory.load_data_paths('mycroft'):
        filename = os.path.join(conf_dir, res_name)
        if os.path.isfile(filename):
            return filename
    filename = os.path.join(os.path.expanduser('~'), '.mycroft', res_name)
    if os.path.isfile(filename):
        return filename
    data_dir = os.path.join(os.path.expanduser(config['data_dir']), 'res')
    filename = os.path.expanduser(os.path.join(data_dir, res_name))
    if os.path.isfile(filename):
        return filename
    filename = os.path.join(os.path.dirname(__file__), '..', 'res', res_name)
    filename = os.path.abspath(os.path.normpath(filename))
    if os.path.isfile(filename):
        return filename
    return None

def read_stripped_lines(filename):
    if False:
        print('Hello World!')
    'Read a file and return a list of stripped lines.\n\n    Args:\n        filename (str): path to file to read.\n\n    Returns:\n        (list) list of lines stripped from leading and ending white chars.\n    '
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                yield line

def read_dict(filename, div='='):
    if False:
        for i in range(10):
            print('nop')
    "Read file into dict.\n\n    A file containing:\n        foo = bar\n        baz = bog\n\n    results in a dict\n    {\n        'foo': 'bar',\n        'baz': 'bog'\n    }\n\n    Args:\n        filename (str):   path to file\n        div (str): deviders between dict keys and values\n\n    Returns:\n        (dict) generated dictionary\n    "
    d = {}
    with open(filename, 'r') as f:
        for line in f:
            (key, val) = line.split(div)
            d[key.strip()] = val.strip()
    return d

def mb_to_bytes(size):
    if False:
        for i in range(10):
            print('nop')
    'Takes a size in MB and returns the number of bytes.\n\n    Args:\n        size(int/float): size in Mega Bytes\n\n    Returns:\n        (int/float) size in bytes\n    '
    return size * 1024 * 1024

def _get_cache_entries(directory):
    if False:
        for i in range(10):
            print('nop')
    'Get information tuple for all regular files in directory.\n\n    Args:\n        directory (str): path to directory to check\n\n    Returns:\n        (tuple) (modification time, size, filepath)\n    '
    entries = (os.path.join(directory, fn) for fn in os.listdir(directory))
    entries = ((os.stat(path), path) for path in entries)
    return ((stat[ST_MTIME], stat[ST_SIZE], path) for (stat, path) in entries if S_ISREG(stat[ST_MODE]))

def _delete_oldest(entries, bytes_needed):
    if False:
        for i in range(10):
            print('nop')
    'Delete files with oldest modification date until space is freed.\n\n    Args:\n        entries (tuple): file + file stats tuple\n        bytes_needed (int): disk space that needs to be freed\n\n    Returns:\n        (list) all removed paths\n    '
    deleted_files = []
    space_freed = 0
    for (moddate, fsize, path) in sorted(entries):
        try:
            os.remove(path)
            space_freed += fsize
            deleted_files.append(path)
        except Exception:
            pass
        if space_freed > bytes_needed:
            break
    return deleted_files

def curate_cache(directory, min_free_percent=5.0, min_free_disk=50):
    if False:
        for i in range(10):
            print('nop')
    'Clear out the directory if needed.\n\n    The curation will only occur if both the precentage and actual disk space\n    is below the limit. This assumes all the files in the directory can be\n    deleted as freely.\n\n    Args:\n        directory (str): directory path that holds cached files\n        min_free_percent (float): percentage (0.0-100.0) of drive to keep free,\n                                  default is 5% if not specified.\n        min_free_disk (float): minimum allowed disk space in MB, default\n                               value is 50 MB if not specified.\n    '
    deleted_files = []
    space = psutil.disk_usage(directory)
    min_free_disk = mb_to_bytes(min_free_disk)
    percent_free = 100.0 - space.percent
    if percent_free < min_free_percent and space.free < min_free_disk:
        LOG.info('Low diskspace detected, cleaning cache')
        bytes_needed = (min_free_percent - percent_free) / 100.0 * space.total
        bytes_needed = int(bytes_needed + 1.0)
        entries = _get_cache_entries(directory)
        deleted_files = _delete_oldest(entries, bytes_needed)
    return deleted_files

def get_cache_directory(domain=None):
    if False:
        for i in range(10):
            print('nop')
    'Get a directory for caching data.\n\n    This directory can be used to hold temporary caches of data to\n    speed up performance.  This directory will likely be part of a\n    small RAM disk and may be cleared at any time.  So code that\n    uses these cached files must be able to fallback and regenerate\n    the file.\n\n    Args:\n        domain (str): The cache domain.  Basically just a subdirectory.\n\n    Returns:\n        (str) a path to the directory where you can cache data\n    '
    config = mycroft.configuration.Configuration.get()
    directory = config.get('cache_path')
    if not directory:
        directory = get_temp_path('mycroft', 'cache')
    return ensure_directory_exists(directory, domain)

def ensure_directory_exists(directory, domain=None, permissions=511):
    if False:
        print('Hello World!')
    'Create a directory and give access rights to all\n\n    Args:\n        directory (str): Root directory\n        domain (str): Domain. Basically a subdirectory to prevent things like\n                      overlapping signal filenames.\n        rights (int): Directory permissions (default is 0o777)\n\n    Returns:\n        (str) a path to the directory\n    '
    if domain:
        directory = os.path.join(directory, domain)
    directory = os.path.normpath(directory)
    directory = os.path.expanduser(directory)
    if not os.path.isdir(directory):
        try:
            save = os.umask(0)
            os.makedirs(directory, permissions)
        except OSError:
            LOG.warning('Failed to create: ' + directory)
        finally:
            os.umask(save)
    return directory

def create_file(filename):
    if False:
        while True:
            i = 10
    'Create the file filename and create any directories needed\n\n    Args:\n        filename: Path to the file to be created\n    '
    ensure_directory_exists(os.path.dirname(filename), permissions=509)
    with open(filename, 'w') as f:
        f.write('')

def get_temp_path(*args):
    if False:
        print('Hello World!')
    "Generate a valid path in the system temp directory.\n\n    This method accepts one or more strings as arguments. The arguments are\n    joined and returned as a complete path inside the systems temp directory.\n    Importantly, this will not create any directories or files.\n\n    Example usage: get_temp_path('mycroft', 'audio', 'example.wav')\n    Will return the equivalent of: '/tmp/mycroft/audio/example.wav'\n\n    Args:\n        path_element (str): directories and/or filename\n\n    Returns:\n        (str) a valid path in the systems temp directory\n    "
    try:
        path = os.path.join(tempfile.gettempdir(), *args)
    except TypeError:
        raise TypeError('Could not create a temp path, get_temp_path() only accepts Strings')
    return path