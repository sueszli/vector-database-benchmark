import contextlib
import errno
import fnmatch
import io
import os
from pathlib import Path
from streamlit import env_util, util
from streamlit.string_util import is_binary_string
CONFIG_FOLDER_NAME = '.streamlit'
APP_STATIC_FOLDER_NAME = 'static'

def get_encoded_file_data(data, encoding='auto'):
    if False:
        i = 10
        return i + 15
    "Coerce bytes to a BytesIO or a StringIO.\n\n    Parameters\n    ----------\n    data : bytes\n    encoding : str\n\n    Returns\n    -------\n    BytesIO or StringIO\n        If the file's data is in a well-known textual format (or if the encoding\n        parameter is set), return a StringIO. Otherwise, return BytesIO.\n\n    "
    if encoding == 'auto':
        if is_binary_string(data):
            encoding = None
        else:
            encoding = 'utf-8'
    if encoding:
        return io.StringIO(data.decode(encoding))
    return io.BytesIO(data)

@contextlib.contextmanager
def streamlit_read(path, binary=False):
    if False:
        for i in range(10):
            print('nop')
    "Opens a context to read this file relative to the streamlit path.\n\n    For example:\n\n    with streamlit_read('foo.txt') as foo:\n        ...\n\n    opens the file `.streamlit/foo.txt`\n\n    path   - the path to write to (within the streamlit directory)\n    binary - set to True for binary IO\n    "
    filename = get_streamlit_file_path(path)
    if os.stat(filename).st_size == 0:
        raise util.Error('Read zero byte file: "%s"' % filename)
    mode = 'r'
    if binary:
        mode += 'b'
    with open(os.path.join(CONFIG_FOLDER_NAME, path), mode) as handle:
        yield handle

@contextlib.contextmanager
def streamlit_write(path, binary=False):
    if False:
        return 10
    "Opens a file for writing within the streamlit path, and\n    ensuring that the path exists. For example:\n\n        with streamlit_write('foo/bar.txt') as bar:\n            ...\n\n    opens the file .streamlit/foo/bar.txt for writing,\n    creating any necessary directories along the way.\n\n    path   - the path to write to (within the streamlit directory)\n    binary - set to True for binary IO\n    "
    mode = 'w'
    if binary:
        mode += 'b'
    path = get_streamlit_file_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, mode) as handle:
            yield handle
    except OSError as e:
        msg = ['Unable to write file: %s' % os.path.abspath(path)]
        if e.errno == errno.EINVAL and env_util.IS_DARWIN:
            msg.append('Python is limited to files below 2GB on OSX. See https://bugs.python.org/issue24658')
        raise util.Error('\n'.join(msg))

def get_static_dir():
    if False:
        while True:
            i = 10
    'Get the folder where static HTML/JS/CSS files live.'
    dirname = os.path.dirname(os.path.normpath(__file__))
    return os.path.normpath(os.path.join(dirname, 'static'))

def get_app_static_dir(main_script_path: str) -> str:
    if False:
        while True:
            i = 10
    'Get the folder where app static files live'
    main_script_path = Path(main_script_path)
    static_dir = main_script_path.parent / APP_STATIC_FOLDER_NAME
    return os.path.abspath(static_dir)

def get_streamlit_file_path(*filepath) -> str:
    if False:
        return 10
    "Return the full path to a file in ~/.streamlit.\n\n    This doesn't guarantee that the file (or its directory) exists.\n    "
    home = os.path.expanduser('~')
    if home is None:
        raise RuntimeError('No home directory.')
    return os.path.join(home, CONFIG_FOLDER_NAME, *filepath)

def get_project_streamlit_file_path(*filepath):
    if False:
        return 10
    "Return the full path to a filepath in ${CWD}/.streamlit.\n\n    This doesn't guarantee that the file (or its directory) exists.\n    "
    return os.path.join(os.getcwd(), CONFIG_FOLDER_NAME, *filepath)

def file_is_in_folder_glob(filepath, folderpath_glob) -> bool:
    if False:
        return 10
    'Test whether a file is in some folder with globbing support.\n\n    Parameters\n    ----------\n    filepath : str\n        A file path.\n    folderpath_glob: str\n        A path to a folder that may include globbing.\n\n    '
    if not folderpath_glob.endswith('*'):
        if folderpath_glob.endswith('/'):
            folderpath_glob += '*'
        else:
            folderpath_glob += '/*'
    file_dir = os.path.dirname(filepath) + '/'
    return fnmatch.fnmatch(file_dir, folderpath_glob)

def get_directory_size(directory: str) -> int:
    if False:
        for i in range(10):
            print('nop')
    'Return the size of a directory in bytes.'
    total_size = 0
    for (dirpath, _, filenames) in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def file_in_pythonpath(filepath) -> bool:
    if False:
        i = 10
        return i + 15
    'Test whether a filepath is in the same folder of a path specified in the PYTHONPATH env variable.\n\n\n    Parameters\n    ----------\n    filepath : str\n        An absolute file path.\n\n    Returns\n    -------\n    boolean\n        True if contained in PYTHONPATH, False otherwise. False if PYTHONPATH is not defined or empty.\n\n    '
    pythonpath = os.environ.get('PYTHONPATH', '')
    if len(pythonpath) == 0:
        return False
    absolute_paths = [os.path.abspath(path) for path in pythonpath.split(os.pathsep)]
    return any((file_is_in_folder_glob(os.path.normpath(filepath), path) for path in absolute_paths))