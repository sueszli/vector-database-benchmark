import os
import platform
if os.name == 'posix':
    if platform.system() == 'Darwin':
        DEFAULT_SOCKET_DIRS = ('/tmp',)
    else:
        DEFAULT_SOCKET_DIRS = ('/var/run', '/var/lib')
else:
    DEFAULT_SOCKET_DIRS = ()

def list_path(root_dir):
    if False:
        i = 10
        return i + 15
    'List directory if exists.\n\n    :param root_dir: str\n    :return: list\n\n    '
    res = []
    if os.path.isdir(root_dir):
        for name in os.listdir(root_dir):
            res.append(name)
    return res

def complete_path(curr_dir, last_dir):
    if False:
        return 10
    'Return the path to complete that matches the last entered component.\n\n    If the last entered component is ~, expanded path would not\n    match, so return all of the available paths.\n\n    :param curr_dir: str\n    :param last_dir: str\n    :return: str\n\n    '
    if not last_dir or curr_dir.startswith(last_dir):
        return curr_dir
    elif last_dir == '~':
        return os.path.join(last_dir, curr_dir)

def parse_path(root_dir):
    if False:
        for i in range(10):
            print('nop')
    'Split path into head and last component for the completer.\n\n    Also return position where last component starts.\n\n    :param root_dir: str path\n    :return: tuple of (string, string, int)\n\n    '
    (base_dir, last_dir, position) = ('', '', 0)
    if root_dir:
        (base_dir, last_dir) = os.path.split(root_dir)
        position = -len(last_dir) if last_dir else 0
    return (base_dir, last_dir, position)

def suggest_path(root_dir):
    if False:
        while True:
            i = 10
    'List all files and subdirectories in a directory.\n\n    If the directory is not specified, suggest root directory,\n    user directory, current and parent directory.\n\n    :param root_dir: string: directory to list\n    :return: list\n\n    '
    if not root_dir:
        return [os.path.abspath(os.sep), '~', os.curdir, os.pardir]
    if '~' in root_dir:
        root_dir = os.path.expanduser(root_dir)
    if not os.path.exists(root_dir):
        (root_dir, _) = os.path.split(root_dir)
    return list_path(root_dir)

def dir_path_exists(path):
    if False:
        return 10
    'Check if the directory path exists for a given file.\n\n    For example, for a file /home/user/.cache/mycli/log, check if\n    /home/user/.cache/mycli exists.\n\n    :param str path: The file path.\n    :return: Whether or not the directory path exists.\n\n    '
    return os.path.exists(os.path.dirname(path))

def guess_socket_location():
    if False:
        while True:
            i = 10
    'Try to guess the location of the default mysql socket file.'
    socket_dirs = filter(os.path.exists, DEFAULT_SOCKET_DIRS)
    for directory in socket_dirs:
        for (r, dirs, files) in os.walk(directory, topdown=True):
            for filename in files:
                (name, ext) = os.path.splitext(filename)
                if name.startswith('mysql') and ext in ('.socket', '.sock'):
                    return os.path.join(r, filename)
            dirs[:] = [d for d in dirs if d.startswith('mysql')]
    return None