import logging
import os
import pathlib
import typing as t
from colorama import Fore
from tmuxp.cli.utils import tmuxp_echo
from tmuxp.types import StrPath
from tmuxp.workspace.constants import VALID_WORKSPACE_DIR_FILE_EXTENSIONS
logger = logging.getLogger(__name__)
if t.TYPE_CHECKING:
    from typing_extensions import TypeAlias
    ValidExtensions: TypeAlias = t.Literal['.yml', '.yaml', '.json']

def is_workspace_file(filename: str, extensions: t.Union['ValidExtensions', t.List['ValidExtensions'], None]=None) -> bool:
    if False:
        print('Hello World!')
    "\n    Return True if file has a valid workspace file type.\n\n    Parameters\n    ----------\n    filename : str\n        filename to check (e.g. ``mysession.json``).\n    extensions : str or list\n        filetypes to check (e.g. ``['.yaml', '.json']``).\n\n    Returns\n    -------\n    bool\n    "
    if extensions is None:
        extensions = ['.yml', '.yaml', '.json']
    extensions = [extensions] if isinstance(extensions, str) else extensions
    return any((filename.endswith(e) for e in extensions))

def in_dir(workspace_dir: t.Union[pathlib.Path, str, None]=None, extensions: t.Optional[t.List['ValidExtensions']]=None) -> t.List[str]:
    if False:
        while True:
            i = 10
    "\n    Return a list of workspace_files in ``workspace_dir``.\n\n    Parameters\n    ----------\n    workspace_dir : str\n        directory to search\n    extensions : list\n        filetypes to check (e.g. ``['.yaml', '.json']``).\n\n    Returns\n    -------\n    list\n    "
    if workspace_dir is None:
        workspace_dir = os.path.expanduser('~/.tmuxp')
    if extensions is None:
        extensions = ['.yml', '.yaml', '.json']
    workspace_files = [filename for filename in os.listdir(workspace_dir) if is_workspace_file(filename, extensions) and (not filename.startswith('.'))]
    return workspace_files

def in_cwd() -> t.List[str]:
    if False:
        for i in range(10):
            print('nop')
    "\n    Return list of workspace_files in current working directory.\n\n    If filename is ``.tmuxp.py``, ``.tmuxp.json``, ``.tmuxp.yaml``.\n\n    Returns\n    -------\n    list\n        workspace_files in current working directory\n\n    Examples\n    --------\n    >>> sorted(in_cwd())\n    ['.tmuxp.json', '.tmuxp.yaml']\n    "
    workspace_files = [filename for filename in os.listdir(os.getcwd()) if filename.startswith('.tmuxp') and is_workspace_file(filename)]
    return workspace_files

def get_workspace_dir() -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Return tmuxp workspace directory.\n\n    ``TMUXP_CONFIGDIR`` environmental variable has precedence if set. We also\n    evaluate XDG default directory from XDG_CONFIG_HOME environmental variable\n    if set or its default. Then the old default ~/.tmuxp is returned for\n    compatibility.\n\n    Returns\n    -------\n    str :\n        absolute path to tmuxp config directory\n    '
    paths = []
    if 'TMUXP_CONFIGDIR' in os.environ:
        paths.append(os.environ['TMUXP_CONFIGDIR'])
    if 'XDG_CONFIG_HOME' in os.environ:
        paths.append(os.path.join(os.environ['XDG_CONFIG_HOME'], 'tmuxp'))
    else:
        paths.append('~/.config/tmuxp/')
    paths.append('~/.tmuxp')
    for path in paths:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            return path
    return path

def find_workspace_file(workspace_file: StrPath, workspace_dir: t.Optional[StrPath]=None) -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the real config path or raise an exception.\n\n    If workspace file is directory, scan for .tmuxp.{yaml,yml,json} in directory. If\n    one or more found, it will warn and pick the first.\n\n    If workspace file is ".", "./" or None, it will scan current directory.\n\n    If workspace file is has no path and only a filename, e.g. "my_workspace.yaml" it\n    will search workspace dir.\n\n    If workspace file has no path and no extension, e.g. "my_workspace", it will scan\n    for file name with yaml, yml and json. If multiple exist, it will warn and pick the\n    first.\n\n    Parameters\n    ----------\n    workspace_file : str\n        workspace file, valid examples:\n\n        - a file name, my_workspace.yaml\n        - relative path, ../my_workspace.yaml or ../project\n        - a period, .\n    '
    if not workspace_dir:
        workspace_dir = get_workspace_dir()
    path = os.path
    (exists, join, isabs) = (path.exists, path.join, path.isabs)
    (dirname, normpath, splitext) = (path.dirname, path.normpath, path.splitext)
    cwd = os.getcwd()
    is_name = False
    file_error = None
    workspace_file = os.path.expanduser(workspace_file)
    if is_pure_name(workspace_file):
        is_name = True
    elif not isabs(workspace_file) or len(dirname(workspace_file)) > 1 or workspace_file == '.' or (workspace_file == '') or (workspace_file == './'):
        workspace_file = normpath(join(cwd, workspace_file))
    if path.isdir(workspace_file) or not splitext(workspace_file)[1]:
        if is_name:
            candidates = [x for x in [f'{join(workspace_dir, workspace_file)}{ext}' for ext in VALID_WORKSPACE_DIR_FILE_EXTENSIONS] if exists(x)]
            if not len(candidates):
                file_error = 'workspace-file not found in workspace dir (yaml/yml/json) %s for name' % workspace_dir
        else:
            candidates = [x for x in [join(workspace_file, ext) for ext in ['.tmuxp.yaml', '.tmuxp.yml', '.tmuxp.json']] if exists(x)]
            if len(candidates) > 1:
                tmuxp_echo(Fore.RED + 'Multiple .tmuxp.{yml,yaml,json} workspace_files in %s' % dirname(workspace_file) + Fore.RESET)
                tmuxp_echo('This is undefined behavior, use only one. Use file names e.g. myproject.json, coolproject.yaml. You can load them by filename.')
            elif not len(candidates):
                file_error = 'No tmuxp files found in directory'
        if len(candidates):
            workspace_file = candidates[0]
    elif not exists(workspace_file):
        file_error = 'file not found'
    if file_error:
        raise FileNotFoundError(file_error, workspace_file)
    return workspace_file

def is_pure_name(path: str) -> bool:
    if False:
        print('Hello World!')
    '\n    Return True if path is a name and not a file path.\n\n    Parameters\n    ----------\n    path : str\n        Path (can be absolute, relative, etc.)\n\n    Returns\n    -------\n    bool\n        True if path is a name of workspace in workspace dir, not file path.\n    '
    return not os.path.isabs(path) and len(os.path.dirname(path)) == 0 and (not os.path.splitext(path)[1]) and (path != '.') and (path != '')