import os
import shutil
from os import PathLike
from edk2toollib.utility_functions import GetHostInfo

def get_codeql_db_path(workspace: PathLike, package: str, target: str, new_path: bool=True) -> str:
    if False:
        i = 10
        return i + 15
    'Return the CodeQL database path for this build.\n\n    Args:\n        workspace (PathLike): The workspace path.\n        package (str): The package name (e.g. "MdeModulePkg")\n        target (str): The target (e.g. "DEBUG")\n        new_path (bool, optional): Whether to create a new database path or\n                                   return an existing path. Defaults to True.\n\n    Returns:\n        str: The absolute path to the CodeQL database directory.\n    '
    codeql_db_dir_name = 'codeql-db-' + package + '-' + target
    codeql_db_dir_name = codeql_db_dir_name.lower()
    codeql_db_path = os.path.join('Build', codeql_db_dir_name)
    codeql_db_path = os.path.join(workspace, codeql_db_path)
    i = 0
    while os.path.isdir(f"{codeql_db_path + '-%s' % i}"):
        i += 1
    if not new_path:
        if i == 0:
            return None
        else:
            i -= 1
    return codeql_db_path + f'-{i}'

def get_codeql_cli_path() -> str:
    if False:
        while True:
            i = 10
    'Return the current CodeQL CLI path.\n\n    Returns:\n        str: The absolute path to the CodeQL CLI application to use for\n             this build.\n    '
    codeql_path = None
    if 'STUART_CODEQL_PATH' in os.environ:
        codeql_path = os.environ['STUART_CODEQL_PATH']
        if GetHostInfo().os == 'Windows':
            codeql_path = os.path.join(codeql_path, 'codeql.exe')
        else:
            codeql_path = os.path.join(codeql_path, 'codeql')
        if not os.path.isfile(codeql_path):
            codeql_path = None
    if not codeql_path:
        codeql_path = shutil.which('codeql')
    return codeql_path