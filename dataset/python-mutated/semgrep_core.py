import importlib.resources
import os
import shutil
import sys
from pathlib import Path
from typing import Optional
from semgrep.verbose_logging import getLogger
logger = getLogger(__name__)
VERSION_STAMP_FILENAME = 'pro-installed-by.txt'

def compute_executable_path(exec_name: str) -> Optional[str]:
    if False:
        i = 10
        return i + 15
    '\n    Determine full executable path if full path is needed to run it.\n\n    Return None if no executable found\n    '
    try:
        with importlib.resources.path('semgrep.bin', exec_name) as path:
            if path.is_file():
                return str(path)
    except FileNotFoundError as e:
        logger.debug(f'Failed to open resource {exec_name}: {e}.')
    which_exec = shutil.which(exec_name)
    if which_exec is not None:
        return which_exec
    relative_path = os.path.join(os.path.dirname(sys.executable), exec_name)
    if os.path.isfile(relative_path):
        return relative_path
    return None

class SemgrepCore:
    _SEMGREP_PATH_: Optional[str] = None
    _PRO_PATH_: Optional[str] = None

    @classmethod
    def executable_path(cls) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Return the path to the semgrep stand-alone executable binary,\n        *not* the Python module.  This is intended for unusual cases\n        that the module plumbing is not set up to handle.\n\n        Raise Exception if the executable is not found.\n        '
        ret = compute_executable_path('semgrep-core')
        if ret is None:
            raise Exception('Could not locate semgrep-core binary')
        return ret

    @classmethod
    def path(cls) -> Path:
        if False:
            while True:
                i = 10
        '\n        Return the path to the semgrep stand-alone program.  Raise Exception if\n        not found.\n        '
        if cls._SEMGREP_PATH_ is None:
            cls._SEMGREP_PATH_ = cls.executable_path()
        return Path(cls._SEMGREP_PATH_)

    @classmethod
    def pro_path(cls) -> Optional[Path]:
        if False:
            i = 10
            return i + 15
        if cls._PRO_PATH_ is None:
            cls._PRO_PATH_ = compute_executable_path('semgrep-core-proprietary')
        return Path(cls._PRO_PATH_) if cls._PRO_PATH_ is not None else None

    @classmethod
    def pro_version_stamp_path(cls) -> Path:
        if False:
            i = 10
            return i + 15
        return cls.path().parent / VERSION_STAMP_FILENAME