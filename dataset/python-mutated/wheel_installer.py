from __future__ import annotations
import logging
import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from installer import install
from installer.destinations import SchemeDictionaryDestination
from installer.sources import WheelFile
from installer.sources import _WheelFileValidationError
from poetry.__version__ import __version__
from poetry.utils._compat import WINDOWS
logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from collections.abc import Collection
    from typing import BinaryIO
    from installer.records import RecordEntry
    from installer.scripts import LauncherKind
    from installer.utils import Scheme
    from poetry.utils.env import Env

class WheelDestination(SchemeDictionaryDestination):
    """ """

    def write_to_fs(self, scheme: Scheme, path: str, stream: BinaryIO, is_executable: bool) -> RecordEntry:
        if False:
            return 10
        from installer.records import Hash
        from installer.records import RecordEntry
        from installer.utils import copyfileobj_with_hashing
        from installer.utils import make_file_executable
        target_path = Path(self.scheme_dict[scheme]) / path
        if target_path.exists():
            logger.warning(f'Installing {target_path} over existing file')
        parent_folder = target_path.parent
        if not parent_folder.exists():
            parent_folder.mkdir(parents=True, exist_ok=True)
        with target_path.open('wb') as f:
            (hash_, size) = copyfileobj_with_hashing(stream, f, self.hash_algorithm)
        if is_executable:
            make_file_executable(target_path)
        return RecordEntry(path, Hash(self.hash_algorithm, hash_), size)

class WheelInstaller:

    def __init__(self, env: Env) -> None:
        if False:
            while True:
                i = 10
        self._env = env
        script_kind: LauncherKind
        if not WINDOWS:
            script_kind = 'posix'
        elif platform.uname()[4].startswith('arm'):
            script_kind = 'win-arm64' if sys.maxsize > 2 ** 32 else 'win-arm'
        else:
            script_kind = 'win-amd64' if sys.maxsize > 2 ** 32 else 'win-ia32'
        self._script_kind = script_kind
        self._bytecode_optimization_levels: Collection[int] = ()
        self.invalid_wheels: dict[Path, list[str]] = {}

    def enable_bytecode_compilation(self, enable: bool=True) -> None:
        if False:
            return 10
        self._bytecode_optimization_levels = (-1,) if enable else ()

    def install(self, wheel: Path) -> None:
        if False:
            while True:
                i = 10
        with WheelFile.open(wheel) as source:
            try:
                source.validate_record(validate_contents=False)
            except _WheelFileValidationError as e:
                self.invalid_wheels[wheel] = e.issues
            scheme_dict = self._env.paths.copy()
            scheme_dict['headers'] = str(Path(scheme_dict['include']) / source.distribution)
            destination = WheelDestination(scheme_dict, interpreter=str(self._env.python), script_kind=self._script_kind, bytecode_optimization_levels=self._bytecode_optimization_levels)
            install(source=source, destination=destination, additional_metadata={'INSTALLER': f'Poetry {__version__}'.encode()})