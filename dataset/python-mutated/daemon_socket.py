"""
This module has logic defining the paths at which Pyre daemons may be
listening.
"""
from __future__ import annotations
import hashlib
import tempfile
from pathlib import Path
from typing import Iterable
from .identifiers import PyreFlavor
MD5_LENGTH = 32

def get_md5(identifier_string: str) -> str:
    if False:
        return 10
    identifier_bytes = identifier_string.encode('utf-8')
    return hashlib.md5(identifier_bytes).hexdigest()

def _get_socket_path_in_root(socket_root: Path, project_identifier: str, flavor: PyreFlavor=PyreFlavor.CLASSIC) -> Path:
    if False:
        return 10
    "\n    Determine where the server socket file is located. We can't directly use\n    `log_directory` because of the ~100 character length limit on Unix socket\n    file paths.\n    "
    project_hash = get_md5(project_identifier)
    flavor_suffix = flavor.path_suffix()
    return socket_root / f'pyre_server_{project_hash}{flavor_suffix}.sock'

def get_default_socket_root() -> Path:
    if False:
        i = 10
        return i + 15
    return Path(tempfile.gettempdir())

def get_socket_path(project_identifier: str, flavor: PyreFlavor) -> Path:
    if False:
        i = 10
        return i + 15
    return _get_socket_path_in_root(get_default_socket_root(), project_identifier, flavor)

def socket_file_glob_pattern() -> str:
    if False:
        print('Hello World!')
    md5_hash_pattern = '[0-9a-f]' * MD5_LENGTH
    return f'pyre_server_{md5_hash_pattern}*.sock'

def find_socket_files(socket_root: Path) -> Iterable[Path]:
    if False:
        i = 10
        return i + 15
    return socket_root.glob(socket_file_glob_pattern())