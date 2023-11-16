"""
Module for simple file-based key-value database management.

This module provides a simple file-based key-value database system, where keys are
represented as filenames and values are the contents of these files. The primary class,
DB, is responsible for the CRUD operations on the database. Additionally, the module
provides a dataclass `DBs` that encapsulates multiple `DB` instances to represent different
databases like memory, logs, preprompts, etc.

Functions:
    archive(dbs: DBs) -> None:
        Archives the memory and workspace databases, moving their contents to
        the archive database with a timestamp.

Classes:
    DB:
        A simple key-value store implemented as a file-based system.

    DBs:
        A dataclass containing multiple DB instances representing different databases.

Imports:
    - datetime: For timestamp generation when archiving.
    - shutil: For moving directories during archiving.
    - dataclasses: For the DBs dataclass definition.
    - pathlib: For path manipulations.
    - typing: For type annotations.
"""
import datetime
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union
from gpt_engineer.data.supported_languages import SUPPORTED_LANGUAGES

class FileRepository:
    """
    A file-based key-value store where keys correspond to filenames and values to file contents.

    This class provides an interface to a file-based database, leveraging file operations to
    facilitate CRUD-like interactions. It allows for quick checks on the existence of keys,
    retrieval of values based on keys, and setting new key-value pairs.

    Attributes
    ----------
    path : Path
        The directory path where the database files are stored.

    Methods
    -------
    __contains__(key: str) -> bool:
        Check if a file (key) exists in the database.

    __getitem__(key: str) -> str:
        Retrieve the content of a file (value) based on its name (key).

    get(key: str, default: Optional[Any] = None) -> Any:
        Fetch content of a file or return a default value if it doesn't exist.

    __setitem__(key: Union[str, Path], val: str):
        Set or update the content of a file in the database.

    Note:
    -----
    Care should be taken when choosing keys (filenames) to avoid potential
    security issues, such as directory traversal. The class implements some checks
    for this but it's essential to validate inputs from untrusted sources.
    """
    'A simple key-value store, where keys are filenames and values are file contents.'

    def __init__(self, path: Union[str, Path]):
        if False:
            while True:
                i = 10
        '\n        Initialize the DB class.\n\n        Parameters\n        ----------\n        path : Union[str, Path]\n            The path to the directory where the database files are stored.\n        '
        self.path: Path = Path(path).absolute()
        self.path.mkdir(parents=True, exist_ok=True)

    def __contains__(self, key: str) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Check if a file with the specified name exists in the database.\n\n        Parameters\n        ----------\n        key : str\n            The name of the file to check.\n\n        Returns\n        -------\n        bool\n            True if the file exists, False otherwise.\n        '
        return (self.path / key).is_file()

    def __getitem__(self, key: str) -> str:
        if False:
            print('Hello World!')
        '\n        Get the content of a file in the database.\n\n        Parameters\n        ----------\n        key : str\n            The name of the file to get the content of.\n\n        Returns\n        -------\n        str\n            The content of the file.\n\n        Raises\n        ------\n        KeyError\n            If the file does not exist in the database.\n        '
        full_path = self.path / key
        if not full_path.is_file():
            raise KeyError(f"File '{key}' could not be found in '{self.path}'")
        with full_path.open('r', encoding='utf-8') as f:
            return f.read()

    def get(self, key: str, default: Optional[Any]=None) -> Any:
        if False:
            while True:
                i = 10
        '\n        Get the content of a file in the database, or a default value if the file does not exist.\n\n        Parameters\n        ----------\n        key : str\n            The name of the file to get the content of.\n        default : any, optional\n            The default value to return if the file does not exist, by default None.\n\n        Returns\n        -------\n        any\n            The content of the file, or the default value if the file does not exist.\n        '
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key: Union[str, Path], val: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Set the content of a file in the database.\n\n        Parameters\n        ----------\n        key : Union[str, Path]\n            The name of the file to set the content of.\n        val : str\n            The content to set.\n\n        Raises\n        ------\n        TypeError\n            If val is not string.\n        '
        if str(key).startswith('../'):
            raise ValueError(f'File name {key} attempted to access parent path.')
        assert isinstance(val, str), 'val must be str'
        full_path = self.path / key
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(val, encoding='utf-8')

    def __delitem__(self, key: Union[str, Path]) -> None:
        if False:
            return 10
        '\n        Delete a file or directory in the database.\n\n        Parameters\n        ----------\n        key : Union[str, Path]\n            The name of the file or directory to delete.\n\n        Raises\n        ------\n        KeyError\n            If the file or directory does not exist in the database.\n        '
        item_path = self.path / key
        if not item_path.exists():
            raise KeyError(f"Item '{key}' could not be found in '{self.path}'")
        if item_path.is_file():
            item_path.unlink()
        elif item_path.is_dir():
            shutil.rmtree(item_path)

    def _supported_files(self, directory: Path) -> str:
        if False:
            return 10
        valid_extensions = {ext for lang in SUPPORTED_LANGUAGES for ext in lang['extensions']}
        file_paths = [str(item) for item in sorted(directory.rglob('*')) if item.is_file() and item.suffix in valid_extensions]
        return '\n'.join(file_paths)

    def _all_files(self, directory: Path) -> str:
        if False:
            return 10
        file_paths = [str(item) for item in sorted(directory.rglob('*')) if item.is_file()]
        return '\n'.join(file_paths)

    def to_path_list_string(self, supported_code_files_only: bool=False) -> str:
        if False:
            while True:
                i = 10
        '\n        Returns directory as a list of file paths. Useful for passing to the LLM where it needs to understand the wider context of files available for reference.\n        '
        if supported_code_files_only:
            return self._supported_files(self.path)
        else:
            return self._all_files(self.path)

@dataclass
class FileRepositories:
    memory: FileRepository
    logs: FileRepository
    preprompts: FileRepository
    input: FileRepository
    workspace: FileRepository
    archive: FileRepository
    project_metadata: FileRepository

def archive(dbs: FileRepositories) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Archive the memory and workspace databases.\n\n    Parameters\n    ----------\n    dbs : DBs\n        The databases to archive.\n    '
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    shutil.move(str(dbs.memory.path), str(dbs.archive.path / timestamp / dbs.memory.path.name))
    exclude_dir = '.gpteng'
    items_to_copy = [f for f in dbs.workspace.path.iterdir() if not f.name == exclude_dir]
    for item_path in items_to_copy:
        destination_path = dbs.archive.path / timestamp / item_path.name
        if item_path.is_file():
            shutil.copy2(item_path, destination_path)
        elif item_path.is_dir():
            shutil.copytree(item_path, destination_path)
    return []