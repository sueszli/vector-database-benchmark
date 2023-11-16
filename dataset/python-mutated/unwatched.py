"""
This module describes a configuration option to modify how Pyre watches
some files in a project.
"""
import dataclasses
from typing import Dict
from . import exceptions

@dataclasses.dataclass(frozen=True)
class UnwatchedFiles:
    root: str
    checksum_path: str

    @staticmethod
    def from_json(json: Dict[str, object]) -> 'UnwatchedFiles':
        if False:
            for i in range(10):
                print('nop')
        root = json.get('root', None)
        if root is None:
            raise exceptions.InvalidConfiguration('Missing `root` field in UnwatchedFiles')
        if not isinstance(root, str):
            raise exceptions.InvalidConfiguration('`root` field in UnwatchedFiles must be a string')
        checksum_path = json.get('checksum_path', None)
        if checksum_path is None:
            raise exceptions.InvalidConfiguration('Missing `checksum_path` field in UnwatchedFiles')
        if not isinstance(checksum_path, str):
            raise exceptions.InvalidConfiguration('`checksum_path` field in UnwatchedFiles must be a string')
        return UnwatchedFiles(root=root, checksum_path=checksum_path)

    def to_json(self) -> Dict[str, str]:
        if False:
            return 10
        return {'root': self.root, 'checksum_path': self.checksum_path}

@dataclasses.dataclass(frozen=True)
class UnwatchedDependency:
    change_indicator: str
    files: UnwatchedFiles

    @staticmethod
    def from_json(json: Dict[str, object]) -> 'UnwatchedDependency':
        if False:
            while True:
                i = 10
        change_indicator = json.get('change_indicator', None)
        if change_indicator is None:
            raise exceptions.InvalidConfiguration('Missing `change_indicator` field in UnwatchedDependency')
        if not isinstance(change_indicator, str):
            raise exceptions.InvalidConfiguration('`change_indicator` field in UnwatchedDependency must be a string')
        files_json = json.get('files', None)
        if files_json is None:
            raise exceptions.InvalidConfiguration('Missing `files` field in UnwatchedDependency')
        if not isinstance(files_json, dict):
            raise exceptions.InvalidConfiguration('`files` field in UnwatchedDependency must be a dict')
        return UnwatchedDependency(change_indicator=change_indicator, files=UnwatchedFiles.from_json(files_json))

    def to_json(self) -> Dict[str, object]:
        if False:
            i = 10
            return i + 15
        return {'change_indicator': str(self.change_indicator), 'files': self.files.to_json()}