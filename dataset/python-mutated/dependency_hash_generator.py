"""Utility Class for Getting Function or Layer Manifest Dependency Hashes"""
import pathlib
from typing import Any, Optional
from samcli.lib.build.workflow_config import get_workflow_config
from samcli.lib.utils.hash import file_checksum

class DependencyHashGenerator:
    _code_uri: str
    _base_dir: str
    _code_dir: str
    _runtime: str
    _manifest_path_override: Optional[str]
    _hash_generator: Any
    _calculated: bool
    _hash: Optional[str]

    def __init__(self, code_uri: str, base_dir: str, runtime: str, manifest_path_override: Optional[str]=None, hash_generator: Any=None):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ----------\n        code_uri : str\n            Relative path specified in the function/layer resource\n        base_dir : str\n            Absolute path which the function/layer dir is located\n        runtime : str\n            Runtime of the function/layer\n        manifest_path_override : Optional[str], optional\n            Override default manifest path for each runtime, by default None\n        hash_generator : Any, optional\n            Hash generation function. Can be hashlib.md5(), hashlib.sha256(), etc, by default None\n        '
        self._code_uri = code_uri
        self._base_dir = base_dir
        self._code_dir = str(pathlib.Path(self._base_dir, self._code_uri).resolve())
        self._runtime = runtime
        self._manifest_path_override = manifest_path_override
        self._hash_generator = hash_generator
        self._calculated = False
        self._hash = None

    def _calculate_dependency_hash(self) -> Optional[str]:
        if False:
            print('Hello World!')
        'Calculate the manifest file hash\n\n        Returns\n        -------\n        Optional[str]\n            Returns manifest hash. If manifest does not exist or not supported, None will be returned.\n        '
        if self._manifest_path_override:
            manifest_file = self._manifest_path_override
        else:
            config = get_workflow_config(self._runtime, self._code_dir, self._base_dir)
            manifest_file = config.manifest_name
        if not manifest_file:
            return None
        manifest_path = pathlib.Path(self._code_dir, manifest_file).resolve()
        if not manifest_path.is_file():
            return None
        return file_checksum(str(manifest_path), hash_generator=self._hash_generator)

    @property
    def hash(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        '\n        Returns\n        -------\n        Optional[str]\n            Hash for dependencies in the manifest.\n            If the manifest does not exist or not supported, this value will be None.\n        '
        if not self._calculated:
            self._hash = self._calculate_dependency_hash()
            self._calculated = True
        return self._hash