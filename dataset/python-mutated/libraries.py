from typing import Dict, Mapping
from dagster._core.utils import check_dagster_package_version
from ..version import __version__

class DagsterLibraryRegistry:
    _libraries: Dict[str, str] = {'dagster': __version__}

    @classmethod
    def register(cls, name: str, version: str):
        if False:
            while True:
                i = 10
        check_dagster_package_version(name, version)
        cls._libraries[name] = version

    @classmethod
    def get(cls) -> Mapping[str, str]:
        if False:
            return 10
        return cls._libraries.copy()