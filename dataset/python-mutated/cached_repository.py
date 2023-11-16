from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from packaging.utils import canonicalize_name
from poetry.core.constraints.version import parse_constraint
from poetry.config.config import Config
from poetry.repositories.repository import Repository
from poetry.utils.cache import FileCache
if TYPE_CHECKING:
    from packaging.utils import NormalizedName
    from poetry.core.constraints.version import Version
    from poetry.core.packages.package import Package
    from poetry.inspection.info import PackageInfo

class CachedRepository(Repository, ABC):
    CACHE_VERSION = parse_constraint('2.0.0')

    def __init__(self, name: str, disable_cache: bool=False, config: Config | None=None) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(name)
        self._disable_cache = disable_cache
        self._cache_dir = (config or Config.create()).repository_cache_directory / name
        self._release_cache: FileCache[dict[str, Any]] = FileCache(path=self._cache_dir)

    @abstractmethod
    def _get_release_info(self, name: NormalizedName, version: Version) -> dict[str, Any]:
        if False:
            i = 10
            return i + 15
        ...

    def get_release_info(self, name: NormalizedName, version: Version) -> PackageInfo:
        if False:
            i = 10
            return i + 15
        '\n        Return the release information given a package name and a version.\n\n        The information is returned from the cache if it exists\n        or retrieved from the remote server.\n        '
        from poetry.inspection.info import PackageInfo
        if self._disable_cache:
            return PackageInfo.load(self._get_release_info(name, version))
        cached = self._release_cache.remember(f'{name}:{version}', lambda : self._get_release_info(name, version))
        cache_version = cached.get('_cache_version', '0.0.0')
        if parse_constraint(cache_version) != self.CACHE_VERSION:
            self._log(f'The cache for {name} {version} is outdated. Refreshing.', level='debug')
            cached = self._get_release_info(name, version)
            self._release_cache.put(f'{name}:{version}', cached)
        return PackageInfo.load(cached)

    def package(self, name: str, version: Version, extras: list[str] | None=None) -> Package:
        if False:
            print('Hello World!')
        return self.get_release_info(canonicalize_name(name), version).to_package(name=name, extras=extras)