from functools import cached_property
from packaging.specifiers import SpecifierSet
from packaging.specifiers import Version
from packaging.version import InvalidVersion
from .syft_repo import SyftRepo
__all__ = ['SyftVersion', 'InvalidVersion']

class SyftVersion:

    def __init__(self, version: str):
        if False:
            i = 10
            return i + 15
        self._ver: Version = self._resolve(version)

    @property
    def version(self) -> Version:
        if False:
            for i in range(10):
                print('nop')
        'Returns the underlying Version object'
        return self._ver

    @property
    def release_tag(self) -> str:
        if False:
            i = 10
            return i + 15
        'Returns the Github release version string (e.g. v0.8.2)'
        return f'v{self.version}'

    @cached_property
    def docker_tag(self) -> str:
        if False:
            print('Hello World!')
        'Returns the docker version/tag (e.g. 0.8.2-beta.26)'
        manifest = SyftRepo.get_manifest(self.release_tag)
        return manifest['dockerTag']

    def match(self, ver_spec: str, prereleases: bool=True) -> bool:
        if False:
            while True:
                i = 10
        _spec = SpecifierSet(ver_spec, prereleases=prereleases)
        return _spec.contains(self.version)

    def valid_version(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.release_tag in SyftRepo.all_versions()

    def _resolve(self, version: str) -> Version:
        if False:
            while True:
                i = 10
        if version == 'latest':
            version = SyftRepo.latest_version()
        if version == 'latest-beta':
            version = SyftRepo.latest_version(beta=True)
        return Version(version)

    def __str__(self) -> str:
        if False:
            return 10
        return str(self._ver)