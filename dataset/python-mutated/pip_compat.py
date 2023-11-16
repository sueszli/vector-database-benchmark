from __future__ import annotations
import optparse
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Iterator, Set, cast
import pip
from pip._internal.cache import WheelCache
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import BaseDistribution
from pip._internal.metadata.pkg_resources import Distribution as _PkgResourcesDist
from pip._internal.models.direct_url import DirectUrl
from pip._internal.network.session import PipSession
from pip._internal.req import InstallRequirement
from pip._internal.req import parse_requirements as _parse_requirements
from pip._internal.req.constructors import install_req_from_parsed_requirement
from pip._vendor.packaging.version import parse as parse_version
from pip._vendor.pkg_resources import Requirement
PIP_VERSION = tuple(map(int, parse_version(pip.__version__).base_version.split('.')))
if TYPE_CHECKING:
    from pip._internal.metadata.importlib import Distribution as _ImportLibDist

@dataclass(frozen=True)
class Distribution:
    key: str
    version: str
    requires: Iterable[Requirement]
    direct_url: DirectUrl | None

    @classmethod
    def from_pip_distribution(cls, dist: BaseDistribution) -> Distribution:
        if False:
            print('Hello World!')
        if isinstance(dist, _PkgResourcesDist):
            return cls._from_pkg_resources(dist)
        else:
            return cls._from_importlib(dist)

    @classmethod
    def _from_pkg_resources(cls, dist: _PkgResourcesDist) -> Distribution:
        if False:
            while True:
                i = 10
        return cls(dist._dist.key, dist._dist.version, dist._dist.requires(), dist.direct_url)

    @classmethod
    def _from_importlib(cls, dist: _ImportLibDist) -> Distribution:
        if False:
            return 10
        "Mimics pkg_resources.Distribution.requires for the case of no\n        extras. This doesn't fulfill that API's `extras` parameter but\n        satisfies the needs of pip-tools."
        reqs = (Requirement.parse(req) for req in dist._dist.requires or ())
        requires = [req for req in reqs if not req.marker or req.marker.evaluate({'extra': None})]
        return cls(dist._dist.name, dist._dist.version, requires, dist.direct_url)

def parse_requirements(filename: str, session: PipSession, finder: PackageFinder | None=None, options: optparse.Values | None=None, constraint: bool=False, isolated: bool=False) -> Iterator[InstallRequirement]:
    if False:
        return 10
    for parsed_req in _parse_requirements(filename, session, finder=finder, options=options, constraint=constraint):
        yield install_req_from_parsed_requirement(parsed_req, isolated=isolated)

def create_wheel_cache(cache_dir: str, format_control: str | None=None) -> WheelCache:
    if False:
        return 10
    kwargs: dict[str, str | None] = {'cache_dir': cache_dir}
    if PIP_VERSION[:2] <= (23, 0):
        kwargs['format_control'] = format_control
    return WheelCache(**kwargs)

def get_dev_pkgs() -> set[str]:
    if False:
        for i in range(10):
            print('nop')
    if PIP_VERSION[:2] <= (23, 1):
        from pip._internal.commands.freeze import DEV_PKGS
        return cast(Set[str], DEV_PKGS)
    from pip._internal.commands.freeze import _dev_pkgs
    return cast(Set[str], _dev_pkgs())