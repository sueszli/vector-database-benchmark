from __future__ import annotations
import itertools
import logging
import re
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import cast
from cleo.ui.progress_indicator import ProgressIndicator
from poetry.core.constraints.version import EmptyConstraint
from poetry.core.constraints.version import Version
from poetry.core.constraints.version import VersionRange
from poetry.core.packages.utils.utils import get_python_constraint_from_marker
from poetry.core.version.markers import AnyMarker
from poetry.core.version.markers import union as marker_union
from poetry.mixology.incompatibility import Incompatibility
from poetry.mixology.incompatibility_cause import DependencyCause
from poetry.mixology.incompatibility_cause import PythonCause
from poetry.mixology.term import Term
from poetry.packages import DependencyPackage
from poetry.packages.direct_origin import DirectOrigin
from poetry.packages.package_collection import PackageCollection
from poetry.puzzle.exceptions import OverrideNeeded
from poetry.repositories.exceptions import PackageNotFound
from poetry.utils.helpers import get_file_hash
if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Iterable
    from collections.abc import Iterator
    from pathlib import Path
    from cleo.io.io import IO
    from packaging.utils import NormalizedName
    from poetry.core.constraints.version import VersionConstraint
    from poetry.core.packages.dependency import Dependency
    from poetry.core.packages.directory_dependency import DirectoryDependency
    from poetry.core.packages.file_dependency import FileDependency
    from poetry.core.packages.package import Package
    from poetry.core.packages.url_dependency import URLDependency
    from poetry.core.packages.vcs_dependency import VCSDependency
    from poetry.core.version.markers import BaseMarker
    from poetry.repositories import RepositoryPool
    from poetry.utils.env import Env
logger = logging.getLogger(__name__)

class IncompatibleConstraintsError(Exception):
    """
    Exception when there are duplicate dependencies with incompatible constraints.
    """

    def __init__(self, package: Package, *dependencies: Dependency, with_sources: bool=False) -> None:
        if False:
            return 10
        constraints = []
        for dep in dependencies:
            constraint = dep.to_pep_508()
            if dep.is_direct_origin():
                constraint += f' ({dep.constraint})'
            if with_sources and dep.source_name:
                constraint += f' ; source={dep.source_name}'
            constraints.append(constraint)
        super().__init__(f'Incompatible constraints in requirements of {package}:\n' + '\n'.join(constraints))

class Indicator(ProgressIndicator):
    CONTEXT: str | None = None

    @staticmethod
    @contextmanager
    def context() -> Iterator[Callable[[str | None], None]]:
        if False:
            i = 10
            return i + 15

        def _set_context(context: str | None) -> None:
            if False:
                print('Hello World!')
            Indicator.CONTEXT = context
        yield _set_context
        _set_context(None)

    def _formatter_context(self) -> str:
        if False:
            while True:
                i = 10
        if Indicator.CONTEXT is None:
            return ' '
        else:
            return f' <c1>{Indicator.CONTEXT}</> '

    def _formatter_elapsed(self) -> str:
        if False:
            while True:
                i = 10
        assert self._start_time is not None
        elapsed = time.time() - self._start_time
        return f'{elapsed:.1f}s'

class Provider:
    UNSAFE_PACKAGES: ClassVar[set[str]] = set()

    def __init__(self, package: Package, pool: RepositoryPool, io: IO, *, installed: list[Package] | None=None, locked: list[Package] | None=None) -> None:
        if False:
            while True:
                i = 10
        self._package = package
        self._pool = pool
        self._direct_origin = DirectOrigin(self._pool.artifact_cache)
        self._io = io
        self._env: Env | None = None
        self._python_constraint = package.python_constraint
        self._is_debugging: bool = self._io.is_debug() or self._io.is_very_verbose()
        self._overrides: dict[DependencyPackage, dict[str, Dependency]] = {}
        self._deferred_cache: dict[Dependency, Package] = {}
        self._load_deferred = True
        self._source_root: Path | None = None
        self._installed_packages = installed if installed is not None else []
        self._direct_origin_packages: dict[str, Package] = {}
        self._locked: dict[NormalizedName, list[DependencyPackage]] = defaultdict(list)
        self._use_latest: Collection[NormalizedName] = []
        for package in locked or []:
            self._locked[package.name].append(DependencyPackage(package.to_dependency(), package))
        for dependency_packages in self._locked.values():
            dependency_packages.sort(key=lambda p: p.package.version, reverse=True)

    @property
    def pool(self) -> RepositoryPool:
        if False:
            i = 10
            return i + 15
        return self._pool

    @property
    def use_latest(self) -> Collection[NormalizedName]:
        if False:
            print('Hello World!')
        return self._use_latest

    def is_debugging(self) -> bool:
        if False:
            print('Hello World!')
        return self._is_debugging

    def set_overrides(self, overrides: dict[DependencyPackage, dict[str, Dependency]]) -> None:
        if False:
            return 10
        self._overrides = overrides

    def load_deferred(self, load_deferred: bool) -> None:
        if False:
            return 10
        self._load_deferred = load_deferred

    @contextmanager
    def use_source_root(self, source_root: Path) -> Iterator[Provider]:
        if False:
            while True:
                i = 10
        original_source_root = self._source_root
        self._source_root = source_root
        try:
            yield self
        finally:
            self._source_root = original_source_root

    @contextmanager
    def use_environment(self, env: Env) -> Iterator[Provider]:
        if False:
            for i in range(10):
                print('nop')
        original_python_constraint = self._python_constraint
        self._env = env
        self._python_constraint = Version.parse(env.marker_env['python_full_version'])
        try:
            yield self
        finally:
            self._env = None
            self._python_constraint = original_python_constraint

    @contextmanager
    def use_latest_for(self, names: Collection[NormalizedName]) -> Iterator[Provider]:
        if False:
            return 10
        self._use_latest = names
        try:
            yield self
        finally:
            self._use_latest = []

    @staticmethod
    def validate_package_for_dependency(dependency: Dependency, package: Package) -> None:
        if False:
            while True:
                i = 10
        if dependency.name != package.name:
            raise RuntimeError(f"The dependency name for {dependency.name} does not match the actual package's name: {package.name}")

    def search_for_installed_packages(self, dependency: Dependency) -> list[Package]:
        if False:
            print('Hello World!')
        '\n        Search for installed packages, when available, that satisfy the given\n        dependency.\n\n        This is useful when dealing with packages that are under development, not\n        published on package sources and/or only available via system installations.\n        '
        if not self._installed_packages:
            return []
        logger.debug('Falling back to installed packages to discover metadata for <c2>%s</>', dependency.complete_name)
        packages = [package for package in self._installed_packages if package.satisfies(dependency, ignore_source_type=True)]
        logger.debug('Found <c2>%d</> compatible packages for <c2>%s</>', len(packages), dependency.complete_name)
        return packages

    def search_for_direct_origin_dependency(self, dependency: Dependency) -> Package:
        if False:
            return 10
        package = self._deferred_cache.get(dependency)
        if package is not None:
            pass
        elif dependency.is_vcs():
            dependency = cast('VCSDependency', dependency)
            package = self._search_for_vcs(dependency)
        elif dependency.is_file():
            dependency = cast('FileDependency', dependency)
            package = self._search_for_file(dependency)
        elif dependency.is_directory():
            dependency = cast('DirectoryDependency', dependency)
            package = self._search_for_directory(dependency)
        elif dependency.is_url():
            dependency = cast('URLDependency', dependency)
            package = self._search_for_url(dependency)
        else:
            raise RuntimeError(f'{dependency}: unknown direct dependency type {dependency.source_type}')
        if dependency.is_vcs():
            dependency._source_reference = package.source_reference
            dependency._source_resolved_reference = package.source_resolved_reference
            dependency._source_subdirectory = package.source_subdirectory
        dependency._constraint = package.version
        dependency._pretty_constraint = package.version.text
        self._deferred_cache[dependency] = package
        return package

    def search_for(self, dependency: Dependency) -> list[DependencyPackage]:
        if False:
            return 10
        '\n        Search for the specifications that match the given dependency.\n\n        The specifications in the returned list will be considered in reverse\n        order, so the latest version ought to be last.\n        '
        if dependency.is_root:
            return PackageCollection(dependency, [self._package])
        if dependency.is_direct_origin():
            package = self.search_for_direct_origin_dependency(dependency)
            self._direct_origin_packages[dependency.name] = package
            return PackageCollection(dependency, [package])
        direct_origin_package = self._direct_origin_packages.get(dependency.name)
        if direct_origin_package and direct_origin_package.satisfies(dependency):
            packages = [direct_origin_package]
            return PackageCollection(dependency, packages)
        packages = self._pool.find_packages(dependency)
        packages.sort(key=lambda p: (not p.yanked, not p.is_prerelease() and (not dependency.allows_prereleases()), p.version), reverse=True)
        if not packages:
            packages = self.search_for_installed_packages(dependency)
        return PackageCollection(dependency, packages)

    def _search_for_vcs(self, dependency: VCSDependency) -> Package:
        if False:
            return 10
        '\n        Search for the specifications that match the given VCS dependency.\n\n        Basically, we clone the repository in a temporary directory\n        and get the information we need by checking out the specified reference.\n        '
        package = self._direct_origin.get_package_from_vcs(dependency.vcs, dependency.source, branch=dependency.branch, tag=dependency.tag, rev=dependency.rev, subdirectory=dependency.source_subdirectory, source_root=self._source_root or (self._env.path.joinpath('src') if self._env else None))
        self.validate_package_for_dependency(dependency=dependency, package=package)
        package.develop = dependency.develop
        return package

    def _search_for_file(self, dependency: FileDependency) -> Package:
        if False:
            i = 10
            return i + 15
        dependency.validate(raise_error=True)
        package = self._direct_origin.get_package_from_file(dependency.full_path)
        self.validate_package_for_dependency(dependency=dependency, package=package)
        if dependency.base is not None:
            package.root_dir = dependency.base
        package.files = [{'file': dependency.path.name, 'hash': 'sha256:' + get_file_hash(dependency.full_path)}]
        return package

    def _search_for_directory(self, dependency: DirectoryDependency) -> Package:
        if False:
            while True:
                i = 10
        dependency.validate(raise_error=True)
        package = self._direct_origin.get_package_from_directory(dependency.full_path)
        self.validate_package_for_dependency(dependency=dependency, package=package)
        package.develop = dependency.develop
        if dependency.base is not None:
            package.root_dir = dependency.base
        return package

    def _search_for_url(self, dependency: URLDependency) -> Package:
        if False:
            return 10
        package = self._direct_origin.get_package_from_url(dependency.url)
        self.validate_package_for_dependency(dependency=dependency, package=package)
        for extra in dependency.extras:
            if extra in package.extras:
                for dep in package.extras[extra]:
                    dep.activate()
                for extra_dep in package.extras[extra]:
                    package.add_dependency(extra_dep)
        return package

    def _get_dependencies_with_overrides(self, dependencies: list[Dependency], package: DependencyPackage) -> list[Dependency]:
        if False:
            for i in range(10):
                print('nop')
        overrides = self._overrides.get(package, {})
        _dependencies = []
        overridden = []
        for dep in dependencies:
            if dep.name in overrides:
                if dep.name in overridden:
                    continue
                if not overrides[dep.name].constraint.is_empty():
                    _dependencies.append(overrides[dep.name])
                overridden.append(dep.name)
                continue
            _dependencies.append(dep)
        return _dependencies

    def incompatibilities_for(self, dependency_package: DependencyPackage) -> list[Incompatibility]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns incompatibilities that encapsulate a given package's dependencies,\n        or that it can't be safely selected.\n\n        If multiple subsequent versions of this package have the same\n        dependencies, this will return incompatibilities that reflect that. It\n        won't return incompatibilities that have already been returned by a\n        previous call to _incompatibilities_for().\n        "
        package = dependency_package.package
        if package.is_root():
            dependencies = package.all_requires
        else:
            dependencies = package.requires
            if not package.python_constraint.allows_all(self._python_constraint):
                transitive_python_constraint = get_python_constraint_from_marker(dependency_package.dependency.transitive_marker)
                intersection = package.python_constraint.intersect(transitive_python_constraint)
                difference = transitive_python_constraint.difference(intersection)
                difference = difference.intersect(self._python_constraint)
                if transitive_python_constraint.is_any() or self._python_constraint.intersect(dependency_package.dependency.python_constraint).is_empty() or intersection.is_empty() or (not difference.is_empty()):
                    return [Incompatibility([Term(package.to_dependency(), True)], PythonCause(package.python_versions, str(self._python_constraint)))]
        _dependencies = [dep for dep in dependencies if dep.name not in self.UNSAFE_PACKAGES and self._python_constraint.allows_any(dep.python_constraint) and (not self._env or dep.marker.validate(self._env.marker_env))]
        dependencies = self._get_dependencies_with_overrides(_dependencies, dependency_package)
        return [Incompatibility([Term(package.to_dependency(), True), Term(dep, False)], DependencyCause()) for dep in dependencies]

    def complete_package(self, dependency_package: DependencyPackage) -> DependencyPackage:
        if False:
            i = 10
            return i + 15
        package = dependency_package.package
        dependency = dependency_package.dependency
        if package.is_root():
            dependency_package = dependency_package.clone()
            package = dependency_package.package
            dependency = dependency_package.dependency
            requires = package.all_requires
        elif package.is_direct_origin():
            requires = package.requires
        else:
            try:
                dependency_package = DependencyPackage(dependency, self._pool.package(package.pretty_name, package.version, extras=list(dependency.extras), repository_name=dependency.source_name))
            except PackageNotFound as e:
                try:
                    dependency_package = next((DependencyPackage(dependency, pkg) for pkg in self.search_for_installed_packages(dependency)))
                except StopIteration:
                    raise e from e
            package = dependency_package.package
            dependency = dependency_package.dependency
            requires = package.requires
        optional_dependencies = []
        _dependencies = []
        if dependency.extras:
            for extra in dependency.extras:
                if extra not in package.extras:
                    continue
                optional_dependencies += [d.name for d in package.extras[extra]]
            dependency_package = dependency_package.with_features(list(dependency.extras))
            package = dependency_package.package
            dependency = dependency_package.dependency
            new_dependency = package.without_features().to_dependency()
            if not new_dependency.source_name and dependency.source_name:
                new_dependency.source_name = dependency.source_name
            _dependencies.append(new_dependency)
        for dep in requires:
            if not self._python_constraint.allows_any(dep.python_constraint):
                continue
            if dep.name in self.UNSAFE_PACKAGES:
                continue
            if self._env and (not dep.marker.validate(self._env.marker_env)):
                continue
            if not package.is_root() and (dep.is_optional() and dep.name not in optional_dependencies or (dep.in_extras and (not set(dep.in_extras).intersection(dependency.extras)))):
                continue
            _dependencies.append(dep)
        if self._load_deferred:
            for dep in _dependencies:
                if dep.is_direct_origin():
                    locked = self.get_locked(dep)
                    if locked is not None and locked.package.is_same_package_as(dep):
                        continue
                    self.search_for_direct_origin_dependency(dep)
        dependencies = self._get_dependencies_with_overrides(_dependencies, dependency_package)
        duplicates: dict[str, list[Dependency]] = defaultdict(list)
        for dep in dependencies:
            duplicates[dep.complete_name].append(dep)
        dependencies = []
        for (dep_name, deps) in duplicates.items():
            if len(deps) == 1:
                dependencies.append(deps[0])
                continue
            self.debug(f'<debug>Duplicate dependencies for {dep_name}</debug>')
            active_extras = None if package.is_root() else dependency.extras
            deps = self._resolve_overlapping_markers(package, deps, active_extras)
            if len(deps) == 1:
                self.debug(f'<debug>Merging requirements for {dep_name}</debug>')
                dependencies.append(deps[0])
                continue

            def fmt_warning(d: Dependency) -> str:
                if False:
                    i = 10
                    return i + 15
                dependency_marker = d.marker if not d.marker.is_any() else '*'
                return f'<c1>{d.name}</c1> <fg=default>(<c2>{d.pretty_constraint}</c2>)</> with markers <b>{dependency_marker}</b>'
            warnings = ', '.join((fmt_warning(d) for d in deps[:-1]))
            warnings += f' and {fmt_warning(deps[-1])}'
            self.debug(f'<warning>Different requirements found for {warnings}.</warning>')
            overrides = []
            overrides_marker_intersection: BaseMarker = AnyMarker()
            for dep_overrides in self._overrides.values():
                for dep in dep_overrides.values():
                    overrides_marker_intersection = overrides_marker_intersection.intersect(dep.marker)
            for dep in deps:
                if not overrides_marker_intersection.intersect(dep.marker).is_empty():
                    current_overrides = self._overrides.copy()
                    package_overrides = current_overrides.get(dependency_package, {}).copy()
                    package_overrides.update({dep.name: dep})
                    current_overrides.update({dependency_package: package_overrides})
                    overrides.append(current_overrides)
            if overrides:
                raise OverrideNeeded(*overrides)
        clean_dependencies = []
        for dep in dependencies:
            if not dependency.transitive_marker.without_extras().is_any():
                transitive_marker_intersection = dependency.transitive_marker.without_extras().intersect(dep.marker.without_extras())
                if transitive_marker_intersection.is_empty():
                    continue
                dep.transitive_marker = transitive_marker_intersection
            if not dependency.python_constraint.is_any():
                python_constraint_intersection = dep.python_constraint.intersect(dependency.python_constraint)
                if python_constraint_intersection.is_empty():
                    continue
            clean_dependencies.append(dep)
        package = package.with_dependency_groups([], only=True)
        dependency_package = DependencyPackage(dependency, package)
        for dep in clean_dependencies:
            package.add_dependency(dep)
        return dependency_package

    def get_locked(self, dependency: Dependency) -> DependencyPackage | None:
        if False:
            for i in range(10):
                print('nop')
        if dependency.name in self._use_latest:
            return None
        locked = self._locked.get(dependency.name, [])
        for dependency_package in locked:
            package = dependency_package.package
            if package.satisfies(dependency):
                return DependencyPackage(dependency, package)
        return None

    def debug(self, message: str, depth: int=0) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not (self._io.is_very_verbose() or self._io.is_debug()):
            return
        if message.startswith('fact:'):
            if 'depends on' in message:
                m = re.match('fact: (.+?) depends on (.+?) \\((.+?)\\)', message)
                if m is None:
                    raise ValueError(f'Unable to parse fact: {message}')
                m2 = re.match('(.+?) \\((.+?)\\)', m.group(1))
                if m2:
                    name = m2.group(1)
                    version = f' (<c2>{m2.group(2)}</c2>)'
                else:
                    name = m.group(1)
                    version = ''
                message = f'<fg=blue>fact</>: <c1>{name}</c1>{version} depends on <c1>{m.group(2)}</c1> (<c2>{m.group(3)}</c2>)'
            elif ' is ' in message:
                message = re.sub('fact: (.+) is (.+)', '<fg=blue>fact</>: <c1>\\1</c1> is <c2>\\2</c2>', message)
            else:
                message = re.sub('(?<=: )(.+?) \\((.+?)\\)', '<c1>\\1</c1> (<c2>\\2</c2>)', message)
                message = f"<fg=blue>fact</>: {message.split('fact: ')[1]}"
        elif message.startswith('selecting '):
            message = re.sub('selecting (.+?) \\((.+?)\\)', '<fg=blue>selecting</> <c1>\\1</c1> (<c2>\\2</c2>)', message)
        elif message.startswith('derived:'):
            m = re.match('derived: (.+?) \\((.+?)\\)$', message)
            if m:
                message = f'<fg=blue>derived</>: <c1>{m.group(1)}</c1> (<c2>{m.group(2)}</c2>)'
            else:
                message = f"<fg=blue>derived</>: <c1>{message.split('derived: ')[1]}</c1>"
        elif message.startswith('conflict:'):
            m = re.match('conflict: (.+?) depends on (.+?) \\((.+?)\\)', message)
            if m:
                m2 = re.match('(.+?) \\((.+?)\\)', m.group(1))
                if m2:
                    name = m2.group(1)
                    version = f' (<c2>{m2.group(2)}</c2>)'
                else:
                    name = m.group(1)
                    version = ''
                message = f'<fg=red;options=bold>conflict</>: <c1>{name}</c1>{version} depends on <c1>{m.group(2)}</c1> (<c2>{m.group(3)}</c2>)'
            else:
                message = f"<fg=red;options=bold>conflict</>: {message.split('conflict: ')[1]}"
        message = message.replace('! ', '<error>!</error> ')
        if self.is_debugging():
            debug_info = str(message)
            debug_info = '\n'.join([f'<debug>{str(depth).rjust(4)}:</debug> {s}' for s in debug_info.split('\n')]) + '\n'
            self._io.write(debug_info)

    def _group_by_source(self, dependencies: Iterable[Dependency]) -> list[list[Dependency]]:
        if False:
            while True:
                i = 10
        '\n        Takes a list of dependencies and returns a list of groups of dependencies,\n        each group containing all dependencies from the same source.\n        '
        groups: list[list[Dependency]] = []
        for dep in dependencies:
            for group in groups:
                if dep.is_same_source_as(group[0]) and dep.source_name == group[0].source_name:
                    group.append(dep)
                    break
            else:
                groups.append([dep])
        return groups

    def _merge_dependencies_by_constraint(self, dependencies: Iterable[Dependency]) -> list[Dependency]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Merge dependencies with the same constraint\n        by building a union of their markers.\n\n        For instance, if we have:\n           - foo (>=2.0) ; python_version >= "3.6" and python_version < "3.7"\n           - foo (>=2.0) ; python_version >= "3.7"\n        we can avoid two overrides by merging them to:\n           - foo (>=2.0) ; python_version >= "3.6"\n        '
        dep_groups = self._group_by_source(dependencies)
        merged_dependencies = []
        for group in dep_groups:
            by_constraint: dict[VersionConstraint, list[Dependency]] = defaultdict(list)
            for dep in group:
                by_constraint[dep.constraint].append(dep)
            for deps in by_constraint.values():
                dep = deps[0]
                if len(deps) > 1:
                    new_markers = (dep.marker for dep in deps)
                    dep.marker = marker_union(*new_markers)
                merged_dependencies.append(dep)
        return merged_dependencies

    def _is_relevant_marker(self, marker: BaseMarker, active_extras: Collection[NormalizedName] | None) -> bool:
        if False:
            print('Hello World!')
        "\n        A marker is relevant if\n        - it is not empty\n        - allowed by the project's python constraint\n        - allowed by active extras of the dependency (not relevant for root package)\n        - allowed by the environment (only during installation)\n        "
        return not marker.is_empty() and self._python_constraint.allows_any(get_python_constraint_from_marker(marker)) and (active_extras is None or marker.validate({'extra': active_extras})) and (not self._env or marker.validate(self._env.marker_env))

    def _resolve_overlapping_markers(self, package: Package, dependencies: list[Dependency], active_extras: Collection[NormalizedName] | None) -> list[Dependency]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert duplicate dependencies with potentially overlapping markers\n        into duplicate dependencies with mutually exclusive markers.\n\n        Therefore, the intersections of all combinations of markers and inverted markers\n        have to be calculated. If such an intersection is relevant (not empty, etc.),\n        the intersection of all constraints, whose markers were not inverted is built\n        and a new dependency with the calculated version constraint and marker is added.\n        (The marker of such a dependency does not overlap with the marker\n        of any other new dependency.)\n        '
        dependencies = self._merge_dependencies_by_constraint(dependencies)
        new_dependencies = []
        for uses in itertools.product([True, False], repeat=len(dependencies)):
            markers = (dep.marker if use else dep.marker.invert() for (use, dep) in sorted(zip(uses, dependencies), key=lambda ud: ud[0], reverse=True))
            used_marker_intersection: BaseMarker = AnyMarker()
            for m in markers:
                used_marker_intersection = used_marker_intersection.intersect(m)
            if not self._is_relevant_marker(used_marker_intersection, active_extras):
                continue
            constraint: VersionConstraint = VersionRange()
            specific_source_dependency = None
            used_dependencies = list(itertools.compress(dependencies, uses))
            for dep in used_dependencies:
                if dep.is_direct_origin() or dep.source_name:
                    if specific_source_dependency and (not dep.is_same_source_as(specific_source_dependency) or dep.source_name != specific_source_dependency.source_name):
                        raise IncompatibleConstraintsError(package, dep, specific_source_dependency, with_sources=True)
                    specific_source_dependency = dep
                constraint = constraint.intersect(dep.constraint)
            if constraint.is_empty():
                raise IncompatibleConstraintsError(package, *used_dependencies)
            if not any(uses):
                constraint = EmptyConstraint()
                used_dependencies = dependencies
            new_dep = (specific_source_dependency if specific_source_dependency else used_dependencies[0]).with_constraint(constraint)
            new_dep.marker = used_marker_intersection
            new_dependencies.append(new_dep)
        return self._merge_dependencies_by_constraint(new_dependencies)