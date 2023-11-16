from __future__ import annotations
import collections
import functools
import time
from typing import TYPE_CHECKING
from typing import Optional
from typing import Tuple
from poetry.core.packages.dependency import Dependency
from poetry.mixology.failure import SolveFailure
from poetry.mixology.incompatibility import Incompatibility
from poetry.mixology.incompatibility_cause import ConflictCause
from poetry.mixology.incompatibility_cause import NoVersionsCause
from poetry.mixology.incompatibility_cause import RootCause
from poetry.mixology.partial_solution import PartialSolution
from poetry.mixology.result import SolverResult
from poetry.mixology.set_relation import SetRelation
from poetry.mixology.term import Term
if TYPE_CHECKING:
    from poetry.core.packages.project_package import ProjectPackage
    from poetry.packages import DependencyPackage
    from poetry.puzzle.provider import Provider
_conflict = object()
DependencyCacheKey = Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]

class DependencyCache:
    """
    A cache of the valid dependencies.

    The key observation here is that during the search - except at backtracking
    - once we have decided that a dependency is invalid, we never need check it
    again.
    """

    def __init__(self, provider: Provider) -> None:
        if False:
            while True:
                i = 10
        self._provider = provider
        self._cache: dict[DependencyCacheKey, list[list[DependencyPackage]]] = collections.defaultdict(list)
        self._cached_dependencies_by_level: dict[int, list[DependencyCacheKey]] = collections.defaultdict(list)
        self._search_for_cached = functools.lru_cache(maxsize=128)(self._search_for)

    def _search_for(self, dependency: Dependency, key: DependencyCacheKey) -> list[DependencyPackage]:
        if False:
            for i in range(10):
                print('nop')
        cache_entries = self._cache[key]
        if cache_entries:
            packages = [p for p in cache_entries[-1] if dependency.constraint.allows(p.package.version)]
        else:
            packages = None
        if not packages:
            packages = self._provider.search_for(dependency)
        return packages

    def search_for(self, dependency: Dependency, decision_level: int) -> list[DependencyPackage]:
        if False:
            while True:
                i = 10
        key = (dependency.complete_name, dependency.source_type, dependency.source_url, dependency.source_reference, dependency.source_subdirectory)
        packages = self._search_for_cached(dependency, key)
        if not self._cache[key] or self._cache[key][-1] is not packages:
            self._cache[key].append(packages)
            self._cached_dependencies_by_level[decision_level].append(key)
        return packages

    def clear_level(self, level: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        if level in self._cached_dependencies_by_level:
            self._search_for_cached.cache_clear()
            for key in self._cached_dependencies_by_level.pop(level):
                self._cache[key].pop()

class VersionSolver:
    """
    The version solver that finds a set of package versions that satisfy the
    root package's dependencies.

    See https://github.com/dart-lang/pub/tree/master/doc/solver.md for details
    on how this solver works.
    """

    def __init__(self, root: ProjectPackage, provider: Provider) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._root = root
        self._provider = provider
        self._dependency_cache = DependencyCache(provider)
        self._incompatibilities: dict[str, list[Incompatibility]] = {}
        self._contradicted_incompatibilities: set[Incompatibility] = set()
        self._contradicted_incompatibilities_by_level: dict[int, set[Incompatibility]] = collections.defaultdict(set)
        self._solution = PartialSolution()

    @property
    def solution(self) -> PartialSolution:
        if False:
            print('Hello World!')
        return self._solution

    def solve(self) -> SolverResult:
        if False:
            i = 10
            return i + 15
        "\n        Finds a set of dependencies that match the root package's constraints,\n        or raises an error if no such set is available.\n        "
        start = time.time()
        root_dependency = Dependency(self._root.name, self._root.version)
        root_dependency.is_root = True
        self._add_incompatibility(Incompatibility([Term(root_dependency, False)], RootCause()))
        try:
            next: str | None = self._root.name
            while next is not None:
                self._propagate(next)
                next = self._choose_package_version()
            return self._result()
        except Exception:
            raise
        finally:
            self._log(f'Version solving took {time.time() - start:.3f} seconds.\nTried {self._solution.attempted_solutions} solutions.')

    def _propagate(self, package: str) -> None:
        if False:
            return 10
        '\n        Performs unit propagation on incompatibilities transitively\n        related to package to derive new assignments for _solution.\n        '
        changed = {package}
        while changed:
            package = changed.pop()
            for incompatibility in reversed(self._incompatibilities[package]):
                if incompatibility in self._contradicted_incompatibilities:
                    continue
                result = self._propagate_incompatibility(incompatibility)
                if result is _conflict:
                    root_cause = self._resolve_conflict(incompatibility)
                    changed.clear()
                    changed.add(str(self._propagate_incompatibility(root_cause)))
                    break
                elif result is not None:
                    changed.add(str(result))

    def _propagate_incompatibility(self, incompatibility: Incompatibility) -> str | object | None:
        if False:
            i = 10
            return i + 15
        "\n        If incompatibility is almost satisfied by _solution, adds the\n        negation of the unsatisfied term to _solution.\n\n        If incompatibility is satisfied by _solution, returns _conflict. If\n        incompatibility is almost satisfied by _solution, returns the\n        unsatisfied term's package name.\n\n        Otherwise, returns None.\n        "
        unsatisfied = None
        for term in incompatibility.terms:
            relation = self._solution.relation(term)
            if relation == SetRelation.DISJOINT:
                self._contradicted_incompatibilities.add(incompatibility)
                self._contradicted_incompatibilities_by_level[self._solution.decision_level].add(incompatibility)
                return None
            elif relation == SetRelation.OVERLAPPING:
                if unsatisfied is not None:
                    return None
                unsatisfied = term
        if unsatisfied is None:
            return _conflict
        self._contradicted_incompatibilities.add(incompatibility)
        self._contradicted_incompatibilities_by_level[self._solution.decision_level].add(incompatibility)
        adverb = 'not ' if unsatisfied.is_positive() else ''
        self._log(f'derived: {adverb}{unsatisfied.dependency}')
        self._solution.derive(unsatisfied.dependency, not unsatisfied.is_positive(), incompatibility)
        complete_name: str = unsatisfied.dependency.complete_name
        return complete_name

    def _resolve_conflict(self, incompatibility: Incompatibility) -> Incompatibility:
        if False:
            print('Hello World!')
        "\n        Given an incompatibility that's satisfied by _solution,\n        The `conflict resolution`_ constructs a new incompatibility that encapsulates\n        the root cause of the conflict and backtracks _solution until the new\n        incompatibility will allow _propagate() to deduce new assignments.\n\n        Adds the new incompatibility to _incompatibilities and returns it.\n\n        .. _conflict resolution:\n        https://github.com/dart-lang/pub/tree/master/doc/solver.md#conflict-resolution\n        "
        self._log(f'conflict: {incompatibility}')
        new_incompatibility = False
        while not incompatibility.is_failure():
            most_recent_term = None
            most_recent_satisfier = None
            difference = None
            previous_satisfier_level = 1
            for term in incompatibility.terms:
                satisfier = self._solution.satisfier(term)
                if most_recent_satisfier is None:
                    most_recent_term = term
                    most_recent_satisfier = satisfier
                elif most_recent_satisfier.index < satisfier.index:
                    previous_satisfier_level = max(previous_satisfier_level, most_recent_satisfier.decision_level)
                    most_recent_term = term
                    most_recent_satisfier = satisfier
                    difference = None
                else:
                    previous_satisfier_level = max(previous_satisfier_level, satisfier.decision_level)
                if most_recent_term == term:
                    difference = most_recent_satisfier.difference(most_recent_term)
                    if difference is not None:
                        previous_satisfier_level = max(previous_satisfier_level, self._solution.satisfier(difference.inverse).decision_level)
            assert most_recent_satisfier is not None
            if previous_satisfier_level < most_recent_satisfier.decision_level or most_recent_satisfier.cause is None:
                for level in range(self._solution.decision_level, previous_satisfier_level, -1):
                    if level in self._contradicted_incompatibilities_by_level:
                        self._contradicted_incompatibilities.difference_update(self._contradicted_incompatibilities_by_level.pop(level))
                    self._dependency_cache.clear_level(level)
                self._solution.backtrack(previous_satisfier_level)
                if new_incompatibility:
                    self._add_incompatibility(incompatibility)
                return incompatibility
            new_terms = [term for term in incompatibility.terms if term != most_recent_term]
            for term in most_recent_satisfier.cause.terms:
                if term.dependency != most_recent_satisfier.dependency:
                    new_terms.append(term)
            if difference is not None:
                inverse = difference.inverse
                if inverse.dependency != most_recent_satisfier.dependency:
                    new_terms.append(inverse)
            incompatibility = Incompatibility(new_terms, ConflictCause(incompatibility, most_recent_satisfier.cause))
            new_incompatibility = True
            partially = '' if difference is None else ' partially'
            self._log(f'! {most_recent_term} is{partially} satisfied by {most_recent_satisfier}')
            self._log(f'! which is caused by "{most_recent_satisfier.cause}"')
            self._log(f'! thus: {incompatibility}')
        raise SolveFailure(incompatibility)

    def _choose_package_version(self) -> str | None:
        if False:
            while True:
                i = 10
        '\n        Tries to select a version of a required package.\n\n        Returns the name of the package whose incompatibilities should be\n        propagated by _propagate(), or None indicating that version solving is\n        complete and a solution has been found.\n        '
        unsatisfied = self._solution.unsatisfied
        if not unsatisfied:
            return None

        class Preference:
            """
            Preference is one of the criteria for choosing which dependency to solve
            first. A higher value means that there are "more options" to satisfy
            a dependency. A lower value takes precedence.
            """
            DIRECT_ORIGIN = 0
            NO_CHOICE = 1
            USE_LATEST = 2
            LOCKED = 3
            DEFAULT = 4

        def _get_min(dependency: Dependency) -> tuple[bool, int, int]:
            if False:
                i = 10
                return i + 15
            if dependency.is_direct_origin():
                return (False, Preference.DIRECT_ORIGIN, -1)
            is_specific_marker = not dependency.marker.is_any()
            use_latest = dependency.name in self._provider.use_latest
            if not use_latest:
                locked = self._provider.get_locked(dependency)
                if locked:
                    return (is_specific_marker, Preference.LOCKED, -1)
            num_packages = len(self._dependency_cache.search_for(dependency, self._solution.decision_level))
            if num_packages < 2:
                preference = Preference.NO_CHOICE
            elif use_latest:
                preference = Preference.USE_LATEST
            else:
                preference = Preference.DEFAULT
            return (is_specific_marker, preference, -num_packages)
        dependency = min(unsatisfied, key=_get_min)
        locked = self._provider.get_locked(dependency)
        if locked is None:
            packages = self._dependency_cache.search_for(dependency, self._solution.decision_level)
            package = next(iter(packages), None)
            if package is None:
                self._add_incompatibility(Incompatibility([Term(dependency, True)], NoVersionsCause()))
                complete_name = dependency.complete_name
                return complete_name
        else:
            package = locked
        package = self._provider.complete_package(package)
        conflict = False
        for incompatibility in self._provider.incompatibilities_for(package):
            self._add_incompatibility(incompatibility)
            conflict = conflict or all((term.dependency.complete_name == dependency.complete_name or self._solution.satisfies(term) for term in incompatibility.terms))
        if not conflict:
            self._solution.decide(package.package)
            self._log(f'selecting {package.package.complete_name} ({package.package.full_pretty_version})')
        complete_name = dependency.complete_name
        return complete_name

    def _result(self) -> SolverResult:
        if False:
            while True:
                i = 10
        '\n        Creates a #SolverResult from the decisions in _solution\n        '
        decisions = self._solution.decisions
        return SolverResult(self._root, [p for p in decisions if not p.is_root()], self._solution.attempted_solutions)

    def _add_incompatibility(self, incompatibility: Incompatibility) -> None:
        if False:
            while True:
                i = 10
        self._log(f'fact: {incompatibility}')
        for term in incompatibility.terms:
            if term.dependency.complete_name not in self._incompatibilities:
                self._incompatibilities[term.dependency.complete_name] = []
            if incompatibility in self._incompatibilities[term.dependency.complete_name]:
                continue
            self._incompatibilities[term.dependency.complete_name].append(incompatibility)

    def _log(self, text: str) -> None:
        if False:
            print('Hello World!')
        self._provider.debug(text, self._solution.attempted_solutions)