"""Utilities to lazily create and visit candidates found.

Creating and visiting a candidate is a *very* costly operation. It involves
fetching, extracting, potentially building modules from source, and verifying
distribution metadata. It is therefore crucial for performance to keep
everything here lazy all the way down, so we only touch candidates that we
absolutely need, and not "download the world" when we only need one version of
something.
"""
import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, Set, Tuple
from pip._vendor.packaging.version import _BaseVersion
from .base import Candidate
IndexCandidateInfo = Tuple[_BaseVersion, Callable[[], Optional[Candidate]]]
if TYPE_CHECKING:
    SequenceCandidate = Sequence[Candidate]
else:
    SequenceCandidate = Sequence

def _iter_built(infos: Iterator[IndexCandidateInfo]) -> Iterator[Candidate]:
    if False:
        print('Hello World!')
    'Iterator for ``FoundCandidates``.\n\n    This iterator is used when the package is not already installed. Candidates\n    from index come later in their normal ordering.\n    '
    versions_found: Set[_BaseVersion] = set()
    for (version, func) in infos:
        if version in versions_found:
            continue
        candidate = func()
        if candidate is None:
            continue
        yield candidate
        versions_found.add(version)

def _iter_built_with_prepended(installed: Candidate, infos: Iterator[IndexCandidateInfo]) -> Iterator[Candidate]:
    if False:
        print('Hello World!')
    'Iterator for ``FoundCandidates``.\n\n    This iterator is used when the resolver prefers the already-installed\n    candidate and NOT to upgrade. The installed candidate is therefore\n    always yielded first, and candidates from index come later in their\n    normal ordering, except skipped when the version is already installed.\n    '
    yield installed
    versions_found: Set[_BaseVersion] = {installed.version}
    for (version, func) in infos:
        if version in versions_found:
            continue
        candidate = func()
        if candidate is None:
            continue
        yield candidate
        versions_found.add(version)

def _iter_built_with_inserted(installed: Candidate, infos: Iterator[IndexCandidateInfo]) -> Iterator[Candidate]:
    if False:
        i = 10
        return i + 15
    'Iterator for ``FoundCandidates``.\n\n    This iterator is used when the resolver prefers to upgrade an\n    already-installed package. Candidates from index are returned in their\n    normal ordering, except replaced when the version is already installed.\n\n    The implementation iterates through and yields other candidates, inserting\n    the installed candidate exactly once before we start yielding older or\n    equivalent candidates, or after all other candidates if they are all newer.\n    '
    versions_found: Set[_BaseVersion] = set()
    for (version, func) in infos:
        if version in versions_found:
            continue
        if installed.version >= version:
            yield installed
            versions_found.add(installed.version)
        candidate = func()
        if candidate is None:
            continue
        yield candidate
        versions_found.add(version)
    if installed.version not in versions_found:
        yield installed

class FoundCandidates(SequenceCandidate):
    """A lazy sequence to provide candidates to the resolver.

    The intended usage is to return this from `find_matches()` so the resolver
    can iterate through the sequence multiple times, but only access the index
    page when remote packages are actually needed. This improve performances
    when suitable candidates are already installed on disk.
    """

    def __init__(self, get_infos: Callable[[], Iterator[IndexCandidateInfo]], installed: Optional[Candidate], prefers_installed: bool, incompatible_ids: Set[int]):
        if False:
            for i in range(10):
                print('nop')
        self._get_infos = get_infos
        self._installed = installed
        self._prefers_installed = prefers_installed
        self._incompatible_ids = incompatible_ids

    def __getitem__(self, index: Any) -> Any:
        if False:
            while True:
                i = 10
        raise NotImplementedError("don't do this")

    def __iter__(self) -> Iterator[Candidate]:
        if False:
            for i in range(10):
                print('nop')
        infos = self._get_infos()
        if not self._installed:
            iterator = _iter_built(infos)
        elif self._prefers_installed:
            iterator = _iter_built_with_prepended(self._installed, infos)
        else:
            iterator = _iter_built_with_inserted(self._installed, infos)
        return (c for c in iterator if id(c) not in self._incompatible_ids)

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError("don't do this")

    @functools.lru_cache(maxsize=1)
    def __bool__(self) -> bool:
        if False:
            while True:
                i = 10
        if self._prefers_installed and self._installed:
            return True
        return any(self)