"""
This module exposes a single function which checks synapse's dependencies are present
and correctly versioned. It makes use of `importlib.metadata` to do so. The details
are a bit murky: there's no easy way to get a map from "extras" to the packages they
require. But this is probably just symptomatic of Python's package management.
"""
import logging
from importlib import metadata
from typing import Iterable, NamedTuple, Optional
from packaging.requirements import Requirement
DISTRIBUTION_NAME = 'matrix-synapse'
__all__ = ['check_requirements']

class DependencyException(Exception):

    @property
    def message(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '\n'.join(['Missing Requirements: %s' % (', '.join(self.dependencies),), 'To install run:', '    pip install --upgrade --force %s' % (' '.join(self.dependencies),), ''])

    @property
    def dependencies(self) -> Iterable[str]:
        if False:
            for i in range(10):
                print('nop')
        for i in self.args[0]:
            yield ('"' + i + '"')
DEV_EXTRAS = {'lint', 'mypy', 'test', 'dev'}
ALL_EXTRAS = metadata.metadata(DISTRIBUTION_NAME).get_all('Provides-Extra')
assert ALL_EXTRAS is not None
RUNTIME_EXTRAS = set(ALL_EXTRAS) - DEV_EXTRAS
VERSION = metadata.version(DISTRIBUTION_NAME)

def _is_dev_dependency(req: Requirement) -> bool:
    if False:
        while True:
            i = 10
    return req.marker is not None and any((req.marker.evaluate({'extra': e}) for e in DEV_EXTRAS))

def _should_ignore_runtime_requirement(req: Requirement) -> bool:
    if False:
        for i in range(10):
            print('nop')
    if req.name == 'setuptools_rust':
        return True
    return False

class Dependency(NamedTuple):
    requirement: Requirement
    must_be_installed: bool

def _generic_dependencies() -> Iterable[Dependency]:
    if False:
        while True:
            i = 10
    'Yield pairs (requirement, must_be_installed).'
    requirements = metadata.requires(DISTRIBUTION_NAME)
    assert requirements is not None
    for raw_requirement in requirements:
        req = Requirement(raw_requirement)
        if _is_dev_dependency(req) or _should_ignore_runtime_requirement(req):
            continue
        must_be_installed = req.marker is None or req.marker.evaluate({'extra': ''})
        yield Dependency(req, must_be_installed)

def _dependencies_for_extra(extra: str) -> Iterable[Dependency]:
    if False:
        i = 10
        return i + 15
    'Yield additional dependencies needed for a given `extra`.'
    requirements = metadata.requires(DISTRIBUTION_NAME)
    assert requirements is not None
    for raw_requirement in requirements:
        req = Requirement(raw_requirement)
        if _is_dev_dependency(req):
            continue
        if req.marker is not None and req.marker.evaluate({'extra': extra}) and (not req.marker.evaluate({'extra': ''})):
            yield Dependency(req, True)

def _not_installed(requirement: Requirement, extra: Optional[str]=None) -> str:
    if False:
        print('Hello World!')
    if extra:
        return f'Synapse {VERSION} needs {requirement.name} for {extra}, but it is not installed'
    else:
        return f'Synapse {VERSION} needs {requirement.name}, but it is not installed'

def _incorrect_version(requirement: Requirement, got: str, extra: Optional[str]=None) -> str:
    if False:
        for i in range(10):
            print('nop')
    if extra:
        return f'Synapse {VERSION} needs {requirement} for {extra}, but got {requirement.name}=={got}'
    else:
        return f'Synapse {VERSION} needs {requirement}, but got {requirement.name}=={got}'

def _no_reported_version(requirement: Requirement, extra: Optional[str]=None) -> str:
    if False:
        for i in range(10):
            print('nop')
    if extra:
        return f"Synapse {VERSION} needs {requirement} for {extra}, but can't determine {requirement.name}'s version"
    else:
        return f"Synapse {VERSION} needs {requirement}, but can't determine {requirement.name}'s version"

def check_requirements(extra: Optional[str]=None) -> None:
    if False:
        i = 10
        return i + 15
    'Check Synapse\'s dependencies are present and correctly versioned.\n\n    If provided, `extra` must be the name of an pacakging extra (e.g. "saml2" in\n    `pip install matrix-synapse[saml2]`).\n\n    If `extra` is None, this function checks that\n    - all mandatory dependencies are installed and correctly versioned, and\n    - each optional dependency that\'s installed is correctly versioned.\n\n    If `extra` is not None, this function checks that\n    - the dependencies needed for that extra are installed and correctly versioned.\n\n    :raises DependencyException: if a dependency is missing or incorrectly versioned.\n    :raises ValueError: if this extra does not exist.\n    '
    if extra is None:
        dependencies = _generic_dependencies()
    elif extra in RUNTIME_EXTRAS:
        dependencies = _dependencies_for_extra(extra)
    else:
        raise ValueError(f"Synapse {VERSION} does not provide the feature '{extra}'")
    deps_unfulfilled = []
    errors = []
    for (requirement, must_be_installed) in dependencies:
        try:
            dist: metadata.Distribution = metadata.distribution(requirement.name)
        except metadata.PackageNotFoundError:
            if must_be_installed:
                deps_unfulfilled.append(requirement.name)
                errors.append(_not_installed(requirement, extra))
        else:
            if dist.version is None:
                deps_unfulfilled.append(requirement.name)
                errors.append(_no_reported_version(requirement, extra))
            elif not requirement.specifier.contains(dist.version, prereleases=True):
                deps_unfulfilled.append(requirement.name)
                errors.append(_incorrect_version(requirement, dist.version, extra))
    if deps_unfulfilled:
        for err in errors:
            logging.error(err)
        raise DependencyException(deps_unfulfilled)