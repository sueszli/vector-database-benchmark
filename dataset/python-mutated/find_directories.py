"""
Tools that we use primiarly for finding .pyre_configuration files;
for the most part these helpers assist us in searching upward
from the current directory to find the nearest containing configuration.

This module also has become something of a dumping ground for filename
constants that are not directly related to directory searching.
"""
from __future__ import annotations
import enum
import itertools
import logging
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, NamedTuple, Optional
CONFIGURATION_FILE: str = '.pyre_configuration'
LOCAL_CONFIGURATION_FILE: str = '.pyre_configuration.local'
CODENAV_CONFIGURATION_FILE: str = '.pyre_configuration.codenav'
BINARY_NAME: str = 'pyre.bin'
CLIENT_NAME: str = 'pyre-client'
LOG_DIRECTORY: str = '.pyre'
LOG: logging.Logger = logging.getLogger(__name__)

def _find_parent_directory_containing(base: Path, target: str, predicate: Callable[[Path], bool], stop_search_after: Optional[int]) -> Optional[Path]:
    if False:
        return 10
    resolved_base = base.resolve(strict=True)
    for (i, candidate_directory) in enumerate(itertools.chain([resolved_base], resolved_base.parents)):
        candidate_path = candidate_directory / target
        try:
            if predicate(candidate_path):
                return candidate_directory
        except PermissionError:
            pass
        if stop_search_after is not None:
            if i >= stop_search_after:
                return None
    return None

def find_parent_directory_containing_file(base: Path, target: str, stop_search_after: Optional[int]=None) -> Optional[Path]:
    if False:
        print('Hello World!')
    '\n    Walk directories upwards from `base`, until the root directory is\n    reached. At each step, check if the `target` file exist, and return\n    the closest such directory if found. Return None if the search is\n    unsuccessful.\n\n    We stop searching after checking `stop_search_after` parent\n    directories of `base` if provided; this is mainly for testing.\n    '

    def is_file(path: Path) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return path.is_file()
    return _find_parent_directory_containing(base, target, predicate=is_file, stop_search_after=stop_search_after)

def find_outermost_directory_containing_file(base: Path, target: str, stop_search_after: Optional[int]) -> Optional[Path]:
    if False:
        i = 10
        return i + 15
    '\n    Walk directories upwards from `base`, until the root directory is\n    reached. At each step, check if the `target` file exist, and return\n    the farthest such directory if found. Return None if the search is\n    unsuccessful.\n\n    We stop searching after checking `stop_search_after` parent\n    directories of `base` if provided; this is mainly for testing.\n    '
    result: Optional[Path] = None
    resolved_base = base.resolve(strict=True)
    for (i, candidate_directory) in enumerate(itertools.chain([resolved_base], resolved_base.parents)):
        candidate_path = candidate_directory / target
        try:
            if candidate_path.is_file():
                result = candidate_directory
        except PermissionError:
            pass
        if stop_search_after is not None:
            if i >= stop_search_after:
                break
    return result

def find_global_root(base: Path) -> Optional[Path]:
    if False:
        return 10
    'Pyre always runs from the directory containing the nearest .pyre_configuration,\n    if one exists.'
    return find_parent_directory_containing_file(base, CONFIGURATION_FILE)

def get_relative_local_root(global_root: Path, local_root: Optional[Path]) -> Optional[str]:
    if False:
        return 10
    if local_root is None:
        return None
    else:
        try:
            return str(local_root.relative_to(global_root))
        except ValueError:
            return None

class FoundRoot(NamedTuple):
    global_root: Path
    local_root: Optional[Path] = None

def find_global_and_local_root(base: Path) -> Optional[FoundRoot]:
    if False:
        while True:
            i = 10
    '\n    Walk directories upwards from `base` and try to find both the global and local\n    pyre configurations.\n    Return `None` if no global configuration is found.\n    If a global configuration exists but no local configuration is found below it,\n    return the path to the global configuration.\n    If both global and local exist, return them as a pair.\n    '
    found_global_root = find_parent_directory_containing_file(base, CONFIGURATION_FILE)
    if found_global_root is None:
        return None
    found_local_root = find_parent_directory_containing_file(base, LOCAL_CONFIGURATION_FILE)
    if found_local_root is None:
        return FoundRoot(found_global_root)
    if found_local_root in found_global_root.parents:
        return FoundRoot(found_global_root)
    else:
        return FoundRoot(found_global_root, found_local_root)

def find_parent_directory_containing_directory(base: Path, target: str, stop_search_after: Optional[int]=None) -> Optional[Path]:
    if False:
        i = 10
        return i + 15
    '\n    Walk directories upwards from base, until the root directory is\n    reached. At each step, check if the target directory exist, and return\n    it if found. Return None if the search is unsuccessful.\n\n    We stop searching after checking `stop_search_after` parent\n    directories of `base` if provided; this is mainly for testing.\n    '

    def is_directory(path: Path) -> bool:
        if False:
            return 10
        return path.is_dir()
    return _find_parent_directory_containing(base, target, predicate=is_directory, stop_search_after=stop_search_after)

def find_typeshed() -> Optional[Path]:
    if False:
        print('Hello World!')
    install_root = Path(sys.prefix)
    bundled_typeshed = install_root / 'lib/pyre_check/typeshed/'
    if bundled_typeshed.is_dir():
        return bundled_typeshed
    LOG.debug('Could not find bundled typeshed. Try importing typeshed directly...')
    try:
        import typeshed
        return Path(typeshed.typeshed)
    except ImportError:
        LOG.error('`import typeshed` failed.')
    return None

class TypeshedLayout(enum.Enum):
    """
    We support three different ways of handling third-party stubs,
    which are traditionally found in subdirectories of `typeshed/stubs`:

    - They can be omitted entirely, so that typeshed only has stdlib stubs
    - They can live in subdirectories of `typeshed/stubs`
    - Or they can be flattened into a `typeshed/combined_stubs` directory

    These approaches are described by TypeshedLayout.

    The reason for this is that some of the operations for finding modules
    are O(number_of_search_roots). We know apriori that the `stubs` directories
    define mutually exclusive modules, so by combining them when we download
    typeshed we make Pyre more performant

    """
    STDLIB_ONLY = 'STDLIB_ONLY'
    STANDARD_THIRD_PARTY = 'STANDARD_THIRD_PARTY'
    COMBINED_THIRD_PARTY = 'COMBINED_THIRD_PARTY'

    @staticmethod
    def combined_stubs_root(typeshed_root: Path) -> Path:
        if False:
            print('Hello World!')
        return typeshed_root / 'combined_stubs'

    @staticmethod
    def standard_stubs_directory(typeshed_root: Path) -> Path:
        if False:
            while True:
                i = 10
        return typeshed_root / 'stubs'

    @staticmethod
    def infer_layout(typeshed_root: Path) -> TypeshedLayout:
        if False:
            while True:
                i = 10
        if TypeshedLayout.combined_stubs_root(typeshed_root).is_dir():
            return TypeshedLayout.COMBINED_THIRD_PARTY
        if TypeshedLayout.standard_stubs_directory(typeshed_root).is_dir():
            return TypeshedLayout.STANDARD_THIRD_PARTY
        else:
            return TypeshedLayout.STDLIB_ONLY

    @staticmethod
    def find_third_party_roots(typeshed_root: Path, layout: Optional[TypeshedLayout]=None) -> List[Path]:
        if False:
            print('Hello World!')
        '\n        Given the root of typeshed, find all subdirectories in it that can be used\n        as search paths for Pyre.\n\n\n        If `layout` is None, we will infer the layout (preferring\n        combined_stubs if available).\n        '
        layout = layout or TypeshedLayout.infer_layout(typeshed_root)
        if layout == TypeshedLayout.STDLIB_ONLY:
            return []
        elif layout == TypeshedLayout.COMBINED_THIRD_PARTY:
            return [TypeshedLayout.combined_stubs_root(typeshed_root)]
        elif layout == TypeshedLayout.STANDARD_THIRD_PARTY:
            return sorted(TypeshedLayout.standard_stubs_directory(typeshed_root).iterdir())
        else:
            raise RuntimeError(f'Unknown layout {layout}')

def find_typeshed_search_paths(typeshed_root: Path, layout: Optional[TypeshedLayout]=None) -> List[Path]:
    if False:
        print('Hello World!')
    search_path = []
    third_party_roots = TypeshedLayout.find_third_party_roots(typeshed_root=typeshed_root, layout=layout)
    for typeshed_subdirectory in itertools.chain([typeshed_root / 'stdlib'], third_party_roots):
        if typeshed_subdirectory.is_dir():
            search_path.append(typeshed_subdirectory)
    return search_path

def find_pyre_directory() -> Optional[Path]:
    if False:
        print('Hello World!')
    install_root = Path(sys.prefix)
    expected_pyre_path = install_root / 'lib/pyre_check/'
    if expected_pyre_path.is_dir():
        return expected_pyre_path
    return None

def find_taint_models_directory() -> Optional[Path]:
    if False:
        return 10
    pyre_check_path = find_pyre_directory()
    if pyre_check_path is None:
        return None
    bundled_taint_models = pyre_check_path / 'taint/'
    if bundled_taint_models.is_dir():
        return bundled_taint_models
    return None

def find_pysa_filters_directory() -> Optional[Path]:
    if False:
        while True:
            i = 10
    pyre_check_path = find_pyre_directory()
    if pyre_check_path is None:
        return None
    expected_pysa_filter_path = pyre_check_path / 'pysa_filters/'
    if expected_pysa_filter_path.is_dir():
        return expected_pysa_filter_path
    return None

def find_repository_root() -> Optional[Path]:
    if False:
        while True:
            i = 10
    repo_root = None
    try:
        hg_root = subprocess.check_output(['hg', 'root'], text=True)
        repo_root = Path(hg_root.strip())
    except subprocess.CalledProcessError as exception:
        LOG.debug(f'`hg root` failed with exception `{exception}`')
        return None
    return repo_root