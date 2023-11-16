from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet, Set, Union
from anyio import Path
from connector_ops.utils import Connector
from pipelines import main_logger
from pipelines.helpers.utils import IGNORED_FILE_EXTENSIONS, METADATA_FILE_NAME

def get_connector_modified_files(connector: Connector, all_modified_files: Set[Path]) -> FrozenSet[Path]:
    if False:
        while True:
            i = 10
    connector_modified_files = set()
    for modified_file in all_modified_files:
        modified_file_path = Path(modified_file)
        if modified_file_path.is_relative_to(connector.code_directory):
            connector_modified_files.add(modified_file)
    return frozenset(connector_modified_files)

def _find_modified_connectors(file_path: Union[str, Path], all_connectors: Set[Connector], dependency_scanning: bool=True) -> Set[Connector]:
    if False:
        print('Hello World!')
    'Find all connectors impacted by the file change.'
    modified_connectors = set()
    for connector in all_connectors:
        if Path(file_path).is_relative_to(Path(connector.code_directory)):
            main_logger.info(f"Adding connector '{connector}' due to connector file modification: {file_path}.")
            modified_connectors.add(connector)
        if dependency_scanning:
            for connector_dependency in connector.get_local_dependency_paths():
                if Path(file_path).is_relative_to(Path(connector_dependency)):
                    modified_connectors.add(connector)
                    main_logger.info(f"Adding connector '{connector}' due to dependency modification: '{file_path}'.")
    return modified_connectors

def _is_ignored_file(file_path: Union[str, Path]) -> bool:
    if False:
        i = 10
        return i + 15
    'Check if the provided file has an ignored extension.'
    return Path(file_path).suffix in IGNORED_FILE_EXTENSIONS

def get_modified_connectors(modified_files: Set[Path], all_connectors: Set[Connector], dependency_scanning: bool) -> Set[Connector]:
    if False:
        return 10
    "Create a mapping of modified connectors (key) and modified files (value).\n    If dependency scanning is enabled any modification to a dependency will trigger connector pipeline for all connectors that depend on it.\n    It currently works only for Java connectors .\n    It's especially useful to trigger tests of strict-encrypt variant when a change is made to the base connector.\n    Or to tests all jdbc connectors when a change is made to source-jdbc or base-java.\n    We'll consider extending the dependency resolution to Python connectors once we confirm that it's needed and feasible in term of scale.\n    "
    modified_connectors = set()
    for modified_file in modified_files:
        if not _is_ignored_file(modified_file):
            modified_connectors.update(_find_modified_connectors(modified_file, all_connectors, dependency_scanning))
    return modified_connectors

@dataclass(frozen=True)
class ConnectorWithModifiedFiles(Connector):
    modified_files: Set[Path] = field(default_factory=frozenset)

    @property
    def has_metadata_change(self) -> bool:
        if False:
            i = 10
            return i + 15
        return any((path.name == METADATA_FILE_NAME for path in self.modified_files))