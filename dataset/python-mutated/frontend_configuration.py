"""
This module contains Python classes representing the information found
in a Pyre configuration file (.pyre_configuration).

The implementation is split into Base and OpenSource so that it is
possible to customize Pyre by implementing a new command-line tool
with additional configuration, using open-source Pyre as a library.
"""
import abc
import dataclasses
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional
from . import configuration as configuration_module, find_directories
LOG: logging.Logger = logging.getLogger(__name__)

@dataclasses.dataclass(frozen=True)
class SavedStateProject:
    name: str
    metadata: Optional[str] = None

class Base(abc.ABC):

    @abc.abstractmethod
    def get_dot_pyre_directory(self) -> Path:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @abc.abstractmethod
    def get_binary_location(self, download_if_needed: bool=False) -> Optional[Path]:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @abc.abstractmethod
    def get_typeshed_location(self, download_if_needed: bool=False) -> Optional[Path]:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @abc.abstractmethod
    def get_binary_version(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @abc.abstractmethod
    def get_content_for_display(self) -> str:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @abc.abstractmethod
    def get_global_root(self) -> Path:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @abc.abstractmethod
    def get_relative_local_root(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @abc.abstractmethod
    def get_excludes(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @abc.abstractmethod
    def is_strict(self) -> bool:
        if False:
            return 10
        raise NotImplementedError()

    @abc.abstractmethod
    def get_remote_logger(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    @abc.abstractmethod
    def get_number_of_workers(self) -> int:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @abc.abstractmethod
    def get_python_version(self) -> configuration_module.PythonVersion:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @abc.abstractmethod
    def get_shared_memory(self) -> configuration_module.SharedMemory:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @abc.abstractmethod
    def get_valid_extension_suffixes(self) -> List[str]:
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    @abc.abstractmethod
    def get_ignore_all_errors(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @abc.abstractmethod
    def get_only_check_paths(self) -> List[str]:
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    @abc.abstractmethod
    def get_existent_user_specified_search_paths(self) -> List[configuration_module.search_path.Element]:
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    @abc.abstractmethod
    def get_existent_source_directories(self) -> List[configuration_module.search_path.Element]:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @abc.abstractmethod
    def get_existent_unwatched_dependency(self) -> Optional[configuration_module.unwatched.UnwatchedDependency]:
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    @abc.abstractmethod
    def is_source_directories_defined(self) -> bool:
        if False:
            return 10
        raise NotImplementedError()

    @abc.abstractmethod
    def get_buck_targets(self) -> Optional[List[str]]:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @abc.abstractmethod
    def uses_buck2(self) -> bool:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @abc.abstractmethod
    def get_buck_mode(self) -> Optional[str]:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @abc.abstractmethod
    def get_buck_isolation_prefix(self) -> Optional[str]:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @abc.abstractmethod
    def get_buck_bxl_builder(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    @abc.abstractmethod
    def get_other_critical_files(self) -> List[str]:
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    @abc.abstractmethod
    def get_taint_models_path(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @abc.abstractmethod
    def get_project_identifier(self) -> str:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @abc.abstractmethod
    def get_enable_readonly_analysis(self) -> Optional[bool]:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @abc.abstractmethod
    def get_enable_unawaited_awaitable_analysis(self) -> Optional[bool]:
        if False:
            return 10
        raise NotImplementedError()

    @abc.abstractmethod
    def get_saved_state_project(self) -> Optional[SavedStateProject]:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @abc.abstractmethod
    def get_include_suppressed_errors(self) -> Optional[bool]:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @abc.abstractmethod
    def get_use_errpy_parser(self) -> bool:
        if False:
            return 10
        raise NotImplementedError()

    def get_local_root(self) -> Optional[Path]:
        if False:
            while True:
                i = 10
        relative_local_root = self.get_relative_local_root()
        if relative_local_root is None:
            return None
        return self.get_global_root() / relative_local_root

    def get_log_directory(self) -> Path:
        if False:
            while True:
                i = 10
        dot_pyre_directory = self.get_dot_pyre_directory()
        relative_local_root = self.get_relative_local_root()
        return dot_pyre_directory if relative_local_root is None else dot_pyre_directory / relative_local_root

    def get_existent_typeshed_search_paths(self) -> List[configuration_module.search_path.Element]:
        if False:
            while True:
                i = 10
        typeshed_root = self.get_typeshed_location(download_if_needed=True)
        if typeshed_root is None:
            return []
        return [configuration_module.search_path.SimpleElement(str(element)) for element in find_directories.find_typeshed_search_paths(typeshed_root)]

    def get_existent_search_paths(self) -> List[configuration_module.search_path.Element]:
        if False:
            while True:
                i = 10
        return [*self.get_existent_user_specified_search_paths(), *self.get_existent_typeshed_search_paths()]

class OpenSource(Base):

    def __init__(self, configuration: configuration_module.Configuration) -> None:
        if False:
            while True:
                i = 10
        self.configuration = configuration

    def get_dot_pyre_directory(self) -> Path:
        if False:
            i = 10
            return i + 15
        return self.configuration.dot_pyre_directory or self.get_global_root() / find_directories.LOG_DIRECTORY

    def get_binary_location(self, download_if_needed: bool=False) -> Optional[Path]:
        if False:
            while True:
                i = 10
        binary = self.configuration.binary
        if binary is not None:
            return Path(binary)
        LOG.info(f'No binary specified, looking for `{find_directories.BINARY_NAME}` in PATH')
        binary_candidate = shutil.which(find_directories.BINARY_NAME)
        if binary_candidate is None:
            binary_candidate_name = os.path.join(os.path.dirname(sys.argv[0]), find_directories.BINARY_NAME)
            binary_candidate = shutil.which(binary_candidate_name)
        return Path(binary_candidate) if binary_candidate is not None else None

    def get_typeshed_location(self, download_if_needed: bool=False) -> Optional[Path]:
        if False:
            i = 10
            return i + 15
        typeshed = self.configuration.typeshed
        if typeshed is not None:
            return Path(typeshed)
        LOG.info('No typeshed specified, looking for it...')
        auto_determined_typeshed = find_directories.find_typeshed()
        if auto_determined_typeshed is None:
            LOG.warning('Could not find a suitable typeshed. Types for Python builtins and standard libraries may be missing!')
            return None
        else:
            LOG.info(f'Found: `{auto_determined_typeshed}`')
            return auto_determined_typeshed

    def get_binary_version(self) -> Optional[str]:
        if False:
            print('Hello World!')
        return None

    def get_content_for_display(self) -> str:
        if False:
            i = 10
            return i + 15
        return json.dumps(self.configuration.to_json(), indent=2)

    def get_global_root(self) -> Path:
        if False:
            while True:
                i = 10
        return self.configuration.global_root

    def get_relative_local_root(self) -> Optional[str]:
        if False:
            return 10
        return self.configuration.relative_local_root

    def get_excludes(self) -> List[str]:
        if False:
            while True:
                i = 10
        return list(self.configuration.excludes)

    def is_strict(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.configuration.strict

    def get_remote_logger(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        return self.configuration.logger

    def get_number_of_workers(self) -> int:
        if False:
            print('Hello World!')
        return self.configuration.get_number_of_workers()

    def get_python_version(self) -> configuration_module.PythonVersion:
        if False:
            i = 10
            return i + 15
        return self.configuration.get_python_version()

    def get_shared_memory(self) -> configuration_module.SharedMemory:
        if False:
            for i in range(10):
                print('nop')
        return self.configuration.shared_memory

    def get_valid_extension_suffixes(self) -> List[str]:
        if False:
            return 10
        return self.configuration.get_valid_extension_suffixes()

    def get_ignore_all_errors(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        return list(self.configuration.ignore_all_errors)

    def get_only_check_paths(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        return list(self.configuration.only_check_paths)

    def get_existent_user_specified_search_paths(self) -> List[configuration_module.search_path.Element]:
        if False:
            return 10
        return self.configuration.expand_and_get_existent_search_paths()

    def get_existent_source_directories(self) -> List[configuration_module.search_path.Element]:
        if False:
            i = 10
            return i + 15
        return self.configuration.expand_and_get_existent_source_directories()

    def get_existent_unwatched_dependency(self) -> Optional[configuration_module.unwatched.UnwatchedDependency]:
        if False:
            i = 10
            return i + 15
        return self.configuration.get_existent_unwatched_dependency()

    def is_source_directories_defined(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.configuration.source_directories is not None

    def get_buck_targets(self) -> Optional[List[str]]:
        if False:
            return 10
        targets = self.configuration.targets
        return list(targets) if targets is not None else None

    def uses_buck2(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.configuration.use_buck2

    def get_buck_mode(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        mode = self.configuration.buck_mode
        return mode.get() if mode is not None else None

    def get_buck_isolation_prefix(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        return self.configuration.isolation_prefix

    def get_buck_bxl_builder(self) -> Optional[str]:
        if False:
            return 10
        return self.configuration.bxl_builder

    def get_other_critical_files(self) -> List[str]:
        if False:
            while True:
                i = 10
        return list(self.configuration.other_critical_files)

    def get_taint_models_path(self) -> List[str]:
        if False:
            print('Hello World!')
        return list(self.configuration.taint_models_path)

    def get_enable_readonly_analysis(self) -> Optional[bool]:
        if False:
            return 10
        return self.configuration.enable_readonly_analysis

    def get_enable_unawaited_awaitable_analysis(self) -> Optional[bool]:
        if False:
            while True:
                i = 10
        return self.configuration.enable_unawaited_awaitable_analysis

    def get_project_identifier(self) -> str:
        if False:
            print('Hello World!')
        return self.configuration.project_identifier

    def get_saved_state_project(self) -> Optional[SavedStateProject]:
        if False:
            while True:
                i = 10
        return None

    def get_include_suppressed_errors(self) -> Optional[bool]:
        if False:
            i = 10
            return i + 15
        return self.configuration.include_suppressed_errors

    def get_use_errpy_parser(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.configuration.use_errpy_parser