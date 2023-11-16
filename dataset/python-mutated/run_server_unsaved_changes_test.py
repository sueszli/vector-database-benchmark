"""
Run an integration test of the pyre server's unsaved changes logic.

You can run this from the pyre-check repository root as follows:
```
# First, set your PYRE_CLIENT to point to the local build of an
# (open-source) pyre client and your PYRE_BINARY to a locally built
# pyre binary.

python3 scripts/run_server_unsaved_changes_test.py     --typeshed-path stubs/typeshed/typeshed     source/command/test/integration/fake_repository/
```
"""
import argparse
import contextlib
import io
import json
import logging
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
from argparse import Namespace
from dataclasses import dataclass
from logging import Logger
from typing import BinaryIO, Dict, Generator, Iterable, List, TextIO, Tuple
LOG: Logger = logging.getLogger(__name__)

class ConnectionFailure(Exception):
    pass

@contextlib.contextmanager
def connect_in_text_mode(socket_path: pathlib.Path) -> Generator[Tuple[TextIO, TextIO], None, None]:
    if False:
        return 10
    '\n    This is a line-oriented higher-level API than `connect`. It can be used\n    when the caller does not want to deal with the complexity of binary I/O.\n\n    The behavior is the same as `connect`, except the streams that are created\n    operates in text mode. Read/write APIs of the streams uses UTF-8 encoded\n    `str` instead of `bytes`. Those operations are also line-buffered, meaning\n    that the streams will automatically be flushed once the newline character\n    is encountered.\n    '
    with connect(socket_path) as (input_channel, output_channel):
        yield (io.TextIOWrapper(input_channel, line_buffering=True, errors='replace'), io.TextIOWrapper(output_channel, line_buffering=True, errors='replace'))

@contextlib.contextmanager
def connect(socket_path: pathlib.Path) -> Generator[Tuple[BinaryIO, BinaryIO], None, None]:
    if False:
        print('Hello World!')
    '\n    Connect to the socket at given path. Once connected, create an input and\n    an output stream from the socket. Both the input stream and the output\n    stream are in raw binary mode: read/write APIs of the streams need to use\n    `bytes` rather than `str`. The API is intended to be used like this:\n\n    ```\n    with connect(socket_path) as (input_stream, output_stream):\n        # Read from input_stream and write into output_stream here\n        ...\n    ```\n\n    Socket creation, connection, and closure will be automatically handled\n    inside this context manager. If any of the socket operations fail, raise\n    `ConnectionFailure`.\n    '
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client_socket:
            client_socket.connect(str(socket_path))
            with client_socket.makefile(mode='rb') as input_channel, client_socket.makefile(mode='wb') as output_channel:
                yield (input_channel, output_channel)
    except OSError as error:
        raise ConnectionFailure() from error

def _fetch_commit_paths(repository_path: str) -> Iterable[pathlib.Path]:
    if False:
        while True:
            i = 10
    return sorted(pathlib.Path(repository_path).iterdir())

class Repository:

    def __init__(self, typeshed_path: pathlib.Path, base_directory: str, repository_path: str, debug: bool) -> None:
        if False:
            print('Hello World!')
        self._commit_paths: Iterable[pathlib.Path] = _fetch_commit_paths(repository_path)
        self._pyre_directory: pathlib.Path = pathlib.Path(base_directory) / 'repository'
        self.debug: bool = debug
        self._initialize_pyre_directory(self._pyre_directory, typeshed_path, base_directory)
        self.socket_path: pathlib.Path = self.start_pyre_and_return_socket_path()
        pass

    @staticmethod
    def _initialize_pyre_directory(pyre_directory: pathlib.Path, typeshed_path: pathlib.Path, base_directory: str) -> None:
        if False:
            i = 10
            return i + 15
        pathlib.Path(pyre_directory).mkdir()
        pyre_configuration_path = pathlib.Path(pyre_directory) / '.pyre_configuration'
        pyre_configuration_path.write_text(json.dumps({'source_directories': ['.'], 'typeshed': str(typeshed_path.absolute()), 'optional_search_path': ['stubs']}))
        watchman_configuration_path = pathlib.Path(pyre_directory) / '.watchmanconfig'
        watchman_configuration_path.write_text(json.dumps({}))

    def get_repository_directory(self) -> pathlib.Path:
        if False:
            print('Hello World!')
        return self._pyre_directory

    def get_commit_paths(self) -> Iterable[pathlib.Path]:
        if False:
            return 10
        return self._commit_paths

    def start_pyre_and_return_socket_path(self) -> pathlib.Path:
        if False:
            return 10
        self.run_pyre('--logging-sections', '-n', 'start')
        result = self.run_pyre('info')
        socket_path = pathlib.Path(json.loads(result)['socket_path'])
        if not socket_path.is_socket():
            raise RuntimeError(f'Socket path: {socket_path} does not point to a socket. Path details: {socket_path.stat()}; Is the pyre server running?')
        return socket_path

    def send_update_request(self, request_message: str) -> str:
        if False:
            i = 10
            return i + 15
        socket_path = self.socket_path
        with connect_in_text_mode(socket_path) as (input_stream, output_stream):
            output_stream.write(f'{request_message}\n')
            result = input_stream.readline().strip()
            return result

    def overlay_update(self, file_path: pathlib.Path) -> str:
        if False:
            print('Hello World!')
        file_contents = pathlib.Path(file_path).read_text()
        python_file_name_without_path = file_path.name
        pyre_directory_file_path = pathlib.Path(self._pyre_directory) / python_file_name_without_path
        message = ['OverlayUpdate', {'overlay_id': str(pyre_directory_file_path), 'source_path': str(pyre_directory_file_path), 'code_update': ['NewCode', file_contents]}]
        return self.send_update_request(json.dumps(message))

    def incremental_update(self, file_path: pathlib.Path) -> str:
        if False:
            return 10
        python_file_name_without_path = file_path.name
        pyre_directory_file_path = pathlib.Path(self._pyre_directory) / python_file_name_without_path
        incremental_update_message = ['IncrementalUpdate', [str(pyre_directory_file_path)]]
        self.send_update_request(json.dumps(incremental_update_message))
        display_type_error_message = ['DisplayTypeError', [str(pyre_directory_file_path)]]
        return self.send_update_request(json.dumps(display_type_error_message))

    def modify_file(self, file_path: pathlib.Path) -> None:
        if False:
            return 10
        file_contents = pathlib.Path(file_path).read_text()
        python_file_name_without_path = file_path.name
        file_path = pathlib.Path(self._pyre_directory) / python_file_name_without_path
        file_path.write_text(file_contents)

    def _create_empty_file(self, file_path: pathlib.Path) -> None:
        if False:
            for i in range(10):
                print('nop')
        python_file_name_without_path = file_path.name
        (pathlib.Path(self._pyre_directory) / python_file_name_without_path).unlink(missing_ok=True)
        (pathlib.Path(self._pyre_directory) / python_file_name_without_path).touch()

    def initiate_empty_files(self, file_list: List[pathlib.Path]) -> None:
        if False:
            while True:
                i = 10
        for file_path in file_list:
            self._create_empty_file(file_path)
            self.incremental_update(file_path)

    def run_pyre(self, command: str, *arguments: str) -> str:
        if False:
            print('Hello World!')
        pyre_client = os.getenv('PYRE_CLIENT', 'pyre')
        standard_error = None if self.debug else subprocess.DEVNULL
        try:
            output = subprocess.run([pyre_client, '--output=json', command, *arguments], cwd=self._pyre_directory, stderr=standard_error, stdout=subprocess.PIPE)
        except subprocess.CalledProcessError as error:
            if error.returncode not in [0, 1]:
                raise error
            output = error.output
        return output.stdout

@dataclass(frozen=True)
class FileErrorsResult:
    file_name: pathlib.Path
    overlay_errors: str
    incremental_errors: str

def _get_file_errors_result(repository: Repository, file_path: pathlib.Path) -> FileErrorsResult:
    if False:
        i = 10
        return i + 15
    overlay_errors_response = repository.overlay_update(file_path)
    repository.modify_file(file_path)
    incremental_errors_response = repository.incremental_update(file_path)
    commit_result = FileErrorsResult(file_path, overlay_errors_response, incremental_errors_response)
    return commit_result

def run_unsaved_changes_test(typeshed_path: pathlib.Path, repository_path: str, debug: bool) -> int:
    if False:
        while True:
            i = 10
    with tempfile.TemporaryDirectory() as base_directory:
        repository = Repository(typeshed_path, base_directory, repository_path, debug)
        try:
            result = 0
            for commit in repository.get_commit_paths():
                discrepancies: Dict[pathlib.Path, Tuple[str, str]] = {}
                python_file_list = list(commit.glob('*.py'))
                repository.initiate_empty_files(python_file_list)
                for file_path in python_file_list:
                    commit_result = _get_file_errors_result(repository, file_path)
                    if commit_result.overlay_errors != commit_result.incremental_errors:
                        discrepancies[file_path] = (commit_result.overlay_errors, commit_result.incremental_errors)
                        LOG.error('Found discrepancies in %s for incremental check, file: %s', commit, file_path)
                result = _print_discrepancies(discrepancies, commit.name)
                if result:
                    break
            repository.run_pyre('stop')
            return result
        except Exception as uncaught_pyre_exception:
            LOG.error('Uncaught exception: `%s`', str(uncaught_pyre_exception))
            LOG.info('Pyre rage: %s', repository.run_pyre('rage'))
            raise uncaught_pyre_exception

def _print_discrepancies(discrepancies: Dict[pathlib.Path, Tuple[str, str]], commit: str) -> int:
    if False:
        return 10
    if len(discrepancies) == 0:
        return 0
    for (file_name, (actual_error, expected_error)) in discrepancies.items():
        LOG.error('Difference found for revision: {}, file_name: {}\n'.format(commit, file_name))
        LOG.error('Actual errors (pyre overlayUpdate): {}\n'.format(actual_error))
        LOG.error('Expected errors (pyre incremental): {}\n'.format(expected_error))
    return 1

def run(repository_location: str, typeshed_path: pathlib.Path, debug: bool) -> int:
    if False:
        for i in range(10):
            print('nop')
    return run_unsaved_changes_test(typeshed_path, repository_location, debug)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('repository_location', help='Path to directory with fake commit list')
    parser.add_argument('--typeshed-path', help='Path to typeshed.')
    parser.add_argument('--debug', action='store_true', default=False)
    arguments: Namespace = parser.parse_args()
    sys.exit(run(arguments.repository_location, pathlib.Path(arguments.typeshed_path), arguments.debug))