"""Utility code for facilitating collection of code coverage when running tests."""
from __future__ import annotations
import dataclasses
import os
import sqlite3
import tempfile
import typing as t
from .config import IntegrationConfig, SanityConfig, TestConfig
from .io import write_text_file, make_dirs, open_binary_file
from .util import ApplicationError, InternalError, COVERAGE_CONFIG_NAME, remove_tree, sanitize_host_name, str_to_version
from .data import data_context
from .util_common import ExitHandler, intercept_python, ResultType
from .host_configs import DockerConfig, HostConfig, OriginConfig, PosixRemoteConfig, PosixSshConfig, PythonConfig
from .constants import SUPPORTED_PYTHON_VERSIONS, CONTROLLER_PYTHON_VERSIONS
from .thread import mutex

@dataclasses.dataclass(frozen=True)
class CoverageVersion:
    """Details about a coverage version and its supported Python versions."""
    coverage_version: str
    schema_version: int
    min_python: tuple[int, int]
    max_python: tuple[int, int]
COVERAGE_VERSIONS = (CoverageVersion('7.3.2', 7, (3, 8), (3, 12)), CoverageVersion('6.5.0', 7, (3, 7), (3, 7)))
'\nThis tuple specifies the coverage version to use for Python version ranges.\n'
CONTROLLER_COVERAGE_VERSION = COVERAGE_VERSIONS[0]
'The coverage version supported on the controller.'

class CoverageError(ApplicationError):
    """Exception caused while attempting to read a coverage file."""

    def __init__(self, path: str, message: str) -> None:
        if False:
            while True:
                i = 10
        self.path = path
        self.message = message
        super().__init__(f'Error reading coverage file "{os.path.relpath(path)}": {message}')

def get_coverage_version(version: str) -> CoverageVersion:
    if False:
        for i in range(10):
            print('nop')
    'Return the coverage version to use with the specified Python version.'
    python_version = str_to_version(version)
    supported_versions = [entry for entry in COVERAGE_VERSIONS if entry.min_python <= python_version <= entry.max_python]
    if not supported_versions:
        raise InternalError(f'Python {version} has no matching entry in COVERAGE_VERSIONS.')
    if len(supported_versions) > 1:
        raise InternalError(f'Python {version} has multiple matching entries in COVERAGE_VERSIONS.')
    coverage_version = supported_versions[0]
    return coverage_version

def get_coverage_file_schema_version(path: str) -> int:
    if False:
        print('Hello World!')
    '\n    Return the schema version from the specified coverage file.\n    SQLite based files report schema version 1 or later.\n    JSON based files are reported as schema version 0.\n    An exception is raised if the file is not recognized or the schema version cannot be determined.\n    '
    with open_binary_file(path) as file_obj:
        header = file_obj.read(16)
    if header.startswith(b'!coverage.py: '):
        return 0
    if header.startswith(b'SQLite'):
        return get_sqlite_schema_version(path)
    raise CoverageError(path, f'Unknown header: {header!r}')

def get_sqlite_schema_version(path: str) -> int:
    if False:
        print('Hello World!')
    'Return the schema version from a SQLite based coverage file.'
    try:
        with sqlite3.connect(path) as connection:
            cursor = connection.cursor()
            cursor.execute('select version from coverage_schema')
            schema_version = cursor.fetchmany(1)[0][0]
    except Exception as ex:
        raise CoverageError(path, f'SQLite error: {ex}') from ex
    if not isinstance(schema_version, int):
        raise CoverageError(path, f'Schema version is {type(schema_version)} instead of {int}: {schema_version}')
    if schema_version < 1:
        raise CoverageError(path, f'Schema version is out-of-range: {schema_version}')
    return schema_version

def cover_python(args: TestConfig, python: PythonConfig, cmd: list[str], target_name: str, env: dict[str, str], capture: bool, data: t.Optional[str]=None, cwd: t.Optional[str]=None) -> tuple[t.Optional[str], t.Optional[str]]:
    if False:
        for i in range(10):
            print('nop')
    'Run a command while collecting Python code coverage.'
    if args.coverage:
        env.update(get_coverage_environment(args, target_name, python.version))
    return intercept_python(args, python, cmd, env, capture, data, cwd)

def get_coverage_platform(config: HostConfig) -> str:
    if False:
        while True:
            i = 10
    'Return the platform label for the given host config.'
    if isinstance(config, PosixRemoteConfig):
        platform = f'remote-{sanitize_host_name(config.name)}'
    elif isinstance(config, DockerConfig):
        platform = f'docker-{sanitize_host_name(config.name)}'
    elif isinstance(config, PosixSshConfig):
        platform = f'ssh-{sanitize_host_name(config.host)}'
    elif isinstance(config, OriginConfig):
        platform = 'origin'
    else:
        raise NotImplementedError(f'Coverage platform label not defined for type: {type(config)}')
    return platform

def get_coverage_environment(args: TestConfig, target_name: str, version: str) -> dict[str, str]:
    if False:
        print('Hello World!')
    'Return environment variables needed to collect code coverage.'
    config_file = get_coverage_config(args)
    coverage_name = '='.join((args.command, target_name, get_coverage_platform(args.controller), f'python-{version}', 'coverage'))
    coverage_dir = os.path.join(data_context().content.root, data_context().content.results_path, ResultType.COVERAGE.name)
    coverage_file = os.path.join(coverage_dir, coverage_name)
    make_dirs(coverage_dir)
    if args.coverage_check:
        coverage_file = ''
    env = dict(COVERAGE_CONF=config_file, COVERAGE_FILE=coverage_file)
    return env

@mutex
def get_coverage_config(args: TestConfig) -> str:
    if False:
        i = 10
        return i + 15
    'Return the path to the coverage config, creating the config if it does not already exist.'
    try:
        return get_coverage_config.path
    except AttributeError:
        pass
    coverage_config = generate_coverage_config(args)
    if args.explain:
        temp_dir = '/tmp/coverage-temp-dir'
    else:
        temp_dir = tempfile.mkdtemp()
        ExitHandler.register(lambda : remove_tree(temp_dir))
    path = os.path.join(temp_dir, COVERAGE_CONFIG_NAME)
    if not args.explain:
        write_text_file(path, coverage_config)
    get_coverage_config.path = path
    return path

def generate_coverage_config(args: TestConfig) -> str:
    if False:
        while True:
            i = 10
    'Generate code coverage configuration for tests.'
    if data_context().content.collection:
        coverage_config = generate_collection_coverage_config(args)
    else:
        coverage_config = generate_ansible_coverage_config()
    return coverage_config

def generate_ansible_coverage_config() -> str:
    if False:
        print('Hello World!')
    'Generate code coverage configuration for Ansible tests.'
    coverage_config = '\n[run]\nbranch = True\nconcurrency =\n    multiprocessing\n    thread\nparallel = True\n\nomit =\n    */python*/dist-packages/*\n    */python*/site-packages/*\n    */python*/distutils/*\n    */pyshared/*\n    */pytest\n    */AnsiballZ_*.py\n    */test/results/*\n'
    return coverage_config

def generate_collection_coverage_config(args: TestConfig) -> str:
    if False:
        print('Hello World!')
    'Generate code coverage configuration for Ansible Collection tests.'
    coverage_config = '\n[run]\nbranch = True\nconcurrency =\n    multiprocessing\n    thread\nparallel = True\ndisable_warnings =\n    no-data-collected\n'
    if isinstance(args, IntegrationConfig):
        coverage_config += '\ninclude =\n    %s/*\n    */%s/*\n' % (data_context().content.root, data_context().content.collection.directory)
    elif isinstance(args, SanityConfig):
        coverage_config += '\ninclude =\n    %s/*\n\nomit =\n    %s/*\n' % (data_context().content.root, os.path.join(data_context().content.root, data_context().content.results_path))
    else:
        coverage_config += '\ninclude =\n     %s/*\n' % data_context().content.root
    return coverage_config

def self_check() -> None:
    if False:
        print('Hello World!')
    'Check for internal errors due to incorrect code changes.'
    for version in SUPPORTED_PYTHON_VERSIONS:
        get_coverage_version(version)
    for version in CONTROLLER_PYTHON_VERSIONS:
        if get_coverage_version(version) != CONTROLLER_COVERAGE_VERSION:
            raise InternalError(f'Controller Python version {version} is not mapped to the latest coverage version.')
self_check()