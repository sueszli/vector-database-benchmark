from __future__ import annotations
import atexit
import multiprocessing
import os
import signal
import sys
import time
from abc import abstractmethod
from dataclasses import dataclass
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Generator
import yaml
from airflow_breeze.global_constants import AIRFLOW_PYTHON_COMPATIBILITY_MATRIX, DEFAULT_PYTHON_MAJOR_MINOR_VERSION
from airflow_breeze.utils.console import Output, get_console
from airflow_breeze.utils.github import download_constraints_file, download_file_from_github
from airflow_breeze.utils.path_utils import AIRFLOW_SOURCES_ROOT, FILES_SBOM_DIR
from airflow_breeze.utils.run_utils import run_command
from airflow_breeze.utils.shared_options import get_dry_run

def start_cdxgen_server(application_root_path: Path, run_in_parallel: bool, parallelism: int) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Start cdxgen server that is used to perform cdxgen scans of applications in child process\n    :param run_in_parallel: run parallel servers\n    :param parallelism: parallelism to use\n    :param application_root_path: path where the application to scan is located\n    '
    run_command(['docker', 'pull', 'ghcr.io/cyclonedx/cdxgen'], check=True)
    if not run_in_parallel:
        fork_cdxgen_server(application_root_path)
    else:
        for i in range(parallelism):
            fork_cdxgen_server(application_root_path, port=9091 + i)
    time.sleep(1)
    get_console().print('[info]Waiting for cdxgen server to start')
    time.sleep(3)

def fork_cdxgen_server(application_root_path, port=9090):
    if False:
        print('Hello World!')
    pid = os.fork()
    if pid:
        atexit.register(os.killpg, pid, signal.SIGTERM)
    else:
        if os.getpid() != os.getsid(0):
            os.setpgid(0, 0)
        run_command(['docker', 'run', '--init', '--rm', '-p', f'{port}:{port}', '-v', '/tmp:/tmp', '-v', f'{application_root_path}:/app', '-t', 'ghcr.io/cyclonedx/cdxgen', '--server', '--server-host', '0.0.0.0', '--server-port', str(port)], check=True)
        sys.exit(0)

def get_port_mapping(x):
    if False:
        return 10
    time.sleep(1)
    return (multiprocessing.current_process().name, 9091 + x)

def get_cdxgen_port_mapping(parallelism: int, pool: Pool) -> dict[str, int]:
    if False:
        i = 10
        return i + 15
    '\n    Map processes from pool to port numbers so that there is always the same port\n    used by the same process in the pool - effectively having one multiprocessing\n    process talking to the same cdxgen server\n\n    :param parallelism: parallelism to use\n    :param pool: pool to map ports for\n    :return: mapping of process name to port\n    '
    port_map: dict[str, int] = dict(pool.map(get_port_mapping, range(parallelism)))
    return port_map

def get_all_airflow_versions_image_name(python_version: str) -> str:
    if False:
        i = 10
        return i + 15
    return f'ghcr.io/apache/airflow/airflow-dev/all-airflow/python{python_version}'

def list_providers_from_providers_requirements(airflow_site_archive_directory: Path) -> Generator[tuple[str, str, str, Path], None, None]:
    if False:
        while True:
            i = 10
    for node_name in os.listdir(PROVIDER_REQUIREMENTS_DIR_PATH):
        if not node_name.startswith('provider'):
            continue
        (provider_id, provider_version) = node_name.rsplit('-', 1)
        provider_documentation_directory = airflow_site_archive_directory / f"apache-airflow-providers-{provider_id.replace('provider-', '').replace('.', '-')}"
        provider_version_documentation_directory = provider_documentation_directory / provider_version
        if not provider_version_documentation_directory.exists():
            get_console().print(f'[warning]The {provider_version_documentation_directory} does not exist. Skipping')
            continue
        yield (node_name, provider_id, provider_version, provider_version_documentation_directory)
TARGET_DIR_NAME = 'provider_requirements'
PROVIDER_REQUIREMENTS_DIR_PATH = FILES_SBOM_DIR / TARGET_DIR_NAME
DOCKER_FILE_PREFIX = f'/files/sbom/{TARGET_DIR_NAME}/'

def get_requirements_for_provider(provider_id: str, airflow_version: str, output: Output | None, provider_version: str | None=None, python_version: str=DEFAULT_PYTHON_MAJOR_MINOR_VERSION, force: bool=False) -> tuple[int, str]:
    if False:
        return 10
    provider_path_array = provider_id.split('.')
    if not provider_version:
        provider_file = (AIRFLOW_SOURCES_ROOT / 'airflow' / 'providers').joinpath(*provider_path_array) / 'provider.yaml'
        provider_version = yaml.safe_load(provider_file.read_text())['versions'][0]
    airflow_core_file_name = f'airflow-{airflow_version}-python{python_version}-requirements.txt'
    airflow_core_path = PROVIDER_REQUIREMENTS_DIR_PATH / airflow_core_file_name
    provider_folder_name = f'provider-{provider_id}-{provider_version}'
    provider_folder_path = PROVIDER_REQUIREMENTS_DIR_PATH / provider_folder_name
    provider_with_core_folder_path = provider_folder_path / f'python{python_version}' / 'with-core'
    provider_with_core_folder_path.mkdir(exist_ok=True, parents=True)
    provider_with_core_path = provider_with_core_folder_path / 'requirements.txt'
    provider_without_core_folder_path = provider_folder_path / f'python{python_version}' / 'without-core'
    provider_without_core_folder_path.mkdir(exist_ok=True, parents=True)
    provider_without_core_file = provider_without_core_folder_path / 'requirements.txt'
    docker_file_provider_with_core_folder_prefix = f'{DOCKER_FILE_PREFIX}{provider_folder_name}/python{python_version}/with-core/'
    if os.path.exists(provider_with_core_path) and os.path.exists(provider_without_core_file) and (force is False):
        get_console(output=output).print(f'[warning] Requirements for provider {provider_id} version {provider_version} python {python_version} already exist, skipping. Set force=True to force generation.')
        return (0, f'Provider requirements already existed, skipped generation for {provider_id} version {provider_version} python {python_version}')
    else:
        provider_folder_path.mkdir(exist_ok=True)
    command = f'\nmkdir -pv {DOCKER_FILE_PREFIX}\n/opt/airflow/airflow-{airflow_version}/bin/pip freeze | sort > {DOCKER_FILE_PREFIX}{airflow_core_file_name}\n/opt/airflow/airflow-{airflow_version}/bin/pip install apache-airflow=={airflow_version}     apache-airflow-providers-{provider_id}=={provider_version}\n/opt/airflow/airflow-{airflow_version}/bin/pip freeze | sort >     {docker_file_provider_with_core_folder_prefix}requirements.txt\nchown --recursive {os.getuid()}:{os.getgid()} {DOCKER_FILE_PREFIX}{provider_folder_name}\n'
    provider_command_result = run_command(['docker', 'run', '--rm', '-e', f'HOST_USER_ID={os.getuid()}', '-e', f'HOST_GROUP_ID={os.getgid()}', '-v', f'{AIRFLOW_SOURCES_ROOT}/files:/files', get_all_airflow_versions_image_name(python_version=python_version), '-c', ';'.join(command.splitlines()[1:-1])], output=output)
    get_console(output=output).print(f'[info]Airflow requirements in {airflow_core_path}')
    get_console(output=output).print(f'[info]Provider requirements in {provider_with_core_path}')
    base_packages = {package.split('==')[0] for package in airflow_core_path.read_text().splitlines()}
    base_packages.add('apache-airflow-providers-' + provider_id.replace('.', '-'))
    provider_packages = sorted([line for line in provider_with_core_path.read_text().splitlines() if line.split('==')[0] not in base_packages])
    get_console(output=output).print(f'[info]Provider {provider_id} has {len(provider_packages)} transitively dependent packages (excluding airflow and its dependencies)')
    get_console(output=output).print(provider_packages)
    provider_without_core_file.write_text(''.join((f'{p}\n' for p in provider_packages)))
    get_console(output=output).print(f'[success]Generated {provider_id}:{provider_version}:python{python_version} requirements in {provider_without_core_file}')
    return (provider_command_result.returncode, f'Provider requirements generated for {provider_id}:{provider_version}:python{python_version}')

def build_all_airflow_versions_base_image(python_version: str, output: Output | None) -> tuple[int, str]:
    if False:
        print('Hello World!')
    '\n    Build an image with all airflow versions pre-installed in separate virtualenvs.\n\n    Image cache was built using stable main/ci tags to not rebuild cache on every\n    main new commit. Tags used are:\n\n    main_ci_images_fixed_tags = {\n        "3.6": "latest",\n        "3.7": "latest",\n        "3.8": "e698dbfe25da10d09c5810938f586535633928a4",\n        "3.9": "e698dbfe25da10d09c5810938f586535633928a4",\n        "3.10": "e698dbfe25da10d09c5810938f586535633928a4",\n        "3.11": "e698dbfe25da10d09c5810938f586535633928a4",\n    }\n    '
    image_name = get_all_airflow_versions_image_name(python_version=python_version)
    dockerfile = f'\nFROM {image_name}\nRUN pip install --upgrade pip --no-cache-dir\n# Prevent setting sources in PYTHONPATH to not interfere with virtualenvs\nENV USE_AIRFLOW_VERSION=none\nENV START_AIRFLOW=none\n    '
    compatible_airflow_versions = [airflow_version for (airflow_version, python_versions) in AIRFLOW_PYTHON_COMPATIBILITY_MATRIX.items() if python_version in python_versions]
    for airflow_version in compatible_airflow_versions:
        dockerfile += f'\n# Create the virtualenv and install the proper airflow version in it\nRUN python -m venv /opt/airflow/airflow-{airflow_version} && /opt/airflow/airflow-{airflow_version}/bin/pip install --no-cache-dir --upgrade pip && /opt/airflow/airflow-{airflow_version}/bin/pip install apache-airflow=={airflow_version}     --constraint https://raw.githubusercontent.com/apache/airflow/constraints-{airflow_version}/constraints-{python_version}.txt\n'
    build_command = run_command(['docker', 'buildx', 'build', '--cache-from', image_name, '--tag', image_name, '-'], input=dockerfile, text=True, check=True, output=output)
    return (build_command.returncode, f'All airflow image built for python {python_version}')

@dataclass
class SbomApplicationJob:
    python_version: str
    target_path: Path

    @abstractmethod
    def produce(self, output: Output | None, port: int) -> tuple[int, str]:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def get_job_name(self) -> str:
        if False:
            while True:
                i = 10
        pass

@dataclass
class SbomCoreJob(SbomApplicationJob):
    airflow_version: str
    application_root_path: Path
    include_provider_dependencies: bool

    def get_job_name(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'{self.airflow_version}:python{self.python_version}'

    def download_dependency_files(self, output: Output | None) -> bool:
        if False:
            for i in range(10):
                print('nop')
        source_dir = self.application_root_path / self.airflow_version / f'python{self.python_version}'
        source_dir.mkdir(parents=True, exist_ok=True)
        lock_file_relative_path = 'airflow/www/yarn.lock'
        download_file_from_github(tag=self.airflow_version, path=lock_file_relative_path, output_file=source_dir / 'yarn.lock')
        if not download_constraints_file(airflow_version=self.airflow_version, python_version=self.python_version, include_provider_dependencies=self.include_provider_dependencies, output_file=source_dir / 'requirements.txt'):
            get_console(output=output).print(f'[warning]Failed to download constraints file for {self.airflow_version} and {self.python_version}. Skipping')
            return False
        return True

    def produce(self, output: Output | None, port: int) -> tuple[int, str]:
        if False:
            for i in range(10):
                print('nop')
        import requests
        get_console(output=output).print(f'[info]Updating sbom for Airflow {self.airflow_version} and python {self.python_version}')
        if not self.download_dependency_files(output):
            return (0, f'SBOM Generate {self.airflow_version}:{self.python_version}')
        get_console(output=output).print(f'[info]Generating sbom for Airflow {self.airflow_version} and python {self.python_version} with cdxgen')
        url = f'http://127.0.0.1:{port}/sbom?path=/app/{self.airflow_version}/python{self.python_version}&project-name=apache-airflow&project-version={self.airflow_version}&multiProject=true'
        get_console(output=output).print(f'[info]Triggering sbom generation in {self.airflow_version} via {url}')
        if not get_dry_run():
            response = requests.get(url)
            if response.status_code != 200:
                get_console(output=output).print(f'[error]Generation for Airflow {self.airflow_version}:python{self.python_version} failed. Status code {response.status_code}')
                return (response.status_code, f'SBOM Generate {self.airflow_version}:python{self.python_version}')
            self.target_path.write_bytes(response.content)
            get_console(output=output).print(f'[success]Generated SBOM for {self.airflow_version}:python{self.python_version}')
        return (0, f'SBOM Generate {self.airflow_version}:python{self.python_version}')

@dataclass
class SbomProviderJob(SbomApplicationJob):
    provider_id: str
    provider_version: str
    folder_name: str

    def get_job_name(self) -> str:
        if False:
            print('Hello World!')
        return f'{self.provider_id}:{self.provider_version}:python{self.python_version}'

    def produce(self, output: Output | None, port: int) -> tuple[int, str]:
        if False:
            print('Hello World!')
        import requests
        get_console(output=output).print(f'[info]Updating sbom for provider {self.provider_id} version {self.provider_version} and python {self.python_version}')
        get_console(output=output).print(f'[info]Generating sbom for provider {self.provider_id} version {self.provider_version} and python {self.python_version}')
        url = f'http://127.0.0.1:{port}/sbom?path=/app/{TARGET_DIR_NAME}/{self.folder_name}/python{self.python_version}/without-core&project-name={self.provider_version}&project-version={self.provider_version}&multiProject=true'
        get_console(output=output).print(f'[info]Triggering sbom generation via {url}')
        if not get_dry_run():
            response = requests.get(url)
            if response.status_code != 200:
                get_console(output=output).print(f'[error]Generation for Airflow {self.provider_id}:{self.provider_version}:{self.python_version} failed. Status code {response.status_code}')
                return (response.status_code, f'SBOM Generate {self.provider_id}:{self.provider_version}:{self.python_version}')
            self.target_path.write_bytes(response.content)
            get_console(output=output).print(f'[success]Generated SBOM for {self.provider_id}:{self.provider_version}:{self.python_version}')
        return (0, f'SBOM Generate {self.provider_id}:{self.provider_version}:{self.python_version}')

def produce_sbom_for_application_via_cdxgen_server(job: SbomApplicationJob, output: Output | None, port_map: dict[str, int] | None=None) -> tuple[int, str]:
    if False:
        i = 10
        return i + 15
    '\n    Produces SBOM for application using cdxgen server.\n    :param job: Job to run\n    :param output: Output to use\n    :param port_map map of process name to port - making sure that one process talks to one server\n         in case parallel processing is used\n    :return: tuple with exit code and output\n    '
    if port_map is None:
        port = 9090
    else:
        port = port_map[multiprocessing.current_process().name]
        get_console(output=output).print(f'[info]Using port {port}')
    return job.produce(output, port)