import os
import subprocess
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence
import dagster._check as check
import docker
import packaging.version
import pytest
import requests
from dagster._core.storage.dagster_run import DagsterRunStatus
from dagster._utils import file_relative_path, library_version_from_core_version, parse_package_version
from dagster_graphql import DagsterGraphQLClient
DAGSTER_CURRENT_BRANCH = 'current_branch'
MAX_TIMEOUT_SECONDS = 20
IS_BUILDKITE = os.getenv('BUILDKITE') is not None
EARLIEST_TESTED_RELEASE = os.getenv('EARLIEST_TESTED_RELEASE')
MOST_RECENT_RELEASE_PLACEHOLDER = 'most_recent'
pytest_plugins = ['dagster_test.fixtures']
MARK_TO_VERSIONS_MAP = {'webserver-earliest-release': (EARLIEST_TESTED_RELEASE, DAGSTER_CURRENT_BRANCH), 'user-code-earliest-release': (DAGSTER_CURRENT_BRANCH, EARLIEST_TESTED_RELEASE), 'webserver-latest-release': (MOST_RECENT_RELEASE_PLACEHOLDER, DAGSTER_CURRENT_BRANCH), 'user-code-latest-release': (DAGSTER_CURRENT_BRANCH, MOST_RECENT_RELEASE_PLACEHOLDER)}

def get_library_version(version: str) -> str:
    if False:
        i = 10
        return i + 15
    if version == DAGSTER_CURRENT_BRANCH:
        return DAGSTER_CURRENT_BRANCH
    else:
        return library_version_from_core_version(version)

def is_0_release(release: str) -> bool:
    if False:
        while True:
            i = 10
    'Returns true if on < 1.0 release of dagster, false otherwise.'
    if release == 'current_branch':
        return False
    version = packaging.version.parse(release)
    return version < packaging.version.Version('1.0')

def infer_user_code_definitions_files(release: str) -> str:
    if False:
        print('Hello World!')
    'Returns `repo.py` if on source or version >=1.0, `legacy_repo.py` otherwise.'
    if release == 'current_branch':
        return 'repo.py'
    else:
        version = packaging.version.parse(release)
        return 'legacy_repo.py' if version < packaging.version.Version('1.0') else 'repo.py'

def infer_webserver_package(release: str) -> str:
    if False:
        print('Hello World!')
    'Returns `dagster-webserver` if on source or version >=1.3.14 (first dagster-webserver\n    release), `dagit` otherwise.\n    '
    if release == 'current_branch':
        return 'dagster-webserver'
    else:
        if not EARLIEST_TESTED_RELEASE:
            check.failed('Environment variable `$EARLIEST_TESTED_RELEASE` must be set.')
        version = packaging.version.parse(release)
        return 'dagit' if version < packaging.version.Version('1.3.14') else 'dagster-webserver'

def assert_run_success(client: DagsterGraphQLClient, run_id: str) -> None:
    if False:
        return 10
    start_time = time.time()
    while True:
        if time.time() - start_time > MAX_TIMEOUT_SECONDS:
            raise Exception('Timed out waiting for launched run to complete')
        status = client.get_run_status(run_id)
        assert status and status != DagsterRunStatus.FAILURE
        if status == DagsterRunStatus.SUCCESS:
            break
        time.sleep(1)

@pytest.fixture(name='dagster_most_recent_release', scope='session')
def dagster_most_recent_release() -> str:
    if False:
        print('Hello World!')
    res = requests.get('https://pypi.org/pypi/dagster/json')
    module_json = res.json()
    releases = module_json['releases']
    release_versions = [parse_package_version(version) for (version, files) in releases.items() if not any((file.get('yanked') for file in files))]
    for release_version in reversed(sorted(release_versions)):
        if not release_version.is_prerelease:
            return str(release_version)
    check.failed('No non-prerelease releases found')

@pytest.fixture(params=[pytest.param(value, marks=getattr(pytest.mark, key), id=key) for (key, value) in MARK_TO_VERSIONS_MAP.items()], scope='session')
def release_test_map(request, dagster_most_recent_release: str) -> Mapping[str, str]:
    if False:
        print('Hello World!')
    webserver_version = request.param[0]
    if webserver_version == MOST_RECENT_RELEASE_PLACEHOLDER:
        webserver_version = dagster_most_recent_release
    user_code_version = request.param[1]
    if user_code_version == MOST_RECENT_RELEASE_PLACEHOLDER:
        user_code_version = dagster_most_recent_release
    return {'webserver': webserver_version, 'user_code': user_code_version}

def check_webserver_connection(host: str, webserver_package: str, retrying_requests) -> None:
    if False:
        return 10
    if webserver_package == 'dagit':
        url_path = 'dagit_info'
        json_key = 'dagit_version'
    else:
        url_path = 'server_info'
        json_key = 'dagster_webserver_version'
    result = retrying_requests.get(f'http://{host}:3000/{url_path}')
    assert result.json().get(json_key)

def upload_docker_logs_to_buildkite():
    if False:
        for i in range(10):
            print('nop')
    client = docker.client.from_env()
    containers = client.containers.list()
    current_test = os.environ['PYTEST_CURRENT_TEST'].split(':')[-1].split(' ')[0]
    logs_dir = f'.docker_logs/{current_test}'
    p = subprocess.Popen(['rm', '-rf', f'{logs_dir}'])
    p.communicate()
    assert p.returncode == 0
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    for c in containers:
        with open(f'{logs_dir}/{c.name}-logs.txt', 'w', encoding='utf8') as log:
            p = subprocess.Popen(['docker', 'logs', c.name], stdout=log, stderr=log)
            p.communicate()
            print(f'container({c.name}) logs dumped')
            if p.returncode != 0:
                q = subprocess.Popen(['docker', 'logs', c.name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                (stdout, stderr) = q.communicate()
                print(f'{c.name} container log dump failed with stdout: ', stdout)
                print(f'{c.name} container logs dump failed with stderr: ', stderr)
    p = subprocess.Popen(['buildkite-agent', 'artifact', 'upload', f'{logs_dir}/**/*'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = p.communicate()
    print('Buildkite artifact added with stdout: ', stdout)
    print('Buildkite artifact added with stderr: ', stderr)

@contextmanager
def docker_service(docker_compose_file: str, webserver_version: str, user_code_version: str) -> Iterator[None]:
    if False:
        i = 10
        return i + 15
    try:
        subprocess.check_output(['docker-compose', '-f', docker_compose_file, 'stop'])
        subprocess.check_output(['docker-compose', '-f', docker_compose_file, 'rm', '-f'])
    except subprocess.CalledProcessError:
        pass
    webserver_library_version = get_library_version(webserver_version)
    webserver_package = infer_webserver_package(webserver_version)
    user_code_library_version = get_library_version(user_code_version)
    user_code_definitions_files = infer_user_code_definitions_files(user_code_version)
    build_process = subprocess.Popen([file_relative_path(docker_compose_file, './build.sh'), webserver_version, webserver_library_version, webserver_package, user_code_version, user_code_library_version, user_code_definitions_files])
    build_process.wait()
    assert build_process.returncode == 0
    env = {'WEBSERVER_PACKAGE': webserver_package, 'USER_CODE_DEFINITIONS_FILE': user_code_definitions_files, **os.environ}
    up_process = subprocess.Popen(['docker-compose', '-f', docker_compose_file, 'up', '--no-start'], env=env)
    up_process.wait()
    assert up_process.returncode == 0
    start_process = subprocess.Popen(['docker-compose', '-f', docker_compose_file, 'start'])
    start_process.wait()
    assert start_process.returncode == 0
    try:
        yield
    except Exception as e:
        print(f'An exception occurred: {e}')
        traceback.print_exc()
        raise e
    finally:
        subprocess.check_output(['docker-compose', '-f', docker_compose_file, 'stop'])
        subprocess.check_output(['docker-compose', '-f', docker_compose_file, 'rm', '-f'])

@pytest.fixture(scope='session')
def graphql_client(release_test_map: Mapping[str, str], retrying_requests) -> Iterator[DagsterGraphQLClient]:
    if False:
        while True:
            i = 10
    webserver_version = release_test_map['webserver']
    webserver_package = infer_webserver_package(webserver_version)
    if IS_BUILDKITE:
        webserver_host = os.environ['BACKCOMPAT_TESTS_WEBSERVER_HOST']
        try:
            check_webserver_connection(webserver_host, webserver_package, retrying_requests)
            yield DagsterGraphQLClient(webserver_host, port_number=3000)
        finally:
            upload_docker_logs_to_buildkite()
    else:
        webserver_host = 'localhost'
        with docker_service(os.path.join(os.getcwd(), 'webserver_service', 'docker-compose.yml'), webserver_version=webserver_version, user_code_version=release_test_map['user_code']):
            print('INSIDE DOCKER SERVICE')
            check_webserver_connection(webserver_host, webserver_package, retrying_requests)
            yield DagsterGraphQLClient(webserver_host, port_number=3000)

def test_backcompat_deployed_pipeline(graphql_client: DagsterGraphQLClient, release_test_map: Mapping[str, str]):
    if False:
        print('Hello World!')
    if is_0_release(release_test_map['user_code']):
        assert_runs_and_exists(graphql_client, 'the_pipeline')

def test_backcompat_deployed_pipeline_subset(graphql_client: DagsterGraphQLClient, release_test_map: Mapping[str, str]):
    if False:
        i = 10
        return i + 15
    if is_0_release(release_test_map['user_code']):
        assert_runs_and_exists(graphql_client, 'the_pipeline', subset_selection=['my_solid'])

def test_backcompat_deployed_job(graphql_client: DagsterGraphQLClient):
    if False:
        for i in range(10):
            print('nop')
    assert_runs_and_exists(graphql_client, 'the_job')

def test_backcompat_deployed_job_subset(graphql_client: DagsterGraphQLClient):
    if False:
        print('Hello World!')
    assert_runs_and_exists(graphql_client, 'the_job', subset_selection=['my_op'])

def test_backcompat_ping_webserver(graphql_client: DagsterGraphQLClient):
    if False:
        while True:
            i = 10
    assert_runs_and_exists(graphql_client, 'test_graphql')

def assert_runs_and_exists(client: DagsterGraphQLClient, name: str, subset_selection: Optional[Sequence[str]]=None):
    if False:
        while True:
            i = 10
    run_id = client.submit_job_execution(job_name=name, run_config={}, op_selection=subset_selection)
    assert_run_success(client, run_id)
    locations = client._get_repo_locations_and_names_with_pipeline(job_name=name)
    assert len(locations) == 1
    assert locations[0].job_name == name