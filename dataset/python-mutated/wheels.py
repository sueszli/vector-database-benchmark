import importlib
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
import tempfile
import time
import urllib.request
from typing import Optional, List, Tuple
from ray_release.config import parse_python_version
from ray_release.test import DEFAULT_PYTHON_VERSION
from ray_release.template import set_test_env_var
from ray_release.exception import RayWheelsUnspecifiedError, RayWheelsNotFoundError, RayWheelsTimeoutError, ReleaseTestSetupError
from ray_release.logger import logger
from ray_release.util import url_exists, python_version_str, resolve_url
from ray_release.aws import upload_to_s3
DEFAULT_BRANCH = 'master'
DEFAULT_GIT_OWNER = 'ray-project'
DEFAULT_GIT_PACKAGE = 'ray'
RELEASE_MANUAL_WHEEL_BUCKET = 'ray-release-manual-wheels'
REPO_URL_TPL = 'https://github.com/{owner}/{package}.git'
INIT_URL_TPL = 'https://raw.githubusercontent.com/{fork}/{commit}/python/ray/__init__.py'
VERSION_URL_TPL = 'https://raw.githubusercontent.com/{fork}/{commit}/python/ray/_version.py'
DEFAULT_REPO = REPO_URL_TPL.format(owner=DEFAULT_GIT_OWNER, package=DEFAULT_GIT_PACKAGE)
RELOAD_MODULES = ['ray', 'ray.job_submission']

def get_ray_version(repo_url: str, commit: str) -> str:
    if False:
        return 10
    assert 'https://github.com/' in repo_url
    (_, fork) = repo_url.split('https://github.com/', maxsplit=2)
    if fork.endswith('.git'):
        fork = fork[:-4]
    init_url = INIT_URL_TPL.format(fork=fork, commit=commit)
    version = ''
    try:
        for line in urllib.request.urlopen(init_url):
            line = line.decode('utf-8')
            if line.startswith('__version__ = '):
                version = line[len('__version__ = '):].strip('"\r\n ')
                break
    except Exception as e:
        raise ReleaseTestSetupError(f"Couldn't load version info from branch URL: {init_url}") from e
    if version == '_version.version':
        u = VERSION_URL_TPL.format(fork=fork, commit=commit)
        try:
            for line in urllib.request.urlopen(u):
                line = line.decode('utf-8')
                if line.startswith('version = '):
                    version = line[len('version = '):].strip('"\r\n ')
                    break
        except Exception as e:
            raise ReleaseTestSetupError(f"Couldn't load version info from branch URL: {init_url}") from e
    if version == '':
        raise RayWheelsNotFoundError(f'Unable to parse Ray version information for repo {repo_url} and commit {commit} (please check this URL: {init_url})')
    return version

def get_latest_commits(repo_url: str, branch: str='master', ref: Optional[str]=None) -> List[str]:
    if False:
        print('Hello World!')
    cur = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        clone_cmd = ['git', 'clone', '--filter=tree:0', '--no-checkout', f'--branch={branch}', repo_url, tmpdir]
        log_cmd = ['git', 'log', '-n', '10', '--pretty=format:%H']
        subprocess.check_output(clone_cmd)
        if ref:
            subprocess.check_output(['git', 'checkout', ref])
        commits = subprocess.check_output(log_cmd).decode(sys.stdout.encoding).split('\n')
    os.chdir(cur)
    return commits

def get_wheels_filename(ray_version: str, python_version: Tuple[int, int]=DEFAULT_PYTHON_VERSION) -> str:
    if False:
        for i in range(10):
            print('nop')
    version_str = python_version_str(python_version)
    suffix = 'm' if python_version[1] <= 7 else ''
    return f'ray-{ray_version}-cp{version_str}-cp{version_str}{suffix}-manylinux2014_x86_64.whl'

def parse_wheels_filename(filename: str) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
    if False:
        i = 10
        return i + 15
    'Parse filename and return Ray version + python version'
    matched = re.search('ray-([0-9a-z\\.]+)-cp([0-9]{2,3})-cp([0-9]{2,3})m?-manylinux2014_x86_64\\.whl$', filename)
    if not matched:
        return (None, None)
    ray_version = matched.group(1)
    py_version_str = matched.group(2)
    try:
        python_version = parse_python_version(py_version_str)
    except Exception:
        return (ray_version, None)
    return (ray_version, python_version)

def get_ray_wheels_url_from_local_wheel(ray_wheels: str) -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    'Upload a local wheel file to S3 and return the downloadable URI\n\n    The uploaded object will have local user and current timestamp encoded\n    in the upload key path, e.g.:\n    "ubuntu_2022_01_01_23:59:99/ray-3.0.0.dev0-cp37-cp37m-manylinux_x86_64.whl"\n\n    Args:\n        ray_wheels: File path with `file://` prefix.\n\n    Return:\n        Downloadable HTTP URL to the uploaded wheel on S3.\n    '
    wheel_path = ray_wheels[len('file://'):]
    wheel_name = os.path.basename(wheel_path)
    if not os.path.exists(wheel_path):
        logger.error(f'Local wheel file: {wheel_path} not found')
        return None
    bucket = RELEASE_MANUAL_WHEEL_BUCKET
    unique_dest_path_prefix = f"{os.getlogin()}_{datetime.now().strftime('%Y_%m_%d_%H:%M:%S')}"
    key_path = f'{unique_dest_path_prefix}/{wheel_name}'
    return upload_to_s3(wheel_path, bucket, key_path)

def get_ray_wheels_url(repo_url: str, branch: str, commit: str, ray_version: str, python_version: Tuple[int, int]=DEFAULT_PYTHON_VERSION) -> str:
    if False:
        i = 10
        return i + 15
    if not repo_url.startswith('https://github.com/ray-project/ray'):
        return f'https://ray-ci-artifact-pr-public.s3.amazonaws.com/{commit}/tmp/artifacts/.whl/{get_wheels_filename(ray_version, python_version)}'
    return f'https://s3-us-west-2.amazonaws.com/ray-wheels/{branch}/{commit}/{get_wheels_filename(ray_version, python_version)}'

def wait_for_url(url, timeout: float=300.0, check_time: float=30.0, status_time: float=60.0) -> str:
    if False:
        for i in range(10):
            print('nop')
    start_time = time.monotonic()
    timeout_at = start_time + timeout
    next_status = start_time + status_time
    logger.info(f'Waiting up to {timeout} seconds until URL is available ({url})')
    while not url_exists(url):
        now = time.monotonic()
        if now >= timeout_at:
            raise RayWheelsTimeoutError(f'Time out when waiting for URL to be available: {url}')
        if now >= next_status:
            logger.info(f'... still waiting for URL {url} ({int(now - start_time)} seconds) ...')
            next_status += status_time
        time.sleep(check_time)
    logger.info(f'URL is now available: {url}')
    return url

def find_and_wait_for_ray_wheels_url(ray_wheels: Optional[str]=None, python_version: Tuple[int, int]=DEFAULT_PYTHON_VERSION, timeout: float=3600.0) -> str:
    if False:
        for i in range(10):
            print('nop')
    ray_wheels_url = find_ray_wheels_url(ray_wheels, python_version=python_version)
    logger.info(f'Using Ray wheels URL: {ray_wheels_url}')
    return wait_for_url(ray_wheels_url, timeout=timeout)

def get_buildkite_repo_branch() -> Tuple[str, str]:
    if False:
        return 10
    if 'BUILDKITE_BRANCH' not in os.environ:
        return (DEFAULT_REPO, DEFAULT_BRANCH)
    branch_str = os.environ['BUILDKITE_BRANCH']
    repo_url = os.environ.get('BUILDKITE_PULL_REQUEST_REPO', None) or os.environ.get('BUILDKITE_REPO', DEFAULT_REPO)
    if ':' in branch_str:
        (owner, branch) = branch_str.split(':', maxsplit=1)
        if not os.environ.get('BUILDKITE_PULL_REQUEST_REPO'):
            repo_url = f'https://github.com/{owner}/ray.git'
    else:
        branch = branch_str
    repo_url = repo_url.replace('git://', 'https://')
    return (repo_url, branch)

def find_ray_wheels_url(ray_wheels: Optional[str]=None, python_version: Tuple[int, int]=DEFAULT_PYTHON_VERSION) -> str:
    if False:
        i = 10
        return i + 15
    if not ray_wheels:
        commit = os.environ.get('BUILDKITE_COMMIT', None)
        if not commit:
            raise RayWheelsUnspecifiedError('No Ray wheels specified. Pass `--ray-wheels` or set `BUILDKITE_COMMIT` environment variable. Hint: You can use `-ray-wheels master` to fetch the latest available master wheels.')
        (repo_url, branch) = get_buildkite_repo_branch()
        if not re.match('\\b([a-f0-9]{40})\\b', commit):
            latest_commits = get_latest_commits(repo_url, branch, ref=commit)
            commit = latest_commits[0]
        ray_version = get_ray_version(repo_url, commit)
        set_test_env_var('RAY_COMMIT', commit)
        set_test_env_var('RAY_BRANCH', branch)
        set_test_env_var('RAY_VERSION', ray_version)
        return get_ray_wheels_url(repo_url, branch, commit, ray_version, python_version)
    if ray_wheels.startswith('file://'):
        logger.info(f'Getting wheel url from local wheel file: {ray_wheels}')
        ray_wheels_url = get_ray_wheels_url_from_local_wheel(ray_wheels)
        if ray_wheels_url is None:
            raise RayWheelsNotFoundError(f"Couldn't get wheel urls from local wheel file({ray_wheels}) by uploading it to S3.")
        return ray_wheels_url
    if ray_wheels.startswith('https://') or ray_wheels.startswith('http://'):
        ray_wheels_url = maybe_rewrite_wheels_url(ray_wheels, python_version=python_version)
        return ray_wheels_url
    if ':' in ray_wheels:
        (owner_or_url, commit_or_branch) = ray_wheels.split(':')
    else:
        owner_or_url = DEFAULT_GIT_OWNER
        commit_or_branch = ray_wheels
    if 'https://' in owner_or_url:
        repo_url = owner_or_url
    else:
        repo_url = REPO_URL_TPL.format(owner=owner_or_url, package=DEFAULT_GIT_PACKAGE)
    if not re.match('\\b([a-f0-9]{40})\\b', commit_or_branch):
        branch = commit_or_branch
        latest_commits = get_latest_commits(repo_url, branch)
        ray_version = get_ray_version(repo_url, latest_commits[0])
        for commit in latest_commits:
            try:
                wheels_url = get_ray_wheels_url(repo_url, branch, commit, ray_version, python_version)
            except Exception as e:
                logger.info(f'Commit not found for PR: {e}')
                continue
            if url_exists(wheels_url):
                set_test_env_var('RAY_COMMIT', commit)
                return wheels_url
            else:
                logger.info(f'Wheels URL for commit {commit} does not exist: {wheels_url}')
        raise RayWheelsNotFoundError(f"Couldn't find latest available wheels for repo {repo_url}, branch {branch} (version {ray_version}). Try again later or check Buildkite logs if wheel builds failed.")
    commit = commit_or_branch
    ray_version = get_ray_version(repo_url, commit)
    branch = os.environ.get('BUILDKITE_BRANCH', DEFAULT_BRANCH)
    wheels_url = get_ray_wheels_url(repo_url, branch, commit, ray_version, python_version)
    set_test_env_var('RAY_COMMIT', commit)
    set_test_env_var('RAY_BRANCH', branch)
    set_test_env_var('RAY_VERSION', ray_version)
    return wheels_url

def maybe_rewrite_wheels_url(ray_wheels_url: str, python_version: Tuple[int, int]) -> str:
    if False:
        return 10
    full_url = resolve_url(ray_wheels_url)
    if is_wheels_url_matching_ray_verison(ray_wheels_url=full_url, python_version=python_version):
        return full_url
    (parsed_ray_version, parsed_python_version) = parse_wheels_filename(full_url)
    if not parsed_ray_version or not python_version:
        logger.warning(f'The passed Ray wheels URL may not work with the python version used in this test! Got python version {python_version} and wheels URL: {ray_wheels_url}.')
        return full_url
    current_filename = get_wheels_filename(parsed_ray_version, parsed_python_version)
    rewritten_filename = get_wheels_filename(parsed_ray_version, python_version)
    new_url = full_url.replace(current_filename, rewritten_filename)
    if new_url != full_url:
        logger.warning(f'The passed Ray wheels URL were for a different python version than used in this test! Found python version {parsed_python_version} but expected {python_version}. The wheels URL was re-written to {new_url}.')
    return new_url

def is_wheels_url_matching_ray_verison(ray_wheels_url: str, python_version: Tuple[int, int]) -> bool:
    if False:
        i = 10
        return i + 15
    'Return True if the wheels URL wheel matches the supplied python version.'
    expected_filename = get_wheels_filename(ray_version='xxx', python_version=python_version)
    expected_filename = expected_filename[7:]
    return ray_wheels_url.endswith(expected_filename)

def install_matching_ray_locally(ray_wheels: Optional[str]):
    if False:
        print('Hello World!')
    if not ray_wheels:
        logger.warning("No Ray wheels found - can't install matching Ray wheels locally!")
        return
    assert 'manylinux2014_x86_64' in ray_wheels, ray_wheels
    if sys.platform == 'darwin':
        platform = 'macosx_10_15_intel'
    elif sys.platform == 'win32':
        platform = 'win_amd64'
    else:
        platform = 'manylinux2014_x86_64'
    ray_wheels = ray_wheels.replace('manylinux2014_x86_64', platform)
    logger.info(f'Installing matching Ray wheels locally: {ray_wheels}')
    subprocess.check_output('pip uninstall -y ray', shell=True, env=os.environ, text=True)
    subprocess.check_output(f'pip install -U {shlex.quote(ray_wheels)}', shell=True, env=os.environ, text=True)
    for module_name in RELOAD_MODULES:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])

def parse_commit_from_wheel_url(url: str) -> str:
    if False:
        while True:
            i = 10
    regex = '/([0-9a-f]{40})/'
    match = re.search(regex, url)
    if match:
        return match.group(1)