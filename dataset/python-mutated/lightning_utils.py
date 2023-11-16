import functools
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional
from packaging.version import Version
from lightning.app import _PROJECT_ROOT, _logger, _root_logger
from lightning.app import __version__ as version
from lightning.app.core.constants import FRONTEND_DIR, PACKAGE_LIGHTNING
from lightning.app.utilities.app_helpers import Logger
from lightning.app.utilities.git import check_github_repository, get_dir_name
logger = Logger(__name__)
LIGHTNING_FRONTEND_RELEASE_URL = 'https://storage.googleapis.com/grid-packages/lightning-ui/v0.0.0/build.tar.gz'

def download_frontend(root: str=_PROJECT_ROOT):
    if False:
        i = 10
        return i + 15
    'Downloads an archive file for a specific release of the Lightning frontend and extracts it to the correct\n    directory.'
    build_dir = 'build'
    frontend_dir = pathlib.Path(FRONTEND_DIR)
    download_dir = tempfile.mkdtemp()
    shutil.rmtree(frontend_dir, ignore_errors=True)
    response = urllib.request.urlopen(LIGHTNING_FRONTEND_RELEASE_URL)
    file = tarfile.open(fileobj=response, mode='r|gz')
    file.extractall(path=download_dir)
    shutil.move(os.path.join(download_dir, build_dir), frontend_dir)
    print('The Lightning UI has successfully been downloaded!')

def _cleanup(*tar_files: str):
    if False:
        while True:
            i = 10
    for tar_file in tar_files:
        shutil.rmtree(os.path.join(_PROJECT_ROOT, 'dist'), ignore_errors=True)
        os.remove(tar_file)

def _prepare_wheel(path):
    if False:
        while True:
            i = 10
    with open('log.txt', 'w') as logfile:
        with subprocess.Popen(['rm', '-r', 'dist'], stdout=logfile, stderr=logfile, bufsize=0, close_fds=True, cwd=path) as proc:
            proc.wait()
        with subprocess.Popen(['python', 'setup.py', 'sdist'], stdout=logfile, stderr=logfile, bufsize=0, close_fds=True, cwd=path) as proc:
            proc.wait()
    os.remove('log.txt')

def _copy_tar(project_root, dest: Path) -> str:
    if False:
        print('Hello World!')
    dist_dir = os.path.join(project_root, 'dist')
    tar_files = os.listdir(dist_dir)
    assert len(tar_files) == 1
    tar_name = tar_files[0]
    tar_path = os.path.join(dist_dir, tar_name)
    shutil.copy(tar_path, dest)
    return tar_name

def get_dist_path_if_editable_install(project_name) -> str:
    if False:
        print('Hello World!')
    'Is distribution an editable install - modified version from pip that\n    fetches egg-info instead of egg-link'
    for path_item in sys.path:
        if not os.path.isdir(path_item):
            continue
        egg_info = os.path.join(path_item, project_name + '.egg-info')
        if os.path.isdir(egg_info):
            return path_item
    return ''

def _prepare_lightning_wheels_and_requirements(root: Path, package_name: str='lightning') -> Optional[Callable]:
    if False:
        while True:
            i = 10
    'This function determines if lightning is installed in editable mode (for developers) and packages the current\n    lightning source along with the app.\n\n    For normal users who install via PyPi or Conda, then this function does not do anything.\n\n    '
    if not get_dist_path_if_editable_install(package_name):
        return None
    os.environ['PACKAGE_NAME'] = 'app' if package_name == 'lightning' + '_app' else 'lightning'
    git_dir_name = get_dir_name() if check_github_repository() else None
    is_lightning = git_dir_name and git_dir_name == package_name
    if PACKAGE_LIGHTNING is None and (not is_lightning) or PACKAGE_LIGHTNING == '0':
        return None
    download_frontend(_PROJECT_ROOT)
    _prepare_wheel(_PROJECT_ROOT)
    print(f'Packaged Lightning with your application. Version: {version}')
    tar_name = _copy_tar(_PROJECT_ROOT, root)
    tar_files = [os.path.join(root, tar_name)]
    if (PACKAGE_LIGHTNING or is_lightning) and (not bool(int(os.getenv('SKIP_LIGHTING_UTILITY_WHEELS_BUILD', '0')))):
        lightning_cloud_project_path = get_dist_path_if_editable_install('lightning_cloud')
        if lightning_cloud_project_path:
            from lightning_cloud.__version__ import __version__ as cloud_version
            print(f'Packaged Lightning Cloud with your application. Version: {cloud_version}')
            _prepare_wheel(lightning_cloud_project_path)
            tar_name = _copy_tar(lightning_cloud_project_path, root)
            tar_files.append(os.path.join(root, tar_name))
    return functools.partial(_cleanup, *tar_files)

def _enable_debugging():
    if False:
        i = 10
        return i + 15
    tar_file = os.path.join(os.getcwd(), f'lightning-{version}.tar.gz')
    if not os.path.exists(tar_file):
        return
    _root_logger.propagate = True
    _logger.propagate = True
    _root_logger.setLevel(logging.DEBUG)
    _root_logger.debug('Setting debugging mode.')

def enable_debugging(func: Callable) -> Callable:
    if False:
        i = 10
        return i + 15
    'This function is used to transform any print into logger.info calls, so it gets tracked in the cloud.'

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if False:
            print('Hello World!')
        _enable_debugging()
        res = func(*args, **kwargs)
        _logger.setLevel(logging.INFO)
        return res
    return wrapper

def _fetch_latest_version(package_name: str) -> str:
    if False:
        i = 10
        return i + 15
    args = [sys.executable, '-m', 'pip', 'install', f'{package_name}==1000']
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0, close_fds=True)
    if proc.stdout:
        logs = ' '.join([line.decode('utf-8') for line in iter(proc.stdout.readline, b'')])
        return logs.split(')\n')[0].split(',')[-1].replace(' ', '')
    return version

def _verify_lightning_version():
    if False:
        while True:
            i = 10
    'This function verifies that users are running the latest lightning version for the cloud.'
    if sys.platform == 'win32':
        return
    lightning_latest_version = _fetch_latest_version('lightning')
    if Version(lightning_latest_version) > Version(version):
        raise Exception(f'You need to use the latest version of Lightning ({lightning_latest_version}) to run in the cloud. Please, run `pip install -U lightning`')