"""
Useful tools for various Paths used inside Airflow Sources.
"""
from __future__ import annotations
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from airflow_breeze import NAME
from airflow_breeze.utils.console import get_console
from airflow_breeze.utils.reinstall import reinstall_breeze, warn_dependencies_changed, warn_non_editable
from airflow_breeze.utils.shared_options import get_verbose, set_forced_answer
AIRFLOW_CFG_FILE = 'setup.cfg'

def search_upwards_for_airflow_sources_root(start_from: Path) -> Path | None:
    if False:
        i = 10
        return i + 15
    root = Path(start_from.root)
    d = start_from
    while d != root:
        attempt = d / AIRFLOW_CFG_FILE
        if attempt.exists() and 'name = apache-airflow\n' in attempt.read_text():
            return attempt.parent
        d = d.parent
    return None

def in_autocomplete() -> bool:
    if False:
        for i in range(10):
            print('nop')
    return os.environ.get(f'_{NAME.upper()}_COMPLETE') is not None

def in_self_upgrade() -> bool:
    if False:
        return 10
    return 'self-upgrade' in sys.argv

def in_help() -> bool:
    if False:
        return 10
    return '--help' in sys.argv or '-h' in sys.argv

def skip_upgrade_check():
    if False:
        print('Hello World!')
    return in_self_upgrade() or in_autocomplete() or in_help() or hasattr(sys, '_called_from_test') or os.environ.get('SKIP_UPGRADE_CHECK')

def skip_group_output():
    if False:
        for i in range(10):
            print('nop')
    return in_autocomplete() or in_help() or os.environ.get('SKIP_GROUP_OUTPUT') is not None

def get_package_setup_metadata_hash() -> str:
    if False:
        i = 10
        return i + 15
    "\n    Retrieves hash of setup files from the source of installation of Breeze.\n\n    This is used in order to determine if we need to upgrade Breeze, because some\n    setup files changed. Blake2b algorithm will not be flagged by security checkers\n    as insecure algorithm (in Python 3.9 and above we can use `usedforsecurity=False`\n    to disable it, but for now it's better to use more secure algorithms.\n    "
    try:
        from importlib.metadata import distribution
    except ImportError:
        from importlib_metadata import distribution
    prefix = 'Package config hash: '
    for line in distribution('apache-airflow-breeze').metadata.as_string().splitlines(keepends=False):
        if line.startswith(prefix):
            return line[len(prefix):]
    return 'NOT FOUND'

def get_sources_setup_metadata_hash(sources: Path) -> str:
    if False:
        while True:
            i = 10
    try:
        the_hash = hashlib.new('blake2b')
        the_hash.update((sources / 'dev' / 'breeze' / 'pyproject.toml').read_bytes())
        return the_hash.hexdigest()
    except FileNotFoundError as e:
        return f'Missing file {e.filename}'

def get_installation_sources_config_metadata_hash() -> str:
    if False:
        print('Hello World!')
    "\n    Retrieves hash of setup.py and setup.cfg files from the source of installation of Breeze.\n\n    This is used in order to determine if we need to upgrade Breeze, because some\n    setup files changed. Blake2b algorithm will not be flagged by security checkers\n    as insecure algorithm (in Python 3.9 and above we can use `usedforsecurity=False`\n    to disable it, but for now it's better to use more secure algorithms.\n    "
    installation_sources = get_installation_airflow_sources()
    if installation_sources is None:
        return 'NOT FOUND'
    return get_sources_setup_metadata_hash(installation_sources)

def get_used_sources_setup_metadata_hash() -> str:
    if False:
        i = 10
        return i + 15
    '\n    Retrieves hash of setup files from the currently used sources.\n    '
    return get_sources_setup_metadata_hash(get_used_airflow_sources())

def set_forced_answer_for_upgrade_check():
    if False:
        i = 10
        return i + 15
    'When we run upgrade check --answer is not parsed yet, so we need to guess it.'
    if '--answer n' in ' '.join(sys.argv).lower() or os.environ.get('ANSWER', '').lower().startswith('n'):
        set_forced_answer('no')
    if '--answer y' in ' '.join(sys.argv).lower() or os.environ.get('ANSWER', '').lower().startswith('y'):
        set_forced_answer('yes')
    if '--answer q' in ' '.join(sys.argv).lower() or os.environ.get('ANSWER', '').lower().startswith('q'):
        set_forced_answer('quit')

def process_breeze_readme(breeze_sources: Path, sources_hash: str):
    if False:
        for i in range(10):
            print('nop')
    breeze_readme = breeze_sources / 'README.md'
    lines = breeze_readme.read_text().splitlines(keepends=True)
    result_lines = []
    for line in lines:
        if line.startswith('Package config hash:'):
            line = f'Package config hash: {sources_hash}\n'
        result_lines.append(line)
    breeze_readme.write_text(''.join(result_lines))

def reinstall_if_setup_changed() -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Prints warning if detected airflow sources are not the ones that Breeze was installed with.\n    :return: True if warning was printed.\n    '
    try:
        package_hash = get_package_setup_metadata_hash()
    except ModuleNotFoundError as e:
        if 'importlib_metadata' in e.msg:
            return False
        if 'apache-airflow-breeze' in e.msg:
            print('Missing Package `apache-airflow-breeze`.\n                   Use `pipx install -e ./dev/breeze` to install the package.')
            return False
    sources_hash = get_installation_sources_config_metadata_hash()
    if sources_hash != package_hash:
        installation_sources = get_installation_airflow_sources()
        if installation_sources is not None:
            breeze_sources = installation_sources / 'dev' / 'breeze'
            warn_dependencies_changed()
            process_breeze_readme(breeze_sources, sources_hash)
            set_forced_answer_for_upgrade_check()
            reinstall_breeze(breeze_sources)
            set_forced_answer(None)
        return True
    return False

def reinstall_if_different_sources(airflow_sources: Path) -> bool:
    if False:
        return 10
    '\n    Prints warning if detected airflow sources are not the ones that Breeze was installed with.\n    :param airflow_sources: source for airflow code that we are operating on\n    :return: True if warning was printed.\n    '
    installation_airflow_sources = get_installation_airflow_sources()
    if installation_airflow_sources and airflow_sources != installation_airflow_sources:
        reinstall_breeze(airflow_sources / 'dev' / 'breeze')
        return True
    return False

def get_installation_airflow_sources() -> Path | None:
    if False:
        print('Hello World!')
    '\n    Retrieves the Root of the Airflow Sources where Breeze was installed from.\n    :return: the Path for Airflow sources.\n    '
    return search_upwards_for_airflow_sources_root(Path(__file__).resolve().parent)

def get_used_airflow_sources() -> Path:
    if False:
        for i in range(10):
            print('nop')
    '\n    Retrieves the Root of used Airflow Sources which we operate on. Those are either Airflow sources found\n    upwards in directory tree or sources where Breeze was installed from.\n    :return: the Path for Airflow sources we use.\n    '
    current_sources = search_upwards_for_airflow_sources_root(Path.cwd())
    if current_sources is None:
        current_sources = get_installation_airflow_sources()
        if current_sources is None:
            warn_non_editable()
            sys.exit(1)
    return current_sources

@lru_cache(maxsize=None)
def find_airflow_sources_root_to_operate_on() -> Path:
    if False:
        while True:
            i = 10
    '\n    Find the root of airflow sources we operate on. Handle the case when Breeze is installed via `pipx` from\n    a different source tree, so it searches upwards of the current directory to find the right root of\n    airflow directory we are actually in. This **might** be different than the sources of Airflow Breeze\n    was installed from.\n\n    If not found, we operate on Airflow sources that we were installed it. This handles the case when\n    we run Breeze from a "random" directory.\n\n    This method also handles the following errors and warnings:\n\n       * It fails (and exits hard) if Breeze is installed in non-editable mode (in which case it will\n         not find the Airflow sources when walking upwards the directory where it is installed)\n       * It warns (with 2 seconds timeout) if you are using Breeze from a different airflow sources than\n         the one you operate on.\n       * If we are running in the same source tree as where Breeze was installed from (so no warning above),\n         it warns (with 2 seconds timeout) if there is a change in setup.* files of Breeze since installation\n         time. In such case usesr is encouraged to re-install Breeze to update dependencies.\n\n    :return: Path for the found sources.\n\n    '
    sources_root_from_env = os.getenv('AIRFLOW_SOURCES_ROOT', None)
    if sources_root_from_env:
        return Path(sources_root_from_env)
    installation_airflow_sources = get_installation_airflow_sources()
    if installation_airflow_sources is None and (not skip_upgrade_check()):
        get_console().print(f'\n[error]Breeze should only be installed with -e flag[/]\n\n[warning]Please go to Airflow sources and run[/]\n\n     {NAME} setup self-upgrade --use-current-airflow-sources\n[warning]If during installation you see warning starting "Ignoring --editable install",[/]\n[warning]make sure you first downgrade "packaging" package to <23.2, for example by:[/]\n\n     pip install "packaging<23.2"\n\n')
        sys.exit(1)
    airflow_sources = get_used_airflow_sources()
    if not skip_upgrade_check():
        reinstall_if_different_sources(airflow_sources)
        reinstall_if_setup_changed()
    os.chdir(str(airflow_sources))
    return airflow_sources
AIRFLOW_SOURCES_ROOT = find_airflow_sources_root_to_operate_on().resolve()
TESTS_PROVIDERS_ROOT = AIRFLOW_SOURCES_ROOT / 'tests' / 'providers'
SYSTEM_TESTS_PROVIDERS_ROOT = AIRFLOW_SOURCES_ROOT / 'tests' / 'system' / 'providers'
AIRFLOW_PROVIDERS_ROOT = AIRFLOW_SOURCES_ROOT / 'airflow' / 'providers'
DOCS_ROOT = AIRFLOW_SOURCES_ROOT / 'docs'
BUILD_CACHE_DIR = AIRFLOW_SOURCES_ROOT / '.build'
GENERATED_DIR = AIRFLOW_SOURCES_ROOT / 'generated'
CONSTRAINTS_CACHE_DIR = BUILD_CACHE_DIR / 'constraints'
PROVIDER_DEPENDENCIES_JSON_FILE_PATH = GENERATED_DIR / 'provider_dependencies.json'
PROVIDER_METADATA_JSON_FILE_PATH = GENERATED_DIR / 'provider_metadata.json'
WWW_CACHE_DIR = BUILD_CACHE_DIR / 'www'
AIRFLOW_TMP_DIR_PATH = AIRFLOW_SOURCES_ROOT / 'tmp'
WWW_ASSET_COMPILE_LOCK = WWW_CACHE_DIR / '.asset_compile.lock'
WWW_ASSET_OUT_FILE = WWW_CACHE_DIR / 'asset_compile.out'
WWW_ASSET_OUT_DEV_MODE_FILE = WWW_CACHE_DIR / 'asset_compile_dev_mode.out'
DAGS_DIR = AIRFLOW_SOURCES_ROOT / 'dags'
FILES_DIR = AIRFLOW_SOURCES_ROOT / 'files'
FILES_SBOM_DIR = FILES_DIR / 'sbom'
HOOKS_DIR = AIRFLOW_SOURCES_ROOT / 'hooks'
KUBE_DIR = AIRFLOW_SOURCES_ROOT / '.kube'
LOGS_DIR = AIRFLOW_SOURCES_ROOT / 'logs'
DIST_DIR = AIRFLOW_SOURCES_ROOT / 'dist'
DOCS_DIR = AIRFLOW_SOURCES_ROOT / 'docs'
SCRIPTS_CI_DIR = AIRFLOW_SOURCES_ROOT / 'scripts' / 'ci'
DOCKER_CONTEXT_DIR = AIRFLOW_SOURCES_ROOT / 'docker-context-files'
CACHE_TMP_FILE_DIR = tempfile.TemporaryDirectory()
OUTPUT_LOG = Path(CACHE_TMP_FILE_DIR.name, 'out.log')
BREEZE_SOURCES_ROOT = AIRFLOW_SOURCES_ROOT / 'dev' / 'breeze'
MSSQL_TMP_DIR_NAME = '.tmp-mssql'

def create_volume_if_missing(volume_name: str):
    if False:
        for i in range(10):
            print('nop')
    from airflow_breeze.utils.run_utils import run_command
    res_inspect = run_command(cmd=['docker', 'volume', 'inspect', volume_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if res_inspect.returncode != 0:
        result = run_command(cmd=['docker', 'volume', 'create', volume_name], check=False, capture_output=True)
        if result.returncode != 0:
            get_console().print(f'[warning]\nMypy Cache volume could not be created. Continuing, but you should make sure your docker works.\n\nError: {result.stdout}\n')

def create_mypy_volume_if_needed():
    if False:
        while True:
            i = 10
    create_volume_if_missing('mypy-cache-volume')

def create_directories_and_files() -> None:
    if False:
        return 10
    '\n    Creates all directories and files that are needed for Breeze to work via docker-compose.\n    Checks if setup has been updates since last time and proposes to upgrade if so.\n    '
    BUILD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DAGS_DIR.mkdir(parents=True, exist_ok=True)
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    HOOKS_DIR.mkdir(parents=True, exist_ok=True)
    KUBE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LOG.mkdir(parents=True, exist_ok=True)
    (AIRFLOW_SOURCES_ROOT / '.bash_aliases').touch()
    (AIRFLOW_SOURCES_ROOT / '.bash_history').touch()
    (AIRFLOW_SOURCES_ROOT / '.inputrc').touch()

def cleanup_python_generated_files():
    if False:
        for i in range(10):
            print('nop')
    if get_verbose():
        get_console().print('[info]Cleaning .pyc and __pycache__')
    permission_errors = []
    for path in AIRFLOW_SOURCES_ROOT.rglob('*.pyc'):
        try:
            path.unlink()
        except PermissionError:
            permission_errors.append(path)
    for path in AIRFLOW_SOURCES_ROOT.rglob('__pycache__'):
        try:
            shutil.rmtree(path)
        except PermissionError:
            permission_errors.append(path)
    if permission_errors:
        if platform.uname().system.lower() == 'linux':
            get_console().print('[warning]There were files that you could not clean-up:\n')
            get_console().print(permission_errors)
            get_console().print('Please run at earliest convenience:\n[warning]breeze ci fix-ownership[/]\n\nIf you have sudo you can use:\n[warning]breeze ci fix-ownership --use-sudo[/]\n\nThis will fix ownership of those.\nYou can also remove those files manually using sudo.')
        else:
            get_console().print('[warnings]There were files that you could not clean-up:\n')
            get_console().print(permission_errors)
            get_console().print('You can also remove those files manually using sudo.')
    if get_verbose():
        get_console().print('[info]Cleaned')