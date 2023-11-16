from __future__ import annotations
from __future__ import print_function
from collections.abc import Callable
import glob
import os
from pathlib import Path
import sys
import nox
TEST_CONFIG = {'ignored_versions': ['2.7', '3.7', '3.9', '3.10'], 'enforce_type_hints': False, 'gcloud_project_env': 'GOOGLE_CLOUD_PROJECT', 'pip_version_override': None, 'envs': {}}
try:
    sys.path.append('.')
    from noxfile_config import TEST_CONFIG_OVERRIDE
except ImportError as e:
    print('No user noxfile_config found: detail: {}'.format(e))
    TEST_CONFIG_OVERRIDE = {}
TEST_CONFIG.update(TEST_CONFIG_OVERRIDE)

def get_pytest_env_vars() -> dict[str, str]:
    if False:
        while True:
            i = 10
    'Returns a dict for pytest invocation.'
    ret = {}
    env_key = TEST_CONFIG['gcloud_project_env']
    ret['GOOGLE_CLOUD_PROJECT'] = os.environ[env_key]
    ret['GCLOUD_PROJECT'] = os.environ[env_key]
    ret.update(TEST_CONFIG['envs'])
    return ret
ALL_VERSIONS = ['2.7', '3.8', '3.9', '3.10', '3.11']
IGNORED_VERSIONS = TEST_CONFIG['ignored_versions']
TESTED_VERSIONS = sorted([v for v in ALL_VERSIONS if v not in IGNORED_VERSIONS])
INSTALL_LIBRARY_FROM_SOURCE = bool(os.environ.get('INSTALL_LIBRARY_FROM_SOURCE', False))
nox.options.error_on_missing_interpreters = True

def _determine_local_import_names(start_dir: str) -> list[str]:
    if False:
        i = 10
        return i + 15
    'Determines all import names that should be considered "local".\n\n    This is used when running the linter to insure that import order is\n    properly checked.\n    '
    file_ext_pairs = [os.path.splitext(path) for path in os.listdir(start_dir)]
    return [basename for (basename, extension) in file_ext_pairs if extension == '.py' or (os.path.isdir(os.path.join(start_dir, basename)) and basename not in '__pycache__')]
FLAKE8_COMMON_ARGS = ['--show-source', '--builtin=gettext', '--max-complexity=20', '--import-order-style=google', '--exclude=.nox,.cache,env,lib,generated_pb2,*_pb2.py,*_pb2_grpc.py', '--ignore=ANN101,E121,E123,E126,E203,E226,E24,E266,E501,E704,W503,W504,I202', '--max-line-length=88']

@nox.session
def lint(session: nox.sessions.Session) -> None:
    if False:
        for i in range(10):
            print('nop')
    if not TEST_CONFIG['enforce_type_hints']:
        session.install('flake8', 'flake8-import-order')
    else:
        session.install('flake8', 'flake8-import-order', 'flake8-annotations')
    local_names = _determine_local_import_names('.')
    args = FLAKE8_COMMON_ARGS + ['--application-import-names', ','.join(local_names), '.']
    session.run('flake8', *args)

@nox.session
def blacken(session: nox.sessions.Session) -> None:
    if False:
        return 10
    session.install('black')
    python_files = [path for path in os.listdir('.') if path.endswith('.py')]
    session.run('black', *python_files)
PYTEST_COMMON_ARGS = ['--junitxml=sponge_log.xml']

def _session_tests(session: nox.sessions.Session, post_install: Callable=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    test_list = glob.glob('*_test.py') + glob.glob('test_*.py')
    test_list.extend(glob.glob('tests'))
    if len(test_list) == 0:
        print('No tests found, skipping directory.')
        return
    if TEST_CONFIG['pip_version_override']:
        pip_version = TEST_CONFIG['pip_version_override']
        session.install(f'pip=={pip_version}')
    else:
        session.install('--upgrade', 'pip')
    'Runs py.test for a particular project.'
    concurrent_args = []
    if os.path.exists('requirements.txt'):
        with open('requirements.txt') as rfile:
            packages = rfile.read()
        if os.path.exists('constraints.txt'):
            session.install('-r', 'requirements.txt', '-c', 'constraints.txt', '--only-binary', ':all')
        elif 'pyspark' in packages:
            session.install('-r', 'requirements.txt', '--use-pep517')
        else:
            session.install('-r', 'requirements.txt', '--only-binary', ':all')
    if os.path.exists('requirements-test.txt'):
        with open('requirements-test.txt') as rtfile:
            packages += rtfile.read()
        if os.path.exists('constraints-test.txt'):
            session.install('-r', 'requirements-test.txt', '-c', 'constraints-test.txt', '--only-binary', ':all')
        else:
            session.install('-r', 'requirements-test.txt', '--only-binary', ':all')
    if INSTALL_LIBRARY_FROM_SOURCE:
        session.install('-e', _get_repo_root())
    if post_install:
        post_install(session)
    if 'pytest-parallel' in packages:
        concurrent_args.extend(['--workers', 'auto', '--tests-per-worker', 'auto'])
    elif 'pytest-xdist' in packages:
        concurrent_args.extend(['-n', 'auto'])
    session.run('pytest', *PYTEST_COMMON_ARGS + session.posargs + concurrent_args, success_codes=[0, 5], env=get_pytest_env_vars())

@nox.session(python=ALL_VERSIONS)
def py(session: nox.sessions.Session) -> None:
    if False:
        while True:
            i = 10
    'Runs py.test for a sample using the specified version of Python.'
    if session.python in TESTED_VERSIONS:
        _session_tests(session)
    else:
        session.skip('SKIPPED: {} tests are disabled for this sample.'.format(session.python))

def _get_repo_root() -> str | None:
    if False:
        print('Hello World!')
    'Returns the root folder of the project.'
    p = Path(os.getcwd())
    for i in range(10):
        if p is None:
            break
        if Path(p / '.git').exists():
            return str(p)
        p = p.parent
    raise Exception('Unable to detect repository root.')
GENERATED_READMES = sorted([x for x in Path('.').rglob('*.rst.in')])

@nox.session
@nox.parametrize('path', GENERATED_READMES)
def readmegen(session: nox.sessions.Session, path: str) -> None:
    if False:
        print('Hello World!')
    '(Re-)generates the readme for a sample.'
    session.install('jinja2', 'pyyaml')
    dir_ = os.path.dirname(path)
    if os.path.exists(os.path.join(dir_, 'requirements.txt')):
        session.install('-r', os.path.join(dir_, 'requirements.txt'))
    in_file = os.path.join(dir_, 'README.rst.in')
    session.run('python', _get_repo_root() + '/scripts/readme-gen/readme_gen.py', in_file)