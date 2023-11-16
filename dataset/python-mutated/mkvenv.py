"""Create a local virtualenv with a PyQt install."""
import argparse
import pathlib
import sys
import re
import os
import os.path
import shutil
import venv as pyvenv
import subprocess
import platform
from typing import List, Tuple, Dict, Union
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from scripts import utils, link_pyqt
REPO_ROOT = pathlib.Path(__file__).parent.parent
PYQT_PACKAGES = ['PyQt5', 'PyQtWebEngine', 'PyQt6', 'PyQt6-WebEngine']

class Error(Exception):
    """Exception for errors in this script."""

    def __init__(self, msg, code=1):
        if False:
            i = 10
            return i + 15
        super().__init__(msg)
        self.code = code

def print_command(*cmd: Union[str, pathlib.Path], venv: bool) -> None:
    if False:
        print('Hello World!')
    'Print a command being run.'
    prefix = 'venv$ ' if venv else '$ '
    utils.print_col(prefix + ' '.join([str(e) for e in cmd]), 'blue')

def parse_args(argv: List[str]=None) -> argparse.Namespace:
    if False:
        i = 10
        return i + 15
    'Parse commandline arguments.'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--update', action='store_true', help="Run 'git pull' before creating the environment.")
    parser.add_argument('--keep', action='store_true', help='Reuse an existing virtualenv.')
    parser.add_argument('--venv-dir', default='.venv', help='Where to place the virtualenv.')
    parser.add_argument('--pyqt-version', choices=pyqt_versions(), default='auto', help='PyQt version to install.')
    parser.add_argument('--pyqt-type', choices=['binary', 'source', 'link', 'wheels', 'skip'], default='binary', help='How to install PyQt/Qt.')
    parser.add_argument('--pyqt-wheels-dir', default='wheels', help='Directory to get PyQt wheels from.')
    parser.add_argument('--pyqt-snapshot', help='Comma-separated list to install from the Riverbank PyQt snapshot server')
    parser.add_argument('--virtualenv', action='store_true', help='Use virtualenv instead of venv.')
    parser.add_argument('--dev', action='store_true', help='Also install dev/test dependencies.')
    parser.add_argument('--skip-docs', action='store_true', help='Skip doc generation.')
    parser.add_argument('--skip-smoke-test', action='store_true', help='Skip Qt smoke test.')
    parser.add_argument('--tox-error', action='store_true', help=argparse.SUPPRESS)
    return parser.parse_args(argv)

def _version_key(v):
    if False:
        while True:
            i = 10
    'Sort PyQt requirement file prefixes.\n\n    If we have a filename like requirements-pyqt-pyinstaller.txt, that should\n    always be sorted after all others (hence we return a "999" key).\n    '
    try:
        return tuple((int(v) for c in v.split('.')))
    except ValueError:
        return (999,)

def pyqt_versions() -> List[str]:
    if False:
        i = 10
        return i + 15
    'Get a list of all available PyQt versions.\n\n    The list is based on the filenames of misc/requirements/ files.\n    '
    version_set = set()
    requirements_dir = REPO_ROOT / 'misc' / 'requirements'
    for req in requirements_dir.glob('requirements-pyqt-*.txt'):
        version_set.add(req.stem.split('-')[-1])
    versions = sorted(version_set, key=_version_key)
    return versions + ['auto']

def _is_qt6_version(version: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Check if the given version is Qt 6.'
    return version in ['auto', '6'] or version.startswith('6.')

def run_venv(venv_dir: pathlib.Path, executable, *args: str, capture_output=False, capture_error=False, env=None) -> subprocess.CompletedProcess:
    if False:
        while True:
            i = 10
    'Run the given command inside the virtualenv.'
    subdir = 'Scripts' if os.name == 'nt' else 'bin'
    if env is None:
        proc_env = None
    else:
        proc_env = os.environ.copy()
        proc_env.update(env)
    try:
        return subprocess.run([str(venv_dir / subdir / executable)] + [str(arg) for arg in args], check=True, text=capture_output or capture_error, stdout=subprocess.PIPE if capture_output else None, stderr=subprocess.PIPE if capture_error else None, env=proc_env)
    except subprocess.CalledProcessError as e:
        raise Error('Subprocess failed, exiting') from e

def pip_install(venv_dir: pathlib.Path, *args: str) -> None:
    if False:
        i = 10
        return i + 15
    'Run a pip install command inside the virtualenv.'
    arg_str = ' '.join((str(arg) for arg in args))
    print_command('pip install', arg_str, venv=True)
    run_venv(venv_dir, 'python', '-m', 'pip', 'install', *args)

def delete_old_venv(venv_dir: pathlib.Path) -> None:
    if False:
        return 10
    'Remove an existing virtualenv directory.'
    if not venv_dir.exists():
        return
    markers = [venv_dir / '.tox-config1', venv_dir / 'pyvenv.cfg', venv_dir / 'Scripts', venv_dir / 'bin']
    if not any((m.exists() for m in markers)):
        raise Error('{} does not look like a virtualenv, cowardly refusing to remove it.'.format(venv_dir))
    print_command('rm -r', venv_dir, venv=False)
    shutil.rmtree(venv_dir)

def create_venv(venv_dir: pathlib.Path, use_virtualenv: bool=False) -> None:
    if False:
        i = 10
        return i + 15
    'Create a new virtualenv.'
    if use_virtualenv:
        print_command('python3 -m virtualenv', venv_dir, venv=False)
        try:
            subprocess.run([sys.executable, '-m', 'virtualenv', venv_dir], check=True)
        except subprocess.CalledProcessError as e:
            raise Error('virtualenv failed, exiting', e.returncode)
    else:
        print_command('python3 -m venv', venv_dir, venv=False)
        pyvenv.create(str(venv_dir), with_pip=True)

def upgrade_seed_pkgs(venv_dir: pathlib.Path) -> None:
    if False:
        return 10
    'Upgrade initial seed packages inside a virtualenv.\n\n    This also makes sure that wheel is installed, which causes pip to use its\n    wheel cache for rebuilds.\n    '
    utils.print_title('Upgrading initial packages')
    pip_install(venv_dir, '-U', 'pip')
    pip_install(venv_dir, '-U', 'setuptools', 'wheel')

def requirements_file(name: str) -> pathlib.Path:
    if False:
        i = 10
        return i + 15
    'Get the filename of a requirements file.'
    return REPO_ROOT / 'misc' / 'requirements' / 'requirements-{}.txt'.format(name)

def pyqt_requirements_file(version: str) -> pathlib.Path:
    if False:
        for i in range(10):
            print('nop')
    'Get the filename of the requirements file for the given PyQt version.'
    name = 'pyqt-6' if version == 'auto' else f'pyqt-{version}'
    return requirements_file(name)

def install_pyqt_binary(venv_dir: pathlib.Path, version: str) -> None:
    if False:
        print('Hello World!')
    'Install PyQt from a binary wheel.'
    utils.print_title('Installing PyQt from binary')
    utils.print_col('No proprietary codec support will be available in qutebrowser.', 'bold')
    if _is_qt6_version(version):
        supported_archs = {'linux': {'x86_64'}, 'win32': {'AMD64'}, 'darwin': {'x86_64', 'arm64'}}
    else:
        supported_archs = {'linux': {'x86_64'}, 'win32': {'x86', 'AMD64'}, 'darwin': {'x86_64'}}
    if sys.platform not in supported_archs:
        utils.print_error(f'{sys.platform} is not a supported platform by PyQt binary packages, this will most likely fail.')
    elif platform.machine() not in supported_archs[sys.platform]:
        utils.print_error(f'{platform.machine()} is not a supported architecture for PyQt binaries on {sys.platform}, this will most likely fail.')
    elif sys.platform == 'linux' and platform.libc_ver()[0] != 'glibc':
        utils.print_error('Non-glibc Linux is not a supported platform for PyQt binaries, this will most likely fail.')
    pip_install(venv_dir, '-r', pyqt_requirements_file(version), '--only-binary', ','.join(PYQT_PACKAGES))

def install_pyqt_source(venv_dir: pathlib.Path, version: str) -> None:
    if False:
        print('Hello World!')
    'Install PyQt from the source tarball.'
    utils.print_title('Installing PyQt from sources')
    pip_install(venv_dir, '-r', pyqt_requirements_file(version), '--verbose', '--no-binary', ','.join(PYQT_PACKAGES))

def install_pyqt_link(venv_dir: pathlib.Path, version: str) -> None:
    if False:
        while True:
            i = 10
    'Install PyQt by linking a system-wide install.'
    utils.print_title('Linking system-wide PyQt')
    lib_path = link_pyqt.get_venv_lib_path(str(venv_dir))
    major_version: str = '6' if _is_qt6_version(version) else '5'
    link_pyqt.link_pyqt(sys.executable, lib_path, version=major_version)

def install_pyqt_wheels(venv_dir: pathlib.Path, wheels_dir: pathlib.Path) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Install PyQt from the wheels/ directory.'
    utils.print_title('Installing PyQt wheels')
    wheels = [str(wheel) for wheel in wheels_dir.glob('*.whl')]
    pip_install(venv_dir, *wheels)

def install_pyqt_snapshot(venv_dir: pathlib.Path, packages: List[str]) -> None:
    if False:
        i = 10
        return i + 15
    'Install PyQt packages from the snapshot server.'
    utils.print_title('Installing PyQt snapshots')
    pip_install(venv_dir, '-U', *packages, '--no-deps', '--pre', '--index-url', 'https://riverbankcomputing.com/pypi/simple/')

def apply_xcb_util_workaround(venv_dir: pathlib.Path, pyqt_type: str, pyqt_version: str) -> None:
    if False:
        while True:
            i = 10
    'If needed (Debian Stable), symlink libxcb-util.so.0 -> .1.\n\n    WORKAROUND for https://bugreports.qt.io/browse/QTBUG-88688\n    '
    utils.print_title('Running xcb-util workaround')
    if not sys.platform.startswith('linux'):
        print('Workaround not needed: Not on Linux.')
        return
    if pyqt_type != 'binary':
        print('Workaround not needed: Not installing from PyQt binaries.')
        return
    if _is_qt6_version(pyqt_version):
        print('Workaround not needed: Not installing Qt 5.15.')
        return
    try:
        libs = _find_libs()
    except subprocess.CalledProcessError as e:
        utils.print_error(f'Workaround failed: ldconfig exited with status {e.returncode}')
        return
    abi_type = 'libc6,x86-64'
    if ('libxcb-util.so.1', abi_type) in libs:
        print('Workaround not needed: libxcb-util.so.1 found.')
        return
    try:
        libxcb_util_libs = libs['libxcb-util.so.0', abi_type]
    except KeyError:
        utils.print_error('Workaround failed: libxcb-util.so.0 not found.')
        return
    if len(libxcb_util_libs) > 1:
        utils.print_error(f'Workaround failed: Multiple matching libxcb-util found: {libxcb_util_libs}')
        return
    libxcb_util_path = pathlib.Path(libxcb_util_libs[0])
    code = ['from PyQt5.QtCore import QLibraryInfo', 'print(QLibraryInfo.location(QLibraryInfo.LibrariesPath))']
    proc = run_venv(venv_dir, 'python', '-c', '; '.join(code), capture_output=True)
    venv_lib_path = pathlib.Path(proc.stdout.strip())
    link_path = venv_lib_path / libxcb_util_path.with_suffix('.1').name
    rel_link_path = venv_dir / link_path.relative_to(venv_dir.resolve())
    print_command('ln -s', libxcb_util_path, rel_link_path, venv=False)
    link_path.symlink_to(libxcb_util_path)

def _find_libs() -> Dict[Tuple[str, str], List[str]]:
    if False:
        print('Hello World!')
    'Find all system-wide .so libraries.'
    all_libs: Dict[Tuple[str, str], List[str]] = {}
    if pathlib.Path('/sbin/ldconfig').exists():
        ldconfig_bin = '/sbin/ldconfig'
    else:
        ldconfig_bin = 'ldconfig'
    ldconfig_proc = subprocess.run([ldconfig_bin, '-p'], check=True, stdout=subprocess.PIPE, encoding=sys.getfilesystemencoding())
    pattern = re.compile('(?P<name>\\S+) \\((?P<abi_type>[^)]+)\\) => (?P<path>.*)')
    for line in ldconfig_proc.stdout.splitlines():
        match = pattern.fullmatch(line.strip())
        if match is None:
            if 'libs found in cache' not in line and 'Cache generated by:' not in line:
                utils.print_col(f'Failed to match ldconfig output: {line}', 'yellow')
            continue
        key = (match.group('name'), match.group('abi_type'))
        path = match.group('path')
        libs = all_libs.setdefault(key, [])
        libs.append(path)
    return all_libs

def run_qt_smoke_test_single(venv_dir: pathlib.Path, *, debug: bool, pyqt_version: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Make sure the Qt installation works.'
    utils.print_title('Running Qt smoke test')
    code = ['import sys', 'from qutebrowser.qt.widgets import QApplication', 'from qutebrowser.qt.core import qVersion, QT_VERSION_STR, PYQT_VERSION_STR', 'print(f"Python: {sys.version}")', 'print(f"qVersion: {qVersion()}")', 'print(f"QT_VERSION_STR: {QT_VERSION_STR}")', 'print(f"PYQT_VERSION_STR: {PYQT_VERSION_STR}")', 'QApplication([])', 'print("Qt seems to work properly!")', 'print()']
    env = {'QUTE_QT_WRAPPER': 'PyQt6' if _is_qt6_version(pyqt_version) else 'PyQt5'}
    if debug:
        env['QT_DEBUG_PLUGINS'] = '1'
    try:
        run_venv(venv_dir, 'python', '-c', '; '.join(code), env=env, capture_error=True)
    except Error as e:
        proc_e = e.__cause__
        assert isinstance(proc_e, subprocess.CalledProcessError), proc_e
        print(proc_e.stderr)
        msg = f'Smoke test failed with status {proc_e.returncode}.'
        if debug:
            msg += ' You might find additional information in the debug output above.'
        raise Error(msg)

def run_qt_smoke_test(venv_dir: pathlib.Path, *, pyqt_version: str) -> None:
    if False:
        print('Hello World!')
    'Make sure the Qt installation works.'
    no_debug = pyqt_version == '6.3' and sys.platform == 'darwin'
    if no_debug:
        try:
            run_qt_smoke_test_single(venv_dir, debug=False, pyqt_version=pyqt_version)
        except Error as e:
            print(e)
            print('Rerunning with debug output...')
            print('NOTE: This will likely segfault due to a Qt bug:')
            print('https://bugreports.qt.io/browse/QTBUG-104415')
            run_qt_smoke_test_single(venv_dir, debug=True, pyqt_version=pyqt_version)
    else:
        run_qt_smoke_test_single(venv_dir, debug=True, pyqt_version=pyqt_version)

def install_requirements(venv_dir: pathlib.Path) -> None:
    if False:
        return 10
    "Install qutebrowser's requirement.txt."
    utils.print_title('Installing other qutebrowser dependencies')
    requirements = REPO_ROOT / 'requirements.txt'
    pip_install(venv_dir, '-r', str(requirements))

def install_dev_requirements(venv_dir: pathlib.Path) -> None:
    if False:
        while True:
            i = 10
    'Install development dependencies.'
    utils.print_title('Installing dev dependencies')
    pip_install(venv_dir, '-r', str(requirements_file('dev')), '-r', str(requirements_file('check-manifest')), '-r', str(requirements_file('flake8')), '-r', str(requirements_file('mypy')), '-r', str(requirements_file('pyroma')), '-r', str(requirements_file('vulture')), '-r', str(requirements_file('yamllint')), '-r', str(requirements_file('tests')))

def install_qutebrowser(venv_dir: pathlib.Path) -> None:
    if False:
        print('Hello World!')
    'Install qutebrowser itself as an editable install.'
    utils.print_title('Installing qutebrowser')
    pip_install(venv_dir, '-e', str(REPO_ROOT))

def regenerate_docs(venv_dir: pathlib.Path):
    if False:
        print('Hello World!')
    'Regenerate docs using asciidoc.'
    utils.print_title('Generating documentation')
    pip_install(venv_dir, '-r', str(requirements_file('docs')))
    script_path = pathlib.Path(__file__).parent / 'asciidoc2html.py'
    print_command('python3 scripts/asciidoc2html.py', venv=True)
    run_venv(venv_dir, 'python', str(script_path))

def update_repo():
    if False:
        while True:
            i = 10
    'Update the git repository via git pull.'
    print_command('git pull', venv=False)
    try:
        subprocess.run(['git', 'pull'], check=True)
    except subprocess.CalledProcessError as e:
        raise Error('git pull failed, exiting') from e

def install_pyqt(venv_dir, args):
    if False:
        while True:
            i = 10
    'Install PyQt in the virtualenv.'
    if args.pyqt_type == 'binary':
        install_pyqt_binary(venv_dir, args.pyqt_version)
        if args.pyqt_snapshot:
            install_pyqt_snapshot(venv_dir, args.pyqt_snapshot.split(','))
    elif args.pyqt_type == 'source':
        install_pyqt_source(venv_dir, args.pyqt_version)
    elif args.pyqt_type == 'link':
        install_pyqt_link(venv_dir, args.pyqt_version)
    elif args.pyqt_type == 'wheels':
        wheels_dir = pathlib.Path(args.pyqt_wheels_dir)
        if not wheels_dir.is_dir():
            raise Error(f"Wheels directory {wheels_dir} doesn't exist or is not a directory")
        install_pyqt_wheels(venv_dir, wheels_dir)
    elif args.pyqt_type == 'skip':
        pass
    else:
        raise AssertionError

def run(args) -> None:
    if False:
        while True:
            i = 10
    'Install qutebrowser in a virtualenv..'
    venv_dir = pathlib.Path(args.venv_dir)
    utils.change_cwd()
    if args.pyqt_version != 'auto' and args.pyqt_type == 'skip':
        raise Error('Cannot use --pyqt-version with --pyqt-type skip')
    if args.pyqt_type == 'link' and args.pyqt_version not in ['auto', '5', '6']:
        raise Error('Invalid --pyqt-version {args.pyqt_version}, only 5 or 6 permitted with --pyqt-type=link')
    if args.pyqt_wheels_dir != 'wheels' and args.pyqt_type != 'wheels':
        raise Error('The --pyqt-wheels-dir option is only available when installing PyQt from wheels')
    if args.pyqt_snapshot and args.pyqt_type != 'binary':
        raise Error('The --pyqt-snapshot option is only available when installing PyQt from binaries')
    if args.update:
        utils.print_title('Updating repository')
        update_repo()
    if not args.keep:
        utils.print_title('Creating virtual environment')
        delete_old_venv(venv_dir)
        create_venv(venv_dir, use_virtualenv=args.virtualenv)
    upgrade_seed_pkgs(venv_dir)
    install_pyqt(venv_dir, args)
    apply_xcb_util_workaround(venv_dir, args.pyqt_type, args.pyqt_version)
    install_requirements(venv_dir)
    install_qutebrowser(venv_dir)
    if args.dev:
        install_dev_requirements(venv_dir)
    if args.pyqt_type != 'skip' and (not args.skip_smoke_test):
        run_qt_smoke_test(venv_dir, pyqt_version=args.pyqt_version)
    if not args.skip_docs:
        regenerate_docs(venv_dir)

def main():
    if False:
        i = 10
        return i + 15
    args = parse_args()
    try:
        run(args)
    except Error as e:
        utils.print_error(str(e))
        sys.exit(e.code)
if __name__ == '__main__':
    main()