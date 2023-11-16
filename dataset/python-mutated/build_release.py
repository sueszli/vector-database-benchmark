"""Build a new release."""
import os
import sys
import time
import shutil
import pathlib
import subprocess
import argparse
import tarfile
import tempfile
import collections
import dataclasses
import re
from typing import Iterable, List, Optional
try:
    import winreg
except ImportError:
    pass
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import qutebrowser
from scripts import utils
from scripts.dev import update_3rdparty, misc_checks
IS_MACOS = sys.platform == 'darwin'
IS_WINDOWS = os.name == 'nt'

@dataclasses.dataclass
class Artifact:
    """A single file being uploaded to GitHub."""
    path: pathlib.Path
    mimetype: str
    description: str

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'{self.path} ({self.mimetype}): {self.description}'

def call_script(name: str, *args: str, python: str=sys.executable) -> None:
    if False:
        return 10
    'Call a given shell script.\n\n    Args:\n        name: The script to call.\n        *args: The arguments to pass.\n        python: The python interpreter to use.\n    '
    subprocess.run([python, REPO_ROOT / 'scripts' / name, *args], check=True)

def call_tox(toxenv: str, *args: str, python: pathlib.Path=pathlib.Path(sys.executable), debug: bool=False) -> None:
    if False:
        while True:
            i = 10
    'Call tox.\n\n    Args:\n        toxenv: Which tox environment to use\n        *args: The arguments to pass.\n        python: The python interpreter to use.\n        debug: Turn on pyinstaller debugging\n    '
    env = os.environ.copy()
    env['PYTHON'] = str(python)
    env['PATH'] = os.environ['PATH'] + os.pathsep + str(python.parent)
    if debug:
        env['PYINSTALLER_DEBUG'] = '1'
    subprocess.run([sys.executable, '-m', 'tox', '-vv', '-e', toxenv, *args], env=env, check=True)

def run_asciidoc2html() -> None:
    if False:
        print('Hello World!')
    'Run the asciidoc2html script.'
    utils.print_title('Running asciidoc2html.py')
    call_script('asciidoc2html.py')

def _maybe_remove(path: pathlib.Path) -> None:
    if False:
        return 10
    'Remove a path if it exists.'
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass

def _filter_whitelisted(output: bytes, patterns: Iterable[str]) -> Iterable[str]:
    if False:
        for i in range(10):
            print('nop')
    'Get all lines not matching any of the given regex patterns.'
    for line in output.decode('utf-8').splitlines():
        if not any((re.fullmatch(pattern, line) for pattern in patterns)):
            yield line

def _smoke_test_run(executable: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    if False:
        while True:
            i = 10
    'Get a subprocess to run a smoke test.'
    argv = [executable, '--no-err-windows', '--nowindow', '--temp-basedir', *args, 'about:blank', ':cmd-later 500 quit']
    return subprocess.run(argv, check=True, capture_output=True)

def smoke_test(executable: pathlib.Path, debug: bool, qt5: bool) -> None:
    if False:
        return 10
    'Try starting the given qutebrowser executable.'
    stdout_whitelist = []
    stderr_whitelist = ['\\[.*\\] PyInstaller Bootloader .*', '\\[.*\\] LOADER: .*']
    if IS_MACOS:
        stderr_whitelist.extend(['objc\\[.*\\]: .* One of the two will be used\\. Which one is undefined\\.', 'QCoreApplication::applicationDirPath: Please instantiate the QApplication object first', '\\[.*:ERROR:mach_port_broker.mm\\(48\\)\\] bootstrap_look_up org\\.chromium\\.Chromium\\.rohitfork\\.1: Permission denied \\(1100\\)', '\\[.*:ERROR:mach_port_broker.mm\\(43\\)\\] bootstrap_look_up: Unknown service name \\(1102\\)', '[0-9:]* WARNING: The available OpenGL surface format was either not version 3\\.2 or higher or not a Core Profile\\.', 'Chromium on macOS will fall back to software rendering in this case\\.', 'Hardware acceleration and features such as WebGL will not be available\\.', 'Unable to create basic Accelerated OpenGL renderer\\.', 'Core Image is now using the software OpenGL renderer\\. This will be slow\\.', '[0-9:]* ERROR: Load error: ERR_FILE_NOT_FOUND', '[0-9:]* WARNING: Failed to load libssl/libcrypto\\.', '\\[.*:ERROR:command_buffer_proxy_impl.cc\\([0-9]*\\)\\] ContextResult::kTransientFailure: Failed to send .*CreateCommandBuffer\\.'])
        if not qt5:
            stderr_whitelist.extend(['[0-9:]* WARNING: Incompatible version of OpenSSL', '[0-9:]* WARNING: Qt WebEngine resources not found at .*', '[0-9:]* WARNING: Installed Qt WebEngine locales directory not found at location /qtwebengine_locales\\. Trying application directory\\.\\.\\.'])
    elif IS_WINDOWS:
        stderr_whitelist.extend(['\\[.*:ERROR:dxva_video_decode_accelerator_win.cc\\(\\d+\\)\\] DXVAVDA fatal error: could not LoadLibrary: .*: The specified module could not be found. \\(0x7E\\)'])
    proc = _smoke_test_run(executable)
    if debug:
        print('Skipping output check for debug build')
        return
    stdout = '\n'.join(_filter_whitelisted(proc.stdout, stdout_whitelist))
    stderr = '\n'.join(_filter_whitelisted(proc.stderr, stderr_whitelist))
    if stdout or stderr:
        print('Unexpected output, running with --debug')
        proc = _smoke_test_run(executable, '--debug')
        debug_stdout = proc.stdout.decode('utf-8')
        debug_stderr = proc.stderr.decode('utf-8')
        lines = ['Unexpected output!', '']
        if stdout:
            lines += ['stdout', '------', '', stdout, '']
        if stderr:
            lines += ['stderr', '------', '', stderr, '']
        if debug_stdout:
            lines += ['debug rerun stdout', '------------------', '', debug_stdout, '']
        if debug_stderr:
            lines += ['debug rerun stderr', '------------------', '', debug_stderr, '']
        raise Exception('\n'.join(lines))

def verify_windows_exe(exe_path: pathlib.Path) -> None:
    if False:
        print('Hello World!')
    'Make sure the Windows .exe has a correct checksum.'
    import pefile
    pe = pefile.PE(exe_path)
    assert pe.verify_checksum()

def verify_mac_app() -> None:
    if False:
        while True:
            i = 10
    'Re-sign and verify the Mac .app.'
    app_path = pathlib.Path('dist') / 'qutebrowser.app'
    subprocess.run(['codesign', '--verify', '--strict', '--deep', '--verbose', app_path], check=True)

def _mac_bin_path(base: pathlib.Path) -> pathlib.Path:
    if False:
        print('Hello World!')
    'Get the macOS qutebrowser binary path.'
    return pathlib.Path(base, 'qutebrowser.app', 'Contents', 'MacOS', 'qutebrowser')

def build_mac(*, gh_token: Optional[str], qt5: bool, skip_packaging: bool, debug: bool) -> List[Artifact]:
    if False:
        return 10
    'Build macOS .dmg/.app.'
    utils.print_title('Cleaning up...')
    for f in ['wc.dmg', 'template.dmg']:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
    for d in ['dist', 'build']:
        shutil.rmtree(d, ignore_errors=True)
    utils.print_title('Updating 3rdparty content')
    update_3rdparty.run(ace=False, pdfjs=True, legacy_pdfjs=qt5, fancy_dmg=False, gh_token=gh_token)
    utils.print_title('Building .app via pyinstaller')
    call_tox(f"pyinstaller{('-qt5' if qt5 else '')}", '-r', debug=debug)
    utils.print_title('Verifying .app')
    verify_mac_app()
    dist_path = pathlib.Path('dist')
    utils.print_title('Running pre-dmg smoke test')
    smoke_test(_mac_bin_path(dist_path), debug=debug, qt5=qt5)
    if skip_packaging:
        return []
    utils.print_title('Building .dmg')
    dmg_makefile_path = REPO_ROOT / 'scripts' / 'dev' / 'Makefile-dmg'
    subprocess.run(['make', '-f', dmg_makefile_path], check=True)
    suffix = '-debug' if debug else ''
    suffix += '-qt5' if qt5 else ''
    dmg_path = dist_path / f'qutebrowser-{qutebrowser.__version__}{suffix}.dmg'
    pathlib.Path('qutebrowser.dmg').rename(dmg_path)
    utils.print_title('Running smoke test')
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            subprocess.run(['hdiutil', 'attach', dmg_path, '-mountpoint', tmp_path], check=True)
            try:
                smoke_test(_mac_bin_path(tmp_path), debug=debug, qt5=qt5)
            finally:
                print('Waiting 10s for dmg to be detachable...')
                time.sleep(10)
                subprocess.run(['hdiutil', 'detach', tmp_path], check=False)
    except PermissionError as e:
        print(f'Failed to remove tempdir: {e}')
    return [Artifact(path=dmg_path, mimetype='application/x-apple-diskimage', description='macOS .dmg')]

def _get_windows_python_path() -> pathlib.Path:
    if False:
        print('Hello World!')
    'Get the path to Python.exe on Windows.'
    parts = (str(sys.version_info.major), str(sys.version_info.minor))
    ver = ''.join(parts)
    dot_ver = '.'.join(parts)
    path = f'SOFTWARE\\Python\\PythonCore\\{dot_ver}\\InstallPath'
    fallback = pathlib.Path('C:', f'Python{ver}', 'python.exe')
    try:
        key = winreg.OpenKeyEx(winreg.HKEY_LOCAL_MACHINE, path)
        return pathlib.Path(winreg.QueryValueEx(key, 'ExecutablePath')[0])
    except FileNotFoundError:
        return fallback

def _build_windows_single(*, qt5: bool, skip_packaging: bool, debug: bool) -> List[Artifact]:
    if False:
        return 10
    'Build on Windows for a single build type.'
    utils.print_title('Running pyinstaller')
    dist_path = pathlib.Path('dist')
    out_path = dist_path / f'qutebrowser-{qutebrowser.__version__}'
    _maybe_remove(out_path)
    python = _get_windows_python_path()
    suffix = '-qt5' if qt5 else ''
    call_tox(f'pyinstaller{suffix}', '-r', python=python, debug=debug)
    out_pyinstaller = dist_path / 'qutebrowser'
    shutil.move(out_pyinstaller, out_path)
    exe_path = out_path / 'qutebrowser.exe'
    utils.print_title('Verifying exe')
    verify_windows_exe(exe_path)
    utils.print_title('Running smoke test')
    smoke_test(exe_path, debug=debug, qt5=qt5)
    if skip_packaging:
        return []
    utils.print_title('Packaging')
    return _package_windows_single(out_path=out_path, debug=debug, qt5=qt5)

def build_windows(*, gh_token: str, skip_packaging: bool, qt5: bool, debug: bool) -> List[Artifact]:
    if False:
        for i in range(10):
            print('nop')
    'Build windows executables/setups.'
    utils.print_title('Updating 3rdparty content')
    update_3rdparty.run(nsis=True, ace=False, pdfjs=True, legacy_pdfjs=qt5, fancy_dmg=False, gh_token=gh_token)
    utils.print_title('Building Windows binaries')
    from scripts.dev import gen_versioninfo
    utils.print_title('Updating VersionInfo file')
    gen_versioninfo.main()
    artifacts = _build_windows_single(skip_packaging=skip_packaging, debug=debug, qt5=qt5)
    return artifacts

def _package_windows_single(*, out_path: pathlib.Path, debug: bool, qt5: bool) -> List[Artifact]:
    if False:
        for i in range(10):
            print('nop')
    'Build the given installer/zip for windows.'
    artifacts = []
    dist_path = pathlib.Path('dist')
    utils.print_subtitle('Building installer...')
    subprocess.run(['makensis.exe', f'/DVERSION={qutebrowser.__version__}', f'/DQT5={qt5}', 'misc/nsis/qutebrowser.nsi'], check=True)
    name_parts = ['qutebrowser', str(qutebrowser.__version__)]
    if debug:
        name_parts.append('debug')
    if qt5:
        name_parts.append('qt5')
    name_parts.append('amd64')
    name = '-'.join(name_parts) + '.exe'
    artifacts.append(Artifact(path=dist_path / name, mimetype='application/vnd.microsoft.portable-executable', description='Windows installer'))
    utils.print_subtitle('Zipping standalone...')
    zip_name_parts = ['qutebrowser', str(qutebrowser.__version__), 'windows', 'standalone']
    if debug:
        zip_name_parts.append('debug')
    if qt5:
        zip_name_parts.append('qt5')
    zip_name = '-'.join(zip_name_parts) + '.zip'
    zip_path = dist_path / zip_name
    shutil.make_archive(str(zip_path.with_suffix('')), 'zip', 'dist', out_path.name)
    artifacts.append(Artifact(path=zip_path, mimetype='application/zip', description='Windows standalone'))
    return artifacts

def build_sdist() -> List[Artifact]:
    if False:
        for i in range(10):
            print('nop')
    'Build an sdist and list the contents.'
    utils.print_title('Building sdist')
    dist_path = pathlib.Path('dist')
    _maybe_remove(dist_path)
    subprocess.run([sys.executable, '-m', 'build'], check=True)
    dist_files = list(dist_path.glob('*.tar.gz'))
    filename = f'qutebrowser-{qutebrowser.__version__}.tar.gz'
    assert dist_files == [dist_path / filename], dist_files
    dist_file = dist_files[0]
    subprocess.run(['gpg', '--detach-sign', '-a', str(dist_file)], check=True)
    by_ext = collections.defaultdict(list)
    with tarfile.open(dist_file) as tar:
        for tarinfo in tar.getmembers():
            if not tarinfo.isfile():
                continue
            path = pathlib.Path(*pathlib.Path(tarinfo.name).parts[1:])
            by_ext[path.suffix].append(path)
    assert '.pyc' not in by_ext
    utils.print_title('sdist contents')
    for (ext, paths) in sorted(by_ext.items()):
        utils.print_subtitle(ext)
        print('\n'.join((str(p) for p in paths)))
    artifacts = [Artifact(path=dist_file, mimetype='application/gzip', description='Source release'), Artifact(path=dist_file.with_suffix(dist_file.suffix + '.asc'), mimetype='application/pgp-signature', description='Source release - PGP signature')]
    return artifacts

def test_makefile() -> None:
    if False:
        print('Hello World!')
    'Make sure the Makefile works correctly.'
    utils.print_title('Testing makefile')
    a2x_path = pathlib.Path(sys.executable).parent / 'a2x'
    assert a2x_path.exists(), a2x_path
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(['make', '-f', 'misc/Makefile', f'DESTDIR={tmpdir}', f'A2X={a2x_path}', 'install'], check=True)

def read_github_token(arg_token: Optional[str], *, optional: bool=False) -> Optional[str]:
    if False:
        i = 10
        return i + 15
    'Read the GitHub API token from disk.'
    if arg_token is not None:
        return arg_token
    if 'GITHUB_TOKEN' in os.environ:
        return os.environ['GITHUB_TOKEN']
    token_path = pathlib.Path.home() / '.gh_token'
    if not token_path.exists():
        if optional:
            return None
        else:
            raise Exception('GitHub token needed, but ~/.gh_token not found, and --gh-token not given.')
    token = token_path.read_text(encoding='ascii').strip()
    return token

def github_upload(artifacts: List[Artifact], tag: str, gh_token: str, experimental: bool) -> None:
    if False:
        return 10
    'Upload the given artifacts to GitHub.\n\n    Args:\n        artifacts: A list of Artifacts to upload.\n        tag: The name of the release tag\n        gh_token: The GitHub token to use\n        experimental: Upload to the experiments repo\n    '
    import github3
    import github3.exceptions
    utils.print_title('Uploading to github...')
    gh = github3.login(token=gh_token)
    if experimental:
        repo = gh.repository('qutebrowser', 'experiments')
    else:
        repo = gh.repository('qutebrowser', 'qutebrowser')
    release = None
    for release in repo.releases():
        if release.tag_name == tag:
            break
    else:
        releases = ', '.join((r.tag_name for r in repo.releases()))
        raise Exception(f'No release found for {tag!r} in {repo.full_name}, found: {releases}')
    for artifact in artifacts:
        while True:
            print(f'Uploading {artifact.path}')
            assets = [asset for asset in release.assets() if asset.name == artifact.path.name]
            if assets:
                print(f'Assets already exist: {assets}')
                if utils.ON_CI:
                    sys.exit(1)
                print('Press enter to continue anyways or Ctrl-C to abort.')
                input()
            try:
                with artifact.path.open('rb') as f:
                    release.upload_asset(artifact.mimetype, artifact.path.name, f, artifact.description)
            except github3.exceptions.ConnectionError as e:
                utils.print_error(f'Failed to upload: {e}')
                if utils.ON_CI:
                    print('Retrying in 30s...')
                    time.sleep(30)
                else:
                    print('Press Enter to retry...', file=sys.stderr)
                    input()
                print('Retrying!')
                assets = [asset for asset in release.assets() if asset.name == artifact.path.name]
                if assets:
                    stray_asset = assets[0]
                    print(f'Deleting stray asset {stray_asset.name}')
                    stray_asset.delete()
            else:
                break

def pypi_upload(artifacts: List[Artifact], experimental: bool) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Upload the given artifacts to PyPI using twine.'
    artifacts = [a for a in artifacts if a.mimetype != 'application/pgp-signature']
    utils.print_title('Uploading to PyPI...')
    if experimental:
        run_twine('upload', artifacts, '-r', 'testpypi')
    else:
        run_twine('upload', artifacts)

def twine_check(artifacts: List[Artifact]) -> None:
    if False:
        print('Hello World!')
    "Check packages using 'twine check'."
    utils.print_title('Running twine check...')
    run_twine('check', artifacts, '--strict')

def run_twine(command: str, artifacts: List[Artifact], *args: str) -> None:
    if False:
        while True:
            i = 10
    paths = [a.path for a in artifacts]
    subprocess.run([sys.executable, '-m', 'twine', command, *args, *paths], check=True)

def main() -> None:
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-docs', action='store_true', help="Don't generate docs")
    parser.add_argument('--gh-token', help='GitHub token to use.', nargs='?')
    parser.add_argument('--upload', action='store_true', required=False, help='Toggle to upload the release to GitHub.')
    parser.add_argument('--no-confirm', action='store_true', required=False, help='Skip confirmation before uploading.')
    parser.add_argument('--skip-packaging', action='store_true', required=False, help='Skip Windows installer/zip generation or macOS DMG.')
    parser.add_argument('--debug', action='store_true', required=False, help='Build a debug build.')
    parser.add_argument('--qt5', action='store_true', required=False, help='Build against PyQt5')
    parser.add_argument('--experimental', action='store_true', required=False, help='Upload to experiments repo and test PyPI')
    args = parser.parse_args()
    utils.change_cwd()
    upload_to_pypi = False
    if args.upload:
        import github3
        gh_token = read_github_token(args.gh_token)
    else:
        gh_token = read_github_token(args.gh_token, optional=True)
        assert not args.experimental
    if not misc_checks.check_git():
        utils.print_error('Refusing to do a release with a dirty git tree')
        sys.exit(1)
    if args.skip_docs:
        pathlib.Path('qutebrowser', 'html', 'doc').mkdir(parents=True, exist_ok=True)
    else:
        run_asciidoc2html()
    if IS_WINDOWS:
        artifacts = build_windows(gh_token=gh_token, skip_packaging=args.skip_packaging, qt5=args.qt5, debug=args.debug)
    elif IS_MACOS:
        artifacts = build_mac(gh_token=gh_token, skip_packaging=args.skip_packaging, qt5=args.qt5, debug=args.debug)
    else:
        test_makefile()
        artifacts = build_sdist()
        twine_check(artifacts)
        upload_to_pypi = True
    if args.upload:
        version_tag = f'v{qutebrowser.__version__}'
        if not args.no_confirm and (not utils.ON_CI):
            utils.print_title(f'Press enter to release {version_tag}...')
            input()
        assert gh_token is not None
        github_upload(artifacts, version_tag, gh_token=gh_token, experimental=args.experimental)
        if upload_to_pypi:
            pypi_upload(artifacts, experimental=args.experimental)
    else:
        print()
        utils.print_title('Artifacts')
        for artifact in artifacts:
            print(artifact)
if __name__ == '__main__':
    main()