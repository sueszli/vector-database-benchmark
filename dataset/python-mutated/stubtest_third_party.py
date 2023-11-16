"""Test typeshed's third party stubs using stubtest"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from textwrap import dedent
from typing import NoReturn
from parse_metadata import NoSuchStubError, get_recursive_requirements, read_metadata
from utils import PYTHON_VERSION, colored, get_mypy_req, make_venv, print_error, print_success_msg

def run_stubtest(dist: Path, *, parser: argparse.ArgumentParser, verbose: bool=False, specified_platforms_only: bool=False) -> bool:
    if False:
        while True:
            i = 10
    dist_name = dist.name
    try:
        metadata = read_metadata(dist_name)
    except NoSuchStubError as e:
        parser.error(str(e))
    print(f'{dist_name}... ', end='')
    stubtest_settings = metadata.stubtest_settings
    if stubtest_settings.skipped:
        print(colored('skipping', 'yellow'))
        return True
    if sys.platform not in stubtest_settings.platforms:
        if specified_platforms_only:
            print(colored('skipping (platform not specified in METADATA.toml)', 'yellow'))
            return True
        print(colored(f"Note: {dist_name} is not currently tested on {sys.platform} in typeshed's CI.", 'yellow'))
    if not metadata.requires_python.contains(PYTHON_VERSION):
        print(colored(f'skipping (requires Python {metadata.requires_python})', 'yellow'))
        return True
    with tempfile.TemporaryDirectory() as tmp:
        venv_dir = Path(tmp)
        try:
            (pip_exe, python_exe) = make_venv(venv_dir)
        except Exception:
            print_error('fail')
            raise
        dist_extras = ', '.join(stubtest_settings.extras)
        dist_req = f'{dist_name}[{dist_extras}]=={metadata.version}'
        if stubtest_settings.stubtest_requirements:
            pip_cmd = [pip_exe, 'install'] + stubtest_settings.stubtest_requirements
            try:
                subprocess.run(pip_cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print_command_failure('Failed to install requirements', e)
                return False
        requirements = get_recursive_requirements(dist_name)
        dists_to_install = [dist_req, get_mypy_req()]
        dists_to_install.extend(requirements.external_pkgs)
        pip_cmd = [pip_exe, 'install'] + dists_to_install
        try:
            subprocess.run(pip_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print_command_failure('Failed to install', e)
            return False
        ignore_missing_stub = ['--ignore-missing-stub'] if stubtest_settings.ignore_missing_stub else []
        packages_to_check = [d.name for d in dist.iterdir() if d.is_dir() and d.name.isidentifier()]
        modules_to_check = [d.stem for d in dist.iterdir() if d.is_file() and d.suffix == '.pyi']
        stubtest_cmd = [python_exe, '-m', 'mypy.stubtest', '--custom-typeshed-dir', str(dist.parent.parent), *ignore_missing_stub, *packages_to_check, *modules_to_check]
        stubs_dir = dist.parent
        mypypath_items = [str(dist)] + [str(stubs_dir / pkg) for pkg in requirements.typeshed_pkgs]
        mypypath = os.pathsep.join(mypypath_items)
        stubtest_env = os.environ | {'MYPYPATH': mypypath, 'MYPY_FORCE_COLOR': '1'}
        allowlist_path = dist / '@tests/stubtest_allowlist.txt'
        if allowlist_path.exists():
            stubtest_cmd.extend(['--allowlist', str(allowlist_path)])
        platform_allowlist = dist / f'@tests/stubtest_allowlist_{sys.platform}.txt'
        if platform_allowlist.exists():
            stubtest_cmd.extend(['--allowlist', str(platform_allowlist)])
        if dist_name == 'uWSGI':
            if not setup_uwsgi_stubtest_command(dist, venv_dir, stubtest_cmd):
                return False
        try:
            subprocess.run(stubtest_cmd, env=stubtest_env, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print_error('fail')
            print_commands(dist, pip_cmd, stubtest_cmd, mypypath)
            print_command_output(e)
            print('Python version: ', file=sys.stderr)
            ret = subprocess.run([sys.executable, '-VV'], capture_output=True)
            print_command_output(ret)
            print('Ran with the following environment:', file=sys.stderr)
            ret = subprocess.run([pip_exe, 'freeze', '--all'], capture_output=True)
            print_command_output(ret)
            if allowlist_path.exists():
                print(f'To fix "unused allowlist" errors, remove the corresponding entries from {allowlist_path}', file=sys.stderr)
                print(file=sys.stderr)
            else:
                print(f'Re-running stubtest with --generate-allowlist.\nAdd the following to {allowlist_path}:', file=sys.stderr)
                ret = subprocess.run(stubtest_cmd + ['--generate-allowlist'], env=stubtest_env, capture_output=True)
                print_command_output(ret)
            return False
        else:
            print_success_msg()
    if verbose:
        print_commands(dist, pip_cmd, stubtest_cmd, mypypath)
    return True

def setup_uwsgi_stubtest_command(dist: Path, venv_dir: Path, stubtest_cmd: list[str]) -> bool:
    if False:
        print('Hello World!')
    'Perform some black magic in order to run stubtest inside uWSGI.\n\n    We have to write the exit code from stubtest to a surrogate file\n    because uwsgi --pyrun does not exit with the exitcode from the\n    python script. We have a second wrapper script that passed the\n    arguments along to the uWSGI script and retrieves the exit code\n    from the file, so it behaves like running stubtest normally would.\n\n    Both generated wrapper scripts are created inside `venv_dir`,\n    which itself is a subdirectory inside a temporary directory,\n    so both scripts will be cleaned up after this function\n    has been executed.\n    '
    uwsgi_ini = dist / '@tests/uwsgi.ini'
    if sys.platform == 'win32':
        print_error('uWSGI is not supported on Windows')
        return False
    uwsgi_script = venv_dir / 'uwsgi_stubtest.py'
    wrapper_script = venv_dir / 'uwsgi_wrapper.py'
    exit_code_surrogate = venv_dir / 'exit_code'
    uwsgi_script_contents = dedent(f'\n        import json\n        import os\n        import sys\n        from mypy.stubtest import main\n\n        sys.argv = json.loads(os.environ.get("STUBTEST_ARGS"))\n        exit_code = main()\n        with open("{exit_code_surrogate}", mode="w") as fp:\n            fp.write(str(exit_code))\n        sys.exit(exit_code)\n        ')
    uwsgi_script.write_text(uwsgi_script_contents)
    uwsgi_exe = venv_dir / 'bin' / 'uwsgi'
    wrapper_script_contents = dedent(f'\n        import json\n        import os\n        import subprocess\n        import sys\n\n        stubtest_env = os.environ | {{"STUBTEST_ARGS": json.dumps(sys.argv)}}\n        uwsgi_cmd = [\n            "{uwsgi_exe}",\n            "--ini",\n            "{uwsgi_ini}",\n            "--spooler",\n            "{venv_dir}",\n            "--pyrun",\n            "{uwsgi_script}",\n        ]\n        subprocess.run(uwsgi_cmd, env=stubtest_env)\n        with open("{exit_code_surrogate}", mode="r") as fp:\n            sys.exit(int(fp.read()))\n        ')
    wrapper_script.write_text(wrapper_script_contents)
    assert stubtest_cmd[1:3] == ['-m', 'mypy.stubtest']
    stubtest_cmd[1:3] = [str(wrapper_script)]
    return True

def print_commands(dist: Path, pip_cmd: list[str], stubtest_cmd: list[str], mypypath: str) -> None:
    if False:
        i = 10
        return i + 15
    print(file=sys.stderr)
    print(' '.join(pip_cmd), file=sys.stderr)
    print(f'MYPYPATH={mypypath}', ' '.join(stubtest_cmd), file=sys.stderr)
    print(file=sys.stderr)

def print_command_failure(message: str, e: subprocess.CalledProcessError) -> None:
    if False:
        while True:
            i = 10
    print_error('fail')
    print(file=sys.stderr)
    print(message, file=sys.stderr)
    print_command_output(e)

def print_command_output(e: subprocess.CalledProcessError | subprocess.CompletedProcess[bytes]) -> None:
    if False:
        for i in range(10):
            print('nop')
    print(e.stdout.decode(), end='', file=sys.stderr)
    print(e.stderr.decode(), end='', file=sys.stderr)
    print(file=sys.stderr)

def main() -> NoReturn:
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--specified-platforms-only', action='store_true', help='skip the test if the current platform is not specified in METADATA.toml/tool.stubtest.platforms')
    parser.add_argument('dists', metavar='DISTRIBUTION', type=str, nargs=argparse.ZERO_OR_MORE)
    args = parser.parse_args()
    typeshed_dir = Path('.').resolve()
    if len(args.dists) == 0:
        dists = sorted((typeshed_dir / 'stubs').iterdir())
    else:
        dists = [typeshed_dir / 'stubs' / d for d in args.dists]
    result = 0
    for (i, dist) in enumerate(dists):
        if i % args.num_shards != args.shard_index:
            continue
        if not run_stubtest(dist, parser=parser, verbose=args.verbose, specified_platforms_only=args.specified_platforms_only):
            result = 1
    sys.exit(result)
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass