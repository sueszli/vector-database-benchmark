"""Verify the contents of the built sdist and wheel."""
from __future__ import annotations

import contextlib
import fnmatch
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import typing as t
import zipfile

from ansible.release import __version__


def collect_sdist_files(complete_file_list: list[str]) -> list[str]:
    """Return a list of files which should be present in the sdist."""
    ignore_patterns = (
        '.azure-pipelines/*',
        '.cherry_picker.toml',
        '.git*',
        '.mailmap',
        'changelogs/README.md',
        'changelogs/config.yaml',
        'changelogs/fragments/*',
        'hacking/*',
    )

    sdist_files = [path for path in complete_file_list if not any(fnmatch.fnmatch(path, ignore) for ignore in ignore_patterns)]

    egg_info = (
        'PKG-INFO',
        'SOURCES.txt',
        'dependency_links.txt',
        'entry_points.txt',
        'not-zip-safe',
        'requires.txt',
        'top_level.txt',
    )

    sdist_files.append('PKG-INFO')
    sdist_files.extend(f'lib/ansible_core.egg-info/{name}' for name in egg_info)

    return sdist_files


def collect_wheel_files(complete_file_list: list[str]) -> list[str]:
    """Return a list of files which should be present in the wheel."""
    wheel_files = []

    for path in complete_file_list:
        if path.startswith('lib/ansible/'):
            prefix = 'lib'
        elif path.startswith('test/lib/ansible_test/'):
            prefix = 'test/lib'
        else:
            continue

        wheel_files.append(os.path.relpath(path, prefix))

    dist_info = (
        'COPYING',
        'METADATA',
        'RECORD',
        'WHEEL',
        'entry_points.txt',
        'top_level.txt',
    )

    wheel_files.append(f'ansible_core-{__version__}.data/scripts/ansible-test')
    wheel_files.extend(f'ansible_core-{__version__}.dist-info/{name}' for name in dist_info)

    return wheel_files


@contextlib.contextmanager
def clean_repository(complete_file_list: list[str]) -> t.Generator[str, None, None]:
    """Copy the files to a temporary directory and yield the path."""
    directories = sorted(set(os.path.dirname(path) for path in complete_file_list))
    directories.remove('')

    with tempfile.TemporaryDirectory() as temp_dir:
        for directory in directories:
            os.makedirs(os.path.join(temp_dir, directory))

        for path in complete_file_list:
            shutil.copy2(path, os.path.join(temp_dir, path), follow_symlinks=False)

        yield temp_dir


def build(source_dir: str, tmp_dir: str) -> tuple[pathlib.Path, pathlib.Path]:
    """Create a sdist and wheel."""
    create = subprocess.run(
        [sys.executable, '-m', 'build', '--no-isolation', '--outdir', tmp_dir],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        cwd=source_dir,
    )

    if create.returncode != 0:
        raise RuntimeError(f'build failed:\n{create.stderr}\n{create.stdout}')

    tmp_dir_files = list(pathlib.Path(tmp_dir).iterdir())

    if len(tmp_dir_files) != 2:
        raise RuntimeError(f'build resulted in {len(tmp_dir_files)} items instead of 2')

    sdist_path = [path for path in tmp_dir_files if path.suffix == '.gz'][0]
    wheel_path = [path for path in tmp_dir_files if path.suffix == '.whl'][0]

    return sdist_path, wheel_path


def list_sdist(path: pathlib.Path) -> list[str]:
    """Return a list of the files in the sdist."""
    item: tarfile.TarInfo

    with tarfile.open(path) as sdist:
        paths = ['/'.join(pathlib.Path(item.path).parts[1:]) for item in sdist.getmembers() if not item.isdir()]

    return paths


def list_wheel(path: pathlib.Path) -> list[str]:
    """Return a list of the files in the wheel."""
    with zipfile.ZipFile(path) as wheel:
        paths = [item.filename for item in wheel.filelist if not item.is_dir()]

    return paths


def check_files(source: str, expected: list[str], actual: list[str]) -> list[str]:
    """Verify the expected files exist and no extra files exist."""
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))

    errors = (
        [f'{path}: missing from {source}' for path in missing] +
        [f'{path}: unexpected in {source}' for path in extra]
    )

    return errors


def main() -> None:
    """Main program entry point."""
    complete_file_list = sys.argv[1:] or sys.stdin.read().splitlines()

    errors = []

    # Limit visible files to those reported by ansible-test.
    # This avoids including files which are not committed to git.
    with clean_repository(complete_file_list) as clean_repo_dir:
        if __version__.endswith('.dev0'):
            # Make sure a changelog exists for this version when testing from devel.
            # When testing from a stable branch the changelog will already exist.
            major_minor_version = '.'.join(__version__.split('.')[:2])
            changelog_path = f'changelogs/CHANGELOG-v{major_minor_version}.rst'
            pathlib.Path(clean_repo_dir, changelog_path).touch()
            complete_file_list.append(changelog_path)

        expected_sdist_files = collect_sdist_files(complete_file_list)
        expected_wheel_files = collect_wheel_files(complete_file_list)

        with tempfile.TemporaryDirectory() as tmp_dir:
            sdist_path, wheel_path = build(clean_repo_dir, tmp_dir)

            actual_sdist_files = list_sdist(sdist_path)
            actual_wheel_files = list_wheel(wheel_path)

            errors.extend(check_files('sdist', expected_sdist_files, actual_sdist_files))
            errors.extend(check_files('wheel', expected_wheel_files, actual_wheel_files))

    for error in errors:
        print(error)


if __name__ == '__main__':
    main()
