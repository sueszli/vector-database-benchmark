from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable
import pytest
from pytest_mock import MockerFixture
from conda.base.constants import CONDA_LOGS_DIR, CONDA_PACKAGE_EXTENSIONS, CONDA_TEMP_EXTENSIONS
from conda.cli.main_clean import _get_size
from conda.core.subdir_data import create_cache_dir
from conda.gateways.logging import set_verbosity
from conda.testing import CondaCLIFixture, TmpEnvFixture
from conda.testing.integration import make_temp_package_cache

def _get_pkgs(pkgs_dir: str | Path) -> list[Path]:
    if False:
        i = 10
        return i + 15
    return [package for package in Path(pkgs_dir).iterdir() if package.is_dir()]

def _get_tars(pkgs_dir: str | Path) -> list[Path]:
    if False:
        i = 10
        return i + 15
    return [file for file in Path(pkgs_dir).iterdir() if file.is_file() and file.name.endswith(CONDA_PACKAGE_EXTENSIONS)]

def _get_index_cache() -> list[Path]:
    if False:
        for i in range(10):
            print('nop')
    return [file for file in Path(create_cache_dir()).iterdir() if file.is_file() and file.name.endswith('.json')]

def _get_tempfiles(pkgs_dir: str | Path) -> list[Path]:
    if False:
        print('Hello World!')
    return [file for file in Path(pkgs_dir).iterdir() if file.is_file() and file.name.endswith(CONDA_TEMP_EXTENSIONS)]

def _get_logfiles(pkgs_dir: str | Path) -> list[Path]:
    if False:
        for i in range(10):
            print('nop')
    try:
        return [file for file in Path(pkgs_dir, CONDA_LOGS_DIR).iterdir()]
    except FileNotFoundError:
        return []

def _get_all(pkgs_dir: str | Path) -> tuple[list[Path], list[Path], list[Path]]:
    if False:
        i = 10
        return i + 15
    return (_get_pkgs(pkgs_dir), _get_tars(pkgs_dir), _get_index_cache())

def has_pkg(name: str, contents: Iterable[str | Path]) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return any((Path(content).name.startswith(f'{name}-') for content in contents))

def test_clean_force_pkgs_dirs(clear_cache, conda_cli: CondaCLIFixture, tmp_env: TmpEnvFixture):
    if False:
        i = 10
        return i + 15
    pkg = 'zlib'
    with make_temp_package_cache() as pkgs_dir:
        pkgs_dir = Path(pkgs_dir)
        assert pkgs_dir.is_dir()
        with tmp_env(pkg):
            (stdout, _, _) = conda_cli('clean', '--force-pkgs-dirs', '--yes', '--json')
            json.loads(stdout)
            assert not pkgs_dir.exists()
        assert not pkgs_dir.exists()

def test_clean_and_packages(clear_cache, conda_cli: CondaCLIFixture, tmp_env: TmpEnvFixture):
    if False:
        while True:
            i = 10
    pkg = 'zlib'
    with make_temp_package_cache() as pkgs_dir:
        assert not has_pkg(pkg, _get_pkgs(pkgs_dir))
        with tmp_env(pkg) as prefix:
            assert has_pkg(pkg, _get_pkgs(pkgs_dir))
            (stdout, _, _) = conda_cli('clean', '--packages', '--yes', '--json')
            json.loads(stdout)
            assert has_pkg(pkg, _get_pkgs(pkgs_dir))
            conda_cli('remove', '--prefix', prefix, pkg, '--yes', '--json')
            (stdout, _, _) = conda_cli('clean', '--packages', '--yes', '--json')
            json.loads(stdout)
            assert not has_pkg(pkg, _get_pkgs(pkgs_dir))
        assert not has_pkg(pkg, _get_pkgs(pkgs_dir))

def test_clean_tarballs(clear_cache, conda_cli: CondaCLIFixture, tmp_env: TmpEnvFixture):
    if False:
        return 10
    pkg = 'zlib'
    with make_temp_package_cache() as pkgs_dir:
        assert not has_pkg(pkg, _get_tars(pkgs_dir))
        with tmp_env(pkg):
            assert has_pkg(pkg, _get_tars(pkgs_dir))
            (stdout, _, _) = conda_cli('clean', '--tarballs', '--yes', '--json')
            json.loads(stdout)
            assert not has_pkg(pkg, _get_tars(pkgs_dir))
        assert not has_pkg(pkg, _get_tars(pkgs_dir))

def test_clean_index_cache(clear_cache, conda_cli: CondaCLIFixture, tmp_env: TmpEnvFixture):
    if False:
        while True:
            i = 10
    pkg = 'zlib'
    with make_temp_package_cache():
        assert not _get_index_cache()
        with tmp_env(pkg):
            assert _get_index_cache()
            (stdout, _, _) = conda_cli('clean', '--index-cache', '--yes', '--json')
            json.loads(stdout)
            assert not _get_index_cache()
        assert not _get_index_cache()

def test_clean_tempfiles(clear_cache, conda_cli: CondaCLIFixture, tmp_env: TmpEnvFixture):
    if False:
        i = 10
        return i + 15
    'Tempfiles are either suffixed with .c~ or .trash.\n\n    .c~ is used to indicate that conda is actively using that file. If the conda process is\n    terminated unexpectedly these .c~ files may remain and hence can be cleaned up after the fact.\n\n    .trash appears to be a legacy suffix that is no longer used by conda.\n\n    Since the presence of .c~ and .trash files are dependent upon irregular termination we create\n    our own temporary files to confirm they get cleaned up.\n    '
    pkg = 'zlib'
    with make_temp_package_cache() as pkgs_dir:
        assert not _get_tempfiles(pkgs_dir)
        with tmp_env(pkg):
            path = _get_tars(pkgs_dir)[0]
            for ext in CONDA_TEMP_EXTENSIONS:
                (path.parent / f'{path.name}{ext}').touch()
            assert len(_get_tempfiles(pkgs_dir)) == len(CONDA_TEMP_EXTENSIONS)
            (stdout, _, _) = conda_cli('clean', '--tempfiles', pkgs_dir, '--yes', '--json')
            json.loads(stdout)
            assert not _get_tempfiles(pkgs_dir)
        assert not _get_tempfiles(pkgs_dir)

def test_clean_logfiles(clear_cache, conda_cli: CondaCLIFixture, tmp_env: TmpEnvFixture):
    if False:
        return 10
    'Logfiles are found in pkgs_dir/.logs.\n\n    Since these log files were uniquely created during the experimental\n    phase of the conda-libmamba-solver.\n    '
    pkg = 'zlib'
    with make_temp_package_cache() as pkgs_dir:
        assert not _get_logfiles(pkgs_dir)
        with tmp_env(pkg):
            logs_dir = Path(pkgs_dir, CONDA_LOGS_DIR)
            logs_dir.mkdir(parents=True, exist_ok=True)
            path = logs_dir / f'{datetime.utcnow():%Y%m%d-%H%M%S-%f}.log'
            path.touch()
            assert path in _get_logfiles(pkgs_dir)
            (stdout, _, _) = conda_cli('clean', '--logfiles', '--yes', '--json')
            json.loads(stdout)
            assert not _get_logfiles(pkgs_dir)
        assert not _get_logfiles(pkgs_dir)

@pytest.mark.parametrize('verbose', [True, False])
def test_clean_all(clear_cache, verbose: bool, conda_cli: CondaCLIFixture, tmp_env: TmpEnvFixture):
    if False:
        print('Hello World!')
    pkg = 'zlib'
    args = ('--yes', '--json')
    if verbose:
        args = (*args, '--verbose')
    with make_temp_package_cache() as pkgs_dir:
        (pkgs, tars, cache) = _get_all(pkgs_dir)
        assert not has_pkg(pkg, pkgs)
        assert not has_pkg(pkg, tars)
        assert not cache
        with tmp_env(pkg) as prefix:
            (pkgs, tars, cache) = _get_all(pkgs_dir)
            assert has_pkg(pkg, pkgs)
            assert has_pkg(pkg, tars)
            assert cache
            (stdout, _, _) = conda_cli('clean', '--all', *args)
            json.loads(stdout)
            (pkgs, tars, cache) = _get_all(pkgs_dir)
            assert has_pkg(pkg, pkgs)
            assert not has_pkg(pkg, tars)
            assert not cache
            conda_cli('remove', '--prefix', prefix, pkg, *args)
            (stdout, _, _) = conda_cli('clean', '--packages', *args)
            json.loads(stdout)
            (pkgs, tars, index_cache) = _get_all(pkgs_dir)
            assert not has_pkg(pkg, pkgs)
            assert not has_pkg(pkg, tars)
            assert not cache
        (pkgs, tars, index_cache) = _get_all(pkgs_dir)
        assert not has_pkg(pkg, pkgs)
        assert not has_pkg(pkg, tars)
        assert not cache
    set_verbosity(0)

@pytest.mark.parametrize('as_json', [True, False])
def test_clean_all_mock_lstat(clear_cache, mocker: MockerFixture, as_json: bool, conda_cli: CondaCLIFixture, tmp_env: TmpEnvFixture):
    if False:
        i = 10
        return i + 15
    pkg = 'zlib'
    args = ('--yes', '--verbose')
    if as_json:
        args = (*args, '--json')
    with make_temp_package_cache() as pkgs_dir, tmp_env(pkg) as prefix:
        (pkgs, tars, cache) = _get_all(pkgs_dir)
        assert has_pkg(pkg, pkgs)
        assert has_pkg(pkg, tars)
        assert cache
        mocker.patch('os.lstat', side_effect=OSError)
        conda_cli('remove', '--prefix', prefix, pkg, *args)
        (stdout, _, _) = conda_cli('clean', '--packages', *args)
        assert 'WARNING:' in stdout
        if as_json:
            json.loads(stdout)
        (pkgs, tars, index_cache) = _get_all(pkgs_dir)
        assert has_pkg(pkg, pkgs)
        assert has_pkg(pkg, tars)
        assert cache
    set_verbosity(0)

def test_get_size(tmp_path: Path):
    if False:
        for i in range(10):
            print('nop')
    warnings: list[str] = []
    path = tmp_path / 'file'
    path.write_text('hello')
    assert _get_size(path, warnings=warnings)
    assert not warnings

def test_get_size_None():
    if False:
        i = 10
        return i + 15
    with pytest.raises(OSError):
        _get_size('not-a-file', warnings=None)

def test_get_size_list():
    if False:
        print('Hello World!')
    warnings: list[str] = []
    with pytest.raises(NotImplementedError):
        _get_size('not-a-file', warnings=warnings)
    assert warnings