from __future__ import annotations
import re
from pathlib import Path
from typing import TYPE_CHECKING
import pytest
from poetry.core.constraints.version import parse_constraint
from poetry.installation.wheel_installer import WheelInstaller
from poetry.utils.env import MockEnv
if TYPE_CHECKING:
    from _pytest.tmpdir import TempPathFactory
    from tests.types import FixtureDirGetter

@pytest.fixture
def env(tmp_path: Path) -> MockEnv:
    if False:
        while True:
            i = 10
    return MockEnv(path=tmp_path)

@pytest.fixture(scope='module')
def demo_wheel(fixture_dir: FixtureDirGetter) -> Path:
    if False:
        i = 10
        return i + 15
    return fixture_dir('distributions/demo-0.1.0-py2.py3-none-any.whl')

@pytest.fixture(scope='module')
def default_installation(tmp_path_factory: TempPathFactory, demo_wheel: Path) -> Path:
    if False:
        return 10
    env = MockEnv(path=tmp_path_factory.mktemp('default_install'))
    installer = WheelInstaller(env)
    installer.install(demo_wheel)
    return Path(env.paths['purelib'])

def test_default_installation_source_dir_content(default_installation: Path) -> None:
    if False:
        for i in range(10):
            print('nop')
    source_dir = default_installation / 'demo'
    assert source_dir.exists()
    assert (source_dir / '__init__.py').exists()

def test_default_installation_dist_info_dir_content(default_installation: Path) -> None:
    if False:
        print('Hello World!')
    dist_info_dir = default_installation / 'demo-0.1.0.dist-info'
    assert dist_info_dir.exists()
    assert (dist_info_dir / 'INSTALLER').exists()
    assert (dist_info_dir / 'METADATA').exists()
    assert (dist_info_dir / 'RECORD').exists()
    assert (dist_info_dir / 'WHEEL').exists()

def test_installer_file_contains_valid_version(default_installation: Path) -> None:
    if False:
        while True:
            i = 10
    installer_file = default_installation / 'demo-0.1.0.dist-info' / 'INSTALLER'
    with open(installer_file) as f:
        installer_content = f.read()
    match = re.match('Poetry (?P<version>.*)', installer_content)
    assert match
    parse_constraint(match.group('version'))

def test_default_installation_no_bytecode(default_installation: Path) -> None:
    if False:
        while True:
            i = 10
    cache_dir = default_installation / 'demo' / '__pycache__'
    assert not cache_dir.exists()

@pytest.mark.parametrize('compile', [True, False])
def test_enable_bytecode_compilation(env: MockEnv, demo_wheel: Path, compile: bool) -> None:
    if False:
        print('Hello World!')
    installer = WheelInstaller(env)
    installer.enable_bytecode_compilation(compile)
    installer.install(demo_wheel)
    cache_dir = Path(env.paths['purelib']) / 'demo' / '__pycache__'
    if compile:
        assert cache_dir.exists()
        assert list(cache_dir.glob('*.pyc'))
        assert not list(cache_dir.glob('*.opt-1.pyc'))
        assert not list(cache_dir.glob('*.opt-2.pyc'))
    else:
        assert not cache_dir.exists()