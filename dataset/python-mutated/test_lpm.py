import os.path
from typing import List
import pytest
from click.testing import CliRunner
from localstack.cli.lpm import cli, console
from localstack.packages import InstallTarget, Package, PackageException, PackageInstaller
from localstack.packages.api import PackagesPluginManager
from localstack.testing.pytest import markers
from localstack.utils.patch import Patch

@pytest.fixture
def runner():
    if False:
        return 10
    return CliRunner()

@markers.skip_offline
def test_list(runner, monkeypatch):
    if False:
        return 10
    monkeypatch.setattr(console, 'no_color', True)
    result = runner.invoke(cli, ['list'])
    assert result.exit_code == 0
    assert 'kinesis-mock/community' in result.output

@markers.skip_offline
def test_install_with_non_existing_package_fails(runner):
    if False:
        for i in range(10):
            print('nop')
    result = runner.invoke(cli, ['install', 'kinesis-mock', 'funny'])
    assert result.exit_code == 1
    assert 'unable to locate installer for package funny' in result.output

@markers.skip_offline
def test_install_with_non_existing_version_fails(runner):
    if False:
        while True:
            i = 10
    result = runner.invoke(cli, ['install', 'kinesis-mock', '--version', 'non-existing-version'])
    assert result.exit_code == 1
    assert 'unable to locate installer for package kinesis-mock and version non-existing-version' in result.output

@markers.skip_offline
def test_install_failure_returns_non_zero_exit_code(runner, monkeypatch):
    if False:
        while True:
            i = 10

    class FailingPackage(Package):

        def __init__(self):
            if False:
                return 10
            super().__init__('Failing Installer', 'latest')

        def get_versions(self) -> List[str]:
            if False:
                for i in range(10):
                    print('nop')
            return ['latest']

        def _get_installer(self, version: str) -> PackageInstaller:
            if False:
                return 10
            return FailingInstaller()

    class FailingInstaller(PackageInstaller):

        def __init__(self):
            if False:
                return 10
            super().__init__('failing-installer', 'latest')

        def _get_install_marker_path(self, install_dir: str) -> str:
            if False:
                print('Hello World!')
            return '/non-existing'

        def _install(self, target: InstallTarget) -> None:
            if False:
                while True:
                    i = 10
            raise PackageException('Failing!')

    class SuccessfulPackage(Package):

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__('Successful Installer', 'latest')

        def get_versions(self) -> List[str]:
            if False:
                while True:
                    i = 10
            return ['latest']

        def _get_installer(self, version: str) -> PackageInstaller:
            if False:
                i = 10
                return i + 15
            return SuccessfulInstaller()

    class SuccessfulInstaller(PackageInstaller):

        def __init__(self):
            if False:
                return 10
            super().__init__('successful-installer', 'latest')

        def _get_install_marker_path(self, install_dir: str) -> str:
            if False:
                while True:
                    i = 10
            return '/non-existing'

        def _install(self, target: InstallTarget) -> None:
            if False:
                i = 10
                return i + 15
            pass

    def patched_get_packages(*_) -> List[Package]:
        if False:
            print('Hello World!')
        return [FailingPackage(), SuccessfulPackage()]
    with Patch.function(target=PackagesPluginManager.get_packages, fn=patched_get_packages):
        result = runner.invoke(cli, ['install', 'successful-installer', 'failing-installer'])
        assert result.exit_code == 1
        assert 'one or more package installations failed.' in result.output

@markers.skip_offline
def test_install_with_package(runner):
    if False:
        for i in range(10):
            print('nop')
    from localstack.services.kinesis.packages import kinesismock_package
    result = runner.invoke(cli, ['install', 'kinesis-mock'])
    assert result.exit_code == 0
    assert os.path.exists(kinesismock_package.get_installed_dir())