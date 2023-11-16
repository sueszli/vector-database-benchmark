from contextlib import contextmanager
from os import PathLike
from typing import Generator, Optional, Union
from unittest.mock import patch
from synapse.util.check_dependencies import DependencyException, check_requirements, metadata
from tests.unittest import TestCase

class DummyDistribution(metadata.Distribution):

    def __init__(self, version: str):
        if False:
            i = 10
            return i + 15
        self._version = version

    @property
    def version(self) -> str:
        if False:
            while True:
                i = 10
        return self._version

    def locate_file(self, path: Union[str, PathLike]) -> PathLike:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def read_text(self, filename: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()
old = DummyDistribution('0.1.2')
old_release_candidate = DummyDistribution('0.1.2rc3')
new = DummyDistribution('1.2.3')
new_release_candidate = DummyDistribution('1.2.3rc4')
distribution_with_no_version = DummyDistribution(None)

class TestDependencyChecker(TestCase):

    @contextmanager
    def mock_installed_package(self, distribution: Optional[DummyDistribution]) -> Generator[None, None, None]:
        if False:
            for i in range(10):
                print('nop')
        'Pretend that looking up any package yields the given `distribution`.\n\n        If `distribution = None`, we pretend that the package is not installed.\n        '

        def mock_distribution(name: str) -> DummyDistribution:
            if False:
                while True:
                    i = 10
            if distribution is None:
                raise metadata.PackageNotFoundError
            else:
                return distribution
        with patch('synapse.util.check_dependencies.metadata.distribution', mock_distribution):
            yield

    def test_mandatory_dependency(self) -> None:
        if False:
            print('Hello World!')
        'Complain if a required package is missing or old.'
        with patch('synapse.util.check_dependencies.metadata.requires', return_value=['dummypkg >= 1']):
            with self.mock_installed_package(None):
                self.assertRaises(DependencyException, check_requirements)
            with self.mock_installed_package(old):
                self.assertRaises(DependencyException, check_requirements)
            with self.mock_installed_package(new):
                check_requirements()

    def test_version_reported_as_none(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Complain if importlib.metadata.version() returns None.\n\n        This shouldn't normally happen, but it was seen in the wild\n        (https://github.com/matrix-org/synapse/issues/12223).\n        "
        with patch('synapse.util.check_dependencies.metadata.requires', return_value=['dummypkg >= 1']):
            with self.mock_installed_package(distribution_with_no_version):
                self.assertRaises(DependencyException, check_requirements)

    def test_checks_ignore_dev_dependencies(self) -> None:
        if False:
            while True:
                i = 10
        'Both generic and per-extra checks should ignore dev dependencies.'
        with patch('synapse.util.check_dependencies.metadata.requires', return_value=["dummypkg >= 1; extra == 'mypy'"]), patch('synapse.util.check_dependencies.RUNTIME_EXTRAS', {'cool-extra'}):
            with self.mock_installed_package(None):
                check_requirements()
                check_requirements('cool-extra')
            with self.mock_installed_package(old):
                check_requirements()
                check_requirements('cool-extra')
            with self.mock_installed_package(new):
                check_requirements()
                check_requirements('cool-extra')

    def test_generic_check_of_optional_dependency(self) -> None:
        if False:
            while True:
                i = 10
        'Complain if an optional package is old.'
        with patch('synapse.util.check_dependencies.metadata.requires', return_value=["dummypkg >= 1; extra == 'cool-extra'"]):
            with self.mock_installed_package(None):
                check_requirements()
            with self.mock_installed_package(old):
                self.assertRaises(DependencyException, check_requirements)
            with self.mock_installed_package(new):
                check_requirements()

    def test_check_for_extra_dependencies(self) -> None:
        if False:
            return 10
        'Complain if a package required for an extra is missing or old.'
        with patch('synapse.util.check_dependencies.metadata.requires', return_value=["dummypkg >= 1; extra == 'cool-extra'"]), patch('synapse.util.check_dependencies.RUNTIME_EXTRAS', {'cool-extra'}):
            with self.mock_installed_package(None):
                self.assertRaises(DependencyException, check_requirements, 'cool-extra')
            with self.mock_installed_package(old):
                self.assertRaises(DependencyException, check_requirements, 'cool-extra')
            with self.mock_installed_package(new):
                check_requirements('cool-extra')

    def test_release_candidates_satisfy_dependency(self) -> None:
        if False:
            return 10
        '\n        Tests that release candidates count as far as satisfying a dependency\n        is concerned.\n        (Regression test, see https://github.com/matrix-org/synapse/issues/12176.)\n        '
        with patch('synapse.util.check_dependencies.metadata.requires', return_value=['dummypkg >= 1']):
            with self.mock_installed_package(old_release_candidate):
                self.assertRaises(DependencyException, check_requirements)
            with self.mock_installed_package(new_release_candidate):
                check_requirements()

    def test_setuptools_rust_ignored(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test a workaround for a `poetry build` problem. Reproduces\n        https://github.com/matrix-org/synapse/issues/13926.\n        '
        with patch('synapse.util.check_dependencies.metadata.requires', return_value=['setuptools_rust >= 1.3']):
            with self.mock_installed_package(None):
                check_requirements()
            with self.mock_installed_package(old):
                check_requirements()