import os
from glob import glob
from typing import List
from dagster_buildkite.python_packages import PythonPackages
from ..defines import GIT_REPO_ROOT
from ..python_version import AvailablePythonVersion
from ..step_builder import CommandStepBuilder
from ..utils import BuildkiteStep, CommandStep, is_feature_branch, is_release_branch, safe_getenv, skip_if_no_python_changes
from .helm import build_helm_steps
from .integration import build_integration_steps
from .packages import build_library_packages_steps
from .test_project import build_test_project_steps
branch_name = safe_getenv('BUILDKITE_BRANCH')

def build_repo_wide_steps() -> List[BuildkiteStep]:
    if False:
        i = 10
        return i + 15
    return [*build_check_changelog_steps(), *build_repo_wide_check_manifest_steps(), *build_repo_wide_pyright_steps(), *build_repo_wide_ruff_steps()]

def build_dagster_steps() -> List[BuildkiteStep]:
    if False:
        while True:
            i = 10
    steps: List[BuildkiteStep] = []
    steps += build_library_packages_steps()
    steps += build_helm_steps()
    steps += build_sql_schema_check_steps()
    steps += build_graphql_python_client_backcompat_steps()
    if not os.getenv('CI_DISABLE_INTEGRATION_TESTS'):
        steps += build_integration_steps()
    if not os.getenv('CI_DISABLE_INTEGRATION_TESTS'):
        steps += build_test_project_steps()
    return steps

def build_repo_wide_ruff_steps() -> List[CommandStep]:
    if False:
        i = 10
        return i + 15
    return [CommandStepBuilder(':zap: ruff').run('pip install -e python_modules/dagster[ruff] -e python_modules/dagster-pipes', 'make check_ruff').on_test_image(AvailablePythonVersion.get_default()).with_skip(skip_if_no_python_changes()).build()]

def build_check_changelog_steps() -> List[CommandStep]:
    if False:
        return 10
    branch_name = safe_getenv('BUILDKITE_BRANCH')
    if not is_release_branch(branch_name):
        return []
    release_number = branch_name.split('-', 1)[-1].replace('-', '.')
    return [CommandStepBuilder(':memo: changelog').on_test_image(AvailablePythonVersion.get_default()).run(f'python scripts/check_changelog.py {release_number}').build()]

def build_repo_wide_pyright_steps() -> List[CommandStep]:
    if False:
        i = 10
        return i + 15
    return [CommandStepBuilder(':pyright: pyright').run('curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain nightly -y', 'pip install -e python_modules/dagster[pyright] -e python_modules/dagster-pipes', 'make pyright').on_test_image(AvailablePythonVersion.get_default()).with_skip(skip_if_no_python_changes()).build()]

def build_repo_wide_check_manifest_steps() -> List[CommandStep]:
    if False:
        print('Hello World!')
    published_packages = ['python_modules/dagit', 'python_modules/dagster', 'python_modules/dagster-graphql', 'python_modules/dagster-webserver', *(os.path.relpath(p, GIT_REPO_ROOT) for p in glob(f'{GIT_REPO_ROOT}/python_modules/libraries/*'))]
    commands = ['pip install check-manifest', *(f'check-manifest {library}' for library in published_packages if not library.endswith('CONTRIBUTING.md'))]
    return [CommandStepBuilder(':white_check_mark: check-manifest').on_test_image(AvailablePythonVersion.get_default()).run(*commands).with_skip(skip_if_no_python_changes()).build()]

def build_sql_schema_check_steps() -> List[CommandStep]:
    if False:
        while True:
            i = 10
    return [CommandStepBuilder(':mysql: mysql-schema').on_test_image(AvailablePythonVersion.get_default()).run('pip install -e python_modules/dagster -e python_modules/dagster-pipes', 'python scripts/check_schemas.py').with_skip(skip_mysql_if_no_changes_to_dependencies(['dagster'])).build()]

def build_graphql_python_client_backcompat_steps() -> List[CommandStep]:
    if False:
        while True:
            i = 10
    return [CommandStepBuilder(':graphql: GraphQL Python Client backcompat').on_test_image(AvailablePythonVersion.get_default()).run('pip install -e python_modules/dagster[test] -e python_modules/dagster-pipes -e python_modules/dagster-graphql -e python_modules/automation', 'dagster-graphql-client query check').with_skip(skip_graphql_if_no_changes_to_dependencies(['dagster', 'dagster-graphql', 'automation'])).build()]

def skip_mysql_if_no_changes_to_dependencies(dependencies: List[str]):
    if False:
        i = 10
        return i + 15
    if not is_feature_branch():
        return None
    for dependency in dependencies:
        if PythonPackages.get(dependency) in PythonPackages.with_changes:
            return None
    return 'Skip unless mysql schemas might have changed'

def skip_graphql_if_no_changes_to_dependencies(dependencies: List[str]):
    if False:
        for i in range(10):
            print('nop')
    if not is_feature_branch():
        return None
    for dependency in dependencies:
        if PythonPackages.get(dependency) in PythonPackages.with_changes:
            return None
    return 'Skip unless GraphQL schemas might have changed'