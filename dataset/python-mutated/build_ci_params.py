from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from airflow_breeze.branch_defaults import DEFAULT_AIRFLOW_CONSTRAINTS_BRANCH
from airflow_breeze.global_constants import get_airflow_version
from airflow_breeze.params.common_build_params import CommonBuildParams
from airflow_breeze.utils.path_utils import BUILD_CACHE_DIR

@dataclass
class BuildCiParams(CommonBuildParams):
    """
    CI build parameters. Those parameters are used to determine command issued to build CI image.
    """
    airflow_constraints_mode: str = 'constraints-source-providers'
    airflow_constraints_reference: str = DEFAULT_AIRFLOW_CONSTRAINTS_BRANCH
    airflow_extras: str = 'devel_ci'
    airflow_pre_cached_pip_packages: bool = True
    force_build: bool = False
    upgrade_to_newer_dependencies: bool = False
    upgrade_on_failure: bool = False
    eager_upgrade_additional_requirements: str = ''
    skip_provider_dependencies_check: bool = False

    @property
    def airflow_version(self):
        if False:
            print('Hello World!')
        return get_airflow_version()

    @property
    def image_type(self) -> str:
        if False:
            while True:
                i = 10
        return 'CI'

    @property
    def extra_docker_build_flags(self) -> list[str]:
        if False:
            for i in range(10):
                print('nop')
        extra_ci_flags = []
        extra_ci_flags.extend(['--build-arg', f'AIRFLOW_CONSTRAINTS_REFERENCE={self.airflow_constraints_reference}'])
        if self.airflow_constraints_location:
            extra_ci_flags.extend(['--build-arg', f'AIRFLOW_CONSTRAINTS_LOCATION={self.airflow_constraints_location}'])
        if self.upgrade_to_newer_dependencies:
            eager_upgrade_arg = self.eager_upgrade_additional_requirements.strip().replace('\n', '')
            if eager_upgrade_arg:
                extra_ci_flags.extend(['--build-arg', f'EAGER_UPGRADE_ADDITIONAL_REQUIREMENTS={eager_upgrade_arg}'])
        return super().extra_docker_build_flags + extra_ci_flags

    @property
    def md5sum_cache_dir(self) -> Path:
        if False:
            i = 10
            return i + 15
        return Path(BUILD_CACHE_DIR, self.airflow_branch, self.python, 'CI')

    @property
    def required_image_args(self) -> list[str]:
        if False:
            return 10
        return ['airflow_branch', 'airflow_constraints_mode', 'airflow_constraints_reference', 'airflow_extras', 'airflow_image_date_created', 'airflow_image_repository', 'airflow_pre_cached_pip_packages', 'airflow_version', 'build_id', 'constraints_github_repository', 'python_base_image', 'upgrade_to_newer_dependencies']

    @property
    def optional_image_args(self) -> list[str]:
        if False:
            i = 10
            return i + 15
        return ['additional_airflow_extras', 'additional_dev_apt_command', 'additional_dev_apt_deps', 'additional_dev_apt_env', 'additional_pip_install_flags', 'additional_python_deps', 'additional_runtime_apt_command', 'additional_runtime_apt_deps', 'additional_runtime_apt_env', 'dev_apt_command', 'dev_apt_deps', 'additional_dev_apt_command', 'additional_dev_apt_deps', 'additional_dev_apt_env', 'additional_airflow_extras', 'additional_pip_install_flags', 'additional_python_deps', 'version_suffix_for_pypi', 'commit_sha', 'build_progress']

    def __post_init__(self):
        if False:
            while True:
                i = 10
        pass