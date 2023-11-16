from abc import ABC
from typing import ClassVar, List
import pipelines.dagger.actions.system.docker
from dagger import CacheSharingMode, CacheVolume
from pipelines.consts import AMAZONCORRETTO_IMAGE
from pipelines.dagger.actions import secrets
from pipelines.helpers.utils import sh_dash_c
from pipelines.models.contexts.pipeline_context import PipelineContext
from pipelines.models.steps import Step, StepResult

class GradleTask(Step, ABC):
    """
    A step to run a Gradle task.

    Attributes:
        title (str): The step title.
        gradle_task_name (str): The Gradle task name to run.
        bind_to_docker_host (bool): Whether to install the docker client and bind it to the host.
        mount_connector_secrets (bool): Whether to mount connector secrets.
    """
    DEFAULT_GRADLE_TASK_OPTIONS = ('--no-daemon', '--no-watch-fs', '--scan', '--build-cache', '--console=plain')
    LOCAL_MAVEN_REPOSITORY_PATH = '/root/.m2'
    GRADLE_DEP_CACHE_PATH = '/root/gradle-cache'
    GRADLE_HOME_PATH = '/root/.gradle'
    gradle_task_name: ClassVar[str]
    bind_to_docker_host: ClassVar[bool] = False
    mount_connector_secrets: ClassVar[bool] = False

    def __init__(self, context: PipelineContext) -> None:
        if False:
            print('Hello World!')
        super().__init__(context)

    @property
    def dependency_cache_volume(self) -> CacheVolume:
        if False:
            return 10
        'This cache volume is for sharing gradle dependencies (jars and poms) across all pipeline runs.'
        return self.context.dagger_client.cache_volume('gradle-dependency-cache')

    @property
    def build_include(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        'Retrieve the list of source code directory required to run a Java connector Gradle task.\n\n        The list is different according to the connector type.\n\n        Returns:\n            List[str]: List of directories or files to be mounted to the container to run a Java connector Gradle task.\n        '
        return [str(dependency_directory) for dependency_directory in self.context.connector.get_local_dependency_paths(with_test_dependencies=True)]

    def _get_gradle_command(self, task: str, *args) -> str:
        if False:
            return 10
        return f"./gradlew {' '.join(self.DEFAULT_GRADLE_TASK_OPTIONS + args)} {task}"

    async def _run(self) -> StepResult:
        include = ['.root', '.env', 'build.gradle', 'deps.toml', 'gradle.properties', 'gradle', 'gradlew', 'settings.gradle', 'build.gradle', 'tools/gradle', 'spotbugs-exclude-filter-file.xml', 'buildSrc', 'tools/bin/build_image.sh', 'tools/lib/lib.sh', 'pyproject.toml'] + self.build_include
        yum_packages_to_install = ['docker', 'findutils', 'jq', 'rsync']
        gradle_container_base = self.dagger_client.container().from_(AMAZONCORRETTO_IMAGE).with_mounted_cache(self.GRADLE_DEP_CACHE_PATH, self.dependency_cache_volume, sharing=CacheSharingMode.LOCKED).with_env_variable('GRADLE_HOME', self.GRADLE_HOME_PATH).with_env_variable('GRADLE_USER_HOME', self.GRADLE_HOME_PATH).with_exec(sh_dash_c(['yum update -y', f"yum install -y {' '.join(yum_packages_to_install)}", 'yum clean all', 'yum remove -y --noautoremove docker', 'yum install -y --downloadonly docker'])).with_env_variable('RUN_IN_AIRBYTE_CI', '1').with_env_variable('TESTCONTAINERS_RYUK_DISABLED', 'true').with_workdir('/airbyte')
        if self.context.s3_build_cache_access_key_id:
            gradle_container_base = gradle_container_base.with_secret_variable('S3_BUILD_CACHE_ACCESS_KEY_ID', self.context.s3_build_cache_access_key_id_secret)
            if self.context.s3_build_cache_secret_key:
                gradle_container_base = gradle_container_base.with_secret_variable('S3_BUILD_CACHE_SECRET_KEY', self.context.s3_build_cache_secret_key_secret)
        warm_dependency_cache_args = ['--write-verification-metadata', 'sha256', '--dry-run']
        if self.context.is_local:
            warm_dependency_cache_args = ['--dry-run']
        with_whole_git_repo = gradle_container_base.with_directory('/airbyte', self.context.get_repo_dir('.')).with_exec(sh_dash_c([f'mkdir -p {self.LOCAL_MAVEN_REPOSITORY_PATH}', f'(rsync -a --stats --mkpath {self.GRADLE_DEP_CACHE_PATH}/ {self.GRADLE_HOME_PATH} || true)', self._get_gradle_command('help', *warm_dependency_cache_args), self._get_gradle_command(':airbyte-cdk:java:airbyte-cdk:publishSnapshotIfNeeded'), f'(rsync -a --stats {self.GRADLE_HOME_PATH}/ {self.GRADLE_DEP_CACHE_PATH} || true)']))
        gradle_container = gradle_container_base.with_directory(self.LOCAL_MAVEN_REPOSITORY_PATH, await with_whole_git_repo.directory(self.LOCAL_MAVEN_REPOSITORY_PATH)).with_mounted_directory('/airbyte', self.context.get_repo_dir('.', include=include)).with_mounted_directory(str(self.context.connector.code_directory), await self.context.get_connector_dir())
        if self.mount_connector_secrets:
            secrets_dir = f'{self.context.connector.code_directory}/secrets'
            gradle_container = gradle_container.with_(await secrets.mounted_connector_secrets(self.context, secrets_dir))
        if self.bind_to_docker_host:
            gradle_container = pipelines.dagger.actions.system.docker.with_bound_docker_host(self.context, gradle_container)
            gradle_container = gradle_container.with_exec(['yum', 'install', '-y', 'docker'])
        connector_task = f':airbyte-integrations:connectors:{self.context.connector.technical_name}:{self.gradle_task_name}'
        gradle_container = gradle_container.with_exec(sh_dash_c([f'(rsync -a --stats --mkpath {self.GRADLE_DEP_CACHE_PATH}/ {self.GRADLE_HOME_PATH} || true)', self._get_gradle_command(connector_task, f'-Ds3BuildCachePrefix={self.context.connector.technical_name}')]))
        return await self.get_step_result(gradle_container)