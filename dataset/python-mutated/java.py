from dagger import CacheVolume, Container, File, Platform
from pipelines.airbyte_ci.connectors.context import ConnectorContext, PipelineContext
from pipelines.consts import AMAZONCORRETTO_IMAGE
from pipelines.dagger.actions.connector.hooks import finalize_build
from pipelines.dagger.actions.connector.normalization import DESTINATION_NORMALIZATION_BUILD_CONFIGURATION, with_normalization
from pipelines.helpers.utils import sh_dash_c

def with_integration_base(context: PipelineContext, build_platform: Platform) -> Container:
    if False:
        print('Hello World!')
    return context.dagger_client.container(platform=build_platform).from_('amazonlinux:2022.0.20220831.1').with_workdir('/airbyte').with_file('base.sh', context.get_repo_dir('airbyte-integrations/bases/base', include=['base.sh']).file('base.sh')).with_env_variable('AIRBYTE_ENTRYPOINT', '/airbyte/base.sh').with_label('io.airbyte.version', '0.1.0').with_label('io.airbyte.name', 'airbyte/integration-base')

def with_integration_base_java(context: PipelineContext, build_platform: Platform) -> Container:
    if False:
        print('Hello World!')
    integration_base = with_integration_base(context, build_platform)
    yum_packages_to_install = ['tar', 'openssl', 'findutils']
    return context.dagger_client.container(platform=build_platform).from_(AMAZONCORRETTO_IMAGE).with_exec(sh_dash_c(['yum update -y', f"yum install -y {' '.join(yum_packages_to_install)}", 'yum clean all'])).with_directory('/airbyte', integration_base.directory('/airbyte')).with_workdir('/airbyte').with_file('dd-java-agent.jar', context.dagger_client.http('https://dtdg.co/latest-java-tracer')).with_file('javabase.sh', context.get_repo_dir('airbyte-integrations/bases/base-java', include=['javabase.sh']).file('javabase.sh')).with_env_variable('AIRBYTE_SPEC_CMD', '/airbyte/javabase.sh --spec').with_env_variable('AIRBYTE_CHECK_CMD', '/airbyte/javabase.sh --check').with_env_variable('AIRBYTE_DISCOVER_CMD', '/airbyte/javabase.sh --discover').with_env_variable('AIRBYTE_READ_CMD', '/airbyte/javabase.sh --read').with_env_variable('AIRBYTE_WRITE_CMD', '/airbyte/javabase.sh --write').with_env_variable('AIRBYTE_ENTRYPOINT', '/airbyte/base.sh').with_label('io.airbyte.version', '0.1.2').with_label('io.airbyte.name', 'airbyte/integration-base-java')

def with_integration_base_java_and_normalization(context: PipelineContext, build_platform: Platform) -> Container:
    if False:
        i = 10
        return i + 15
    yum_packages_to_install = ['python3', 'python3-devel', 'jq', 'sshpass', 'git']
    additional_yum_packages = DESTINATION_NORMALIZATION_BUILD_CONFIGURATION[context.connector.technical_name]['yum_packages']
    yum_packages_to_install += additional_yum_packages
    dbt_adapter_package = DESTINATION_NORMALIZATION_BUILD_CONFIGURATION[context.connector.technical_name]['dbt_adapter']
    normalization_integration_name = DESTINATION_NORMALIZATION_BUILD_CONFIGURATION[context.connector.technical_name]['integration_name']
    pip_cache: CacheVolume = context.dagger_client.cache_volume('pip_cache')
    return with_integration_base_java(context, build_platform).with_exec(sh_dash_c(['yum update -y', f"yum install -y {' '.join(yum_packages_to_install)}", 'yum clean all', 'alternatives --install /usr/bin/python python /usr/bin/python3 60'])).with_mounted_cache('/root/.cache/pip', pip_cache).with_exec(sh_dash_c(['python -m ensurepip --upgrade', "pip3 install 'Cython<3.0' 'pyyaml~=5.4' --no-build-isolation", "pip3 install 'pytz~=2023.3'", f'pip3 install {dbt_adapter_package}', "pip3 install 'urllib3<2'"])).with_directory('airbyte_normalization', with_normalization(context, build_platform).directory('/airbyte')).with_workdir('airbyte_normalization').with_exec(sh_dash_c(['mv * ..'])).with_workdir('/airbyte').with_exec(['rm', '-rf', 'airbyte_normalization']).with_workdir('/airbyte/normalization_code').with_exec(['pip3', 'install', '.']).with_workdir('/airbyte/normalization_code/dbt-template/').with_exec(['dbt', 'deps']).with_workdir('/airbyte').with_file('run_with_normalization.sh', context.get_repo_dir('airbyte-integrations/bases/base-java', include=['run_with_normalization.sh']).file('run_with_normalization.sh')).with_env_variable('AIRBYTE_NORMALIZATION_INTEGRATION', normalization_integration_name).with_env_variable('AIRBYTE_ENTRYPOINT', '/airbyte/run_with_normalization.sh')

async def with_airbyte_java_connector(context: ConnectorContext, connector_java_tar_file: File, build_platform: Platform) -> Container:
    application = context.connector.technical_name
    build_stage = with_integration_base_java(context, build_platform).with_workdir('/airbyte').with_env_variable('APPLICATION', context.connector.technical_name).with_file(f'{application}.tar', connector_java_tar_file).with_exec(sh_dash_c([f'tar xf {application}.tar --strip-components=1', f'rm -rf {application}.tar']))
    if context.connector.supports_normalization and DESTINATION_NORMALIZATION_BUILD_CONFIGURATION[context.connector.technical_name]['supports_in_connector_normalization']:
        base = with_integration_base_java_and_normalization(context, build_platform)
        entrypoint = ['/airbyte/run_with_normalization.sh']
    else:
        base = with_integration_base_java(context, build_platform)
        entrypoint = ['/airbyte/base.sh']
    connector_container = base.with_workdir('/airbyte').with_env_variable('APPLICATION', application).with_mounted_directory('built_artifacts', build_stage.directory('/airbyte')).with_exec(sh_dash_c(['mv built_artifacts/* .'])).with_label('io.airbyte.version', context.metadata['dockerImageTag']).with_label('io.airbyte.name', context.metadata['dockerRepository']).with_entrypoint(entrypoint)
    return await finalize_build(context, connector_container)