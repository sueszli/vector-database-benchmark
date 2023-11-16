from dagger import CacheSharingMode, CacheVolume, Client, Container
from pipelines.airbyte_ci.connectors.context import PipelineContext
from pipelines.consts import CONNECTOR_TESTING_REQUIREMENTS, PIP_CACHE_PATH, PIP_CACHE_VOLUME_NAME, POETRY_CACHE_PATH, POETRY_CACHE_VOLUME_NAME, PYPROJECT_TOML_FILE_PATH
from pipelines.helpers.utils import sh_dash_c

def with_python_base(context: PipelineContext, python_version: str='3.10') -> Container:
    if False:
        for i in range(10):
            print('nop')
    'Build a Python container with a cache volume for pip cache.\n\n    Args:\n        context (PipelineContext): The current test context, providing a dagger client and a repository directory.\n        python_image_name (str, optional): The python image to use to build the python base environment. Defaults to "python:3.9-slim".\n\n    Raises:\n        ValueError: Raised if the python_image_name is not a python image.\n\n    Returns:\n        Container: The python base environment container.\n    '
    pip_cache: CacheVolume = context.dagger_client.cache_volume('pip_cache')
    base_container = context.dagger_client.container().from_(f'python:{python_version}-slim').with_mounted_cache('/root/.cache/pip', pip_cache).with_exec(sh_dash_c(['apt-get update', 'apt-get install -y build-essential cmake g++ libffi-dev libstdc++6 git', 'pip install pip==23.1.2']))
    return base_container

def with_testing_dependencies(context: PipelineContext) -> Container:
    if False:
        for i in range(10):
            print('nop')
    'Build a testing environment by installing testing dependencies on top of a python base environment.\n\n    Args:\n        context (PipelineContext): The current test context, providing a dagger client and a repository directory.\n\n    Returns:\n        Container: The testing environment container.\n    '
    python_environment: Container = with_python_base(context)
    pyproject_toml_file = context.get_repo_dir('.', include=[PYPROJECT_TOML_FILE_PATH]).file(PYPROJECT_TOML_FILE_PATH)
    return python_environment.with_exec(['pip', 'install'] + CONNECTOR_TESTING_REQUIREMENTS).with_file(f'/{PYPROJECT_TOML_FILE_PATH}', pyproject_toml_file)

def with_pip_cache(container: Container, dagger_client: Client) -> Container:
    if False:
        for i in range(10):
            print('nop')
    'Mounts the pip cache in the container.\n    Args:\n        container (Container): A container with python installed\n\n    Returns:\n        Container: A container with the pip cache mounted.\n    '
    pip_cache_volume = dagger_client.cache_volume(PIP_CACHE_VOLUME_NAME)
    return container.with_mounted_cache(PIP_CACHE_PATH, pip_cache_volume, sharing=CacheSharingMode.SHARED)

def with_poetry_cache(container: Container, dagger_client: Client) -> Container:
    if False:
        while True:
            i = 10
    'Mounts the poetry cache in the container.\n    Args:\n        container (Container): A container with python installed\n\n    Returns:\n        Container: A container with the poetry cache mounted.\n    '
    poetry_cache_volume = dagger_client.cache_volume(POETRY_CACHE_VOLUME_NAME)
    return container.with_mounted_cache(POETRY_CACHE_PATH, poetry_cache_volume, sharing=CacheSharingMode.SHARED)