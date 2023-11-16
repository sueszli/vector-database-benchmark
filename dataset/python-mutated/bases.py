from __future__ import annotations
from typing import Callable, Final
import dagger
from base_images import bases, published_image
from base_images import sanity_checks as base_sanity_checks
from base_images.python import sanity_checks as python_sanity_checks
from base_images.root_images import PYTHON_3_9_18

class AirbytePythonConnectorBaseImage(bases.AirbyteConnectorBaseImage):
    root_image: Final[published_image.PublishedImage] = PYTHON_3_9_18
    repository: Final[str] = 'airbyte/python-connector-base'
    pip_cache_name: Final[str] = 'pip_cache'
    nltk_data_path: Final[str] = '/usr/share/nltk_data'
    ntlk_data = {'tokenizers': {'https://github.com/nltk/nltk_data/raw/5db857e6f7df11eabb5e5665836db9ec8df07e28/packages/tokenizers/punkt.zip'}, 'taggers': {'https://github.com/nltk/nltk_data/raw/5db857e6f7df11eabb5e5665836db9ec8df07e28/packages/taggers/averaged_perceptron_tagger.zip'}}

    def install_cdk_system_dependencies(self) -> Callable:
        if False:
            return 10

        def get_nltk_data_dir() -> dagger.Directory:
            if False:
                while True:
                    i = 10
            'Returns a dagger directory containing the nltk data.\n\n            Returns:\n                dagger.Directory: A dagger directory containing the nltk data.\n            '
            data_container = self.dagger_client.container().from_('bash:latest')
            for (nltk_data_subfolder, nltk_data_urls) in self.ntlk_data.items():
                full_nltk_data_path = f'{self.nltk_data_path}/{nltk_data_subfolder}'
                for nltk_data_url in nltk_data_urls:
                    zip_file = self.dagger_client.http(nltk_data_url)
                    data_container = data_container.with_file('/tmp/data.zip', zip_file).with_exec(['mkdir', '-p', full_nltk_data_path], skip_entrypoint=True).with_exec(['unzip', '-o', '/tmp/data.zip', '-d', full_nltk_data_path], skip_entrypoint=True).with_exec(['rm', '/tmp/data.zip'], skip_entrypoint=True)
            return data_container.directory(self.nltk_data_path)

        def with_tesseract_and_poppler(container: dagger.Container) -> dagger.Container:
            if False:
                return 10
            '\n            Installs Tesseract-OCR and Poppler-utils in the base image.\n            These tools are necessary for OCR (Optical Character Recognition) processes and working with PDFs, respectively.\n            '
            container = container.with_exec(['sh', '-c', 'apt-get update && apt-get install -y tesseract-ocr=5.3.0-2 poppler-utils=22.12.0-2+b1'], skip_entrypoint=True)
            return container

        def with_file_based_connector_dependencies(container: dagger.Container) -> dagger.Container:
            if False:
                for i in range(10):
                    print('nop')
            '\n            Installs the dependencies for file-based connectors. This includes:\n            - tesseract-ocr\n            - poppler-utils\n            - nltk data\n            '
            container = with_tesseract_and_poppler(container)
            container = container.with_exec(['mkdir', self.nltk_data_path], skip_entrypoint=True).with_directory(self.nltk_data_path, get_nltk_data_dir())
            return container
        return with_file_based_connector_dependencies

    def get_container(self, platform: dagger.Platform) -> dagger.Container:
        if False:
            i = 10
            return i + 15
        'Returns the container used to build the base image.\n        We currently use the python:3.9.18-slim-bookworm image as a base.\n        We set the container system timezone to UTC.\n        We then upgrade pip and install poetry.\n\n        Args:\n            platform (dagger.Platform): The platform this container should be built for.\n\n        Returns:\n            dagger.Container: The container used to build the base image.\n        '
        pip_cache_volume: dagger.CacheVolume = self.dagger_client.cache_volume(AirbytePythonConnectorBaseImage.pip_cache_name)
        return self.get_base_container(platform).with_mounted_cache('/root/.cache/pip', pip_cache_volume).with_exec(['ln', '-snf', '/usr/share/zoneinfo/Etc/UTC', '/etc/localtime']).with_exec(['pip', 'install', '--upgrade', 'pip==23.2.1']).with_env_variable('POETRY_VIRTUALENVS_CREATE', 'false').with_env_variable('POETRY_VIRTUALENVS_IN_PROJECT', 'false').with_env_variable('POETRY_NO_INTERACTION', '1').with_exec(['pip', 'install', 'poetry==1.6.1'], skip_entrypoint=True).with_exec(['sh', '-c', 'apt update && apt-get install -y socat=1.7.4.4-2']).with_(self.install_cdk_system_dependencies())

    async def run_sanity_checks(self, platform: dagger.Platform):
        """Runs sanity checks on the base image container.
        This method is called before image publication.
        Consider it like a pre-flight check before take-off to the remote registry.

        Args:
            platform (dagger.Platform): The platform on which the sanity checks should run.
        """
        container = self.get_container(platform)
        await base_sanity_checks.check_timezone_is_utc(container)
        await base_sanity_checks.check_a_command_is_available_using_version_option(container, 'bash')
        await python_sanity_checks.check_python_version(container, '3.9.18')
        await python_sanity_checks.check_pip_version(container, '23.2.1')
        await python_sanity_checks.check_poetry_version(container, '1.6.1')
        await python_sanity_checks.check_python_image_has_expected_env_vars(container)
        await base_sanity_checks.check_a_command_is_available_using_version_option(container, 'socat', '-V')
        await base_sanity_checks.check_socat_version(container, '1.7.4.4')
        await python_sanity_checks.check_cdk_system_dependencies(container)