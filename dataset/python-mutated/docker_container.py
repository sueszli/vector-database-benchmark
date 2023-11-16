import os
from typing import List
from ci.ray_ci.container import Container
from ci.ray_ci.builder_container import DEFAULT_ARCHITECTURE
PLATFORM = ['cpu', 'cu11.5.2', 'cu11.6.2', 'cu11.7.1', 'cu11.8.0', 'cu12.1.1']
GPU_PLATFORM = 'cu11.8.0'
DEFAULT_PYTHON_VERSION = '3.8'

class DockerContainer(Container):
    """
    Container for building and publishing ray docker images
    """

    def __init__(self, python_version: str, platform: str, image_type: str, architecture: str=DEFAULT_ARCHITECTURE) -> None:
        if False:
            i = 10
            return i + 15
        assert 'RAYCI_CHECKOUT_DIR' in os.environ, 'RAYCI_CHECKOUT_DIR not set'
        rayci_checkout_dir = os.environ['RAYCI_CHECKOUT_DIR']
        self.python_version = python_version
        self.platform = platform
        self.image_type = image_type
        self.architecture = architecture
        super().__init__('forge' if architecture == 'x86_64' else 'forge-aarch64', volumes=[f'{rayci_checkout_dir}:/rayci', '/var/run/docker.sock:/var/run/docker.sock'])

    def _get_image_version_tags(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        branch = os.environ.get('BUILDKITE_BRANCH')
        sha_tag = os.environ['BUILDKITE_COMMIT'][:6]
        pr = os.environ.get('BUILDKITE_PULL_REQUEST', 'false')
        if branch == 'master':
            return [sha_tag, 'nightly']
        if branch and branch.startswith('releases/'):
            release_name = branch[len('releases/'):]
            return [f'{release_name}.{sha_tag}']
        if pr != 'false':
            return [f'pr-{pr}.{sha_tag}']
        return [sha_tag]

    def _get_canonical_tag(self) -> str:
        if False:
            return 10
        return self._get_image_tags()[0]

    def get_python_version_tag(self) -> str:
        if False:
            print('Hello World!')
        return f"-py{self.python_version.replace('.', '')}"

    def get_platform_tag(self) -> str:
        if False:
            print('Hello World!')
        if self.platform == 'cpu':
            return '-cpu'
        versions = self.platform.split('.')
        return f'-{versions[0]}{versions[1]}'

    def _get_image_tags(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        versions = self._get_image_version_tags()
        platforms = [self.get_platform_tag()]
        if self.platform == 'cpu' and self.image_type == 'ray':
            platforms.append('')
        elif self.platform == GPU_PLATFORM:
            platforms.append('-gpu')
            if self.image_type == 'ray-ml':
                platforms.append('')
        py_versions = [self.get_python_version_tag()]
        if self.python_version == DEFAULT_PYTHON_VERSION:
            py_versions.append('')
        tags = []
        for version in versions:
            for platform in platforms:
                for py_version in py_versions:
                    if self.architecture == DEFAULT_ARCHITECTURE:
                        tag = f'{version}{py_version}{platform}'
                    else:
                        tag = f'{version}{py_version}{platform}-{self.architecture}'
                    tags.append(tag)
        return tags