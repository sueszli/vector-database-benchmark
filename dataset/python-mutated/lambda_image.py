"""
Generates a Docker Image to be used for invoking a function locally
"""
import hashlib
import logging
import os
import platform
import re
import sys
import uuid
from enum import Enum
from pathlib import Path
from typing import Optional
import docker
from samcli.commands.local.cli_common.user_exceptions import DockerDistributionAPIError, ImageBuildException
from samcli.commands.local.lib.exceptions import InvalidIntermediateImageError
from samcli.lib.constants import DOCKER_MIN_API_VERSION
from samcli.lib.utils.architecture import has_runtime_multi_arch_image
from samcli.lib.utils.packagetype import IMAGE, ZIP
from samcli.lib.utils.stream_writer import StreamWriter
from samcli.lib.utils.tar import create_tarball
from samcli.local.docker.utils import get_docker_platform, get_rapid_name
LOG = logging.getLogger(__name__)
RAPID_IMAGE_TAG_PREFIX = 'rapid'

class Runtime(Enum):
    nodejs12x = 'nodejs12.x'
    nodejs14x = 'nodejs14.x'
    nodejs16x = 'nodejs16.x'
    nodejs18x = 'nodejs18.x'
    nodejs20x = 'nodejs20.x'
    python37 = 'python3.7'
    python38 = 'python3.8'
    python39 = 'python3.9'
    python310 = 'python3.10'
    python311 = 'python3.11'
    ruby27 = 'ruby2.7'
    ruby32 = 'ruby3.2'
    java8 = 'java8'
    java8al2 = 'java8.al2'
    java11 = 'java11'
    java17 = 'java17'
    go1x = 'go1.x'
    dotnet6 = 'dotnet6'
    provided = 'provided'
    providedal2 = 'provided.al2'
    providedal2023 = 'provided.al2023'

    @classmethod
    def has_value(cls, value):
        if False:
            while True:
                i = 10
        '\n        Checks if the enum has this value\n\n        :param string value: Value to check\n        :return bool: True, if enum has the value\n        '
        return any((value == item.value for item in cls))

    @classmethod
    def get_image_name_tag(cls, runtime: str, architecture: str) -> str:
        if False:
            while True:
                i = 10
        "\n        Returns the image name and tag for a particular runtime\n\n        Parameters\n        ----------\n        runtime : str\n            AWS Lambda runtime\n        architecture : str\n            Architecture for the runtime\n\n        Returns\n        -------\n        str\n            Image name and tag for the runtime's base image, like `python:3.7` or `provided:al2`\n        "
        runtime_image_tag = ''
        if runtime == cls.provided.value:
            runtime_image_tag = 'provided:alami'
        elif runtime.startswith('provided'):
            runtime_image_tag = runtime.replace('.', ':')
        elif runtime.startswith('dotnet'):
            runtime_image_tag = runtime.replace('dotnet', 'dotnet:')
        else:
            runtime_image_tag = re.sub('^([a-z]+)([0-9][a-z0-9\\.]*)$', '\\1:\\2', runtime)
            runtime_image_tag = runtime_image_tag.replace('.x', '')
        if has_runtime_multi_arch_image(runtime):
            runtime_image_tag = f'{runtime_image_tag}-{architecture}'
        return runtime_image_tag

class LambdaImage:
    _LAYERS_DIR = '/opt'
    _INVOKE_REPO_PREFIX = 'public.ecr.aws/lambda'
    _SAM_INVOKE_REPO_PREFIX = 'public.ecr.aws/sam/emulation'
    _SAM_CLI_REPO_NAME = 'samcli/lambda'
    _RAPID_SOURCE_PATH = Path(__file__).parent.joinpath('..', 'rapid').resolve()

    def __init__(self, layer_downloader, skip_pull_image, force_image_build, docker_client=None, invoke_images=None):
        if False:
            return 10
        '\n\n        Parameters\n        ----------\n        layer_downloader samcli.local.layers.layer_downloader.LayerDownloader\n            LayerDownloader to download layers locally\n        skip_pull_image bool\n            True if the image should not be pulled from DockerHub\n        force_image_build bool\n            True to download the layer and rebuild the image even if it exists already on the system\n        docker_client docker.DockerClient\n            Optional docker client object\n        '
        self.layer_downloader = layer_downloader
        self.skip_pull_image = skip_pull_image
        self.force_image_build = force_image_build
        self.docker_client = docker_client or docker.from_env(version=DOCKER_MIN_API_VERSION)
        self.invoke_images = invoke_images

    def build(self, runtime, packagetype, image, layers, architecture, stream=None, function_name=None):
        if False:
            return 10
        '\n        Build the image if one is not already on the system that matches the runtime and layers\n\n        Parameters\n        ----------\n        runtime : str\n            Name of the Lambda runtime\n        packagetype : str\n            Packagetype for the Lambda\n        image : str\n            Pre-defined invocation image.\n        layers : list(samcli.commands.local.lib.provider.Layer)\n            List of layers\n        architecture : str\n            Architecture type either x86_64 or arm64 on AWS lambda\n        stream : io.RawIOBase\n            stream to write\n        function_name : str\n            The name of the function that the image is building for\n\n        Returns\n        -------\n        str\n            The image to be used (REPOSITORY:TAG)\n        '
        base_image = None
        tag_prefix = ''
        if packagetype == IMAGE:
            base_image = image
        elif packagetype == ZIP:
            runtime_image_tag = Runtime.get_image_name_tag(runtime, architecture)
            if self.invoke_images:
                base_image = self.invoke_images.get(function_name, self.invoke_images.get(None))
            if not base_image:
                runtime_only_number = re.split('[:-]', runtime_image_tag)[1]
                tag_prefix = f'{runtime_only_number}-'
                base_image = f'{self._INVOKE_REPO_PREFIX}/{runtime_image_tag}'
                if platform.system().lower() == 'windows' and runtime in [Runtime.go1x.value, Runtime.java8.value]:
                    LOG.info('Falling back to a previous version of the emulation image')
                    base_image = f'{base_image}.2023.08.02.10'
        if not base_image:
            raise InvalidIntermediateImageError(f'Invalid PackageType, PackageType needs to be one of [{ZIP}, {IMAGE}]')
        if image:
            self.skip_pull_image = True
        image_repo = base_image.split(':')[0].replace('@', '')
        rapid_image = f'{image_repo}:{tag_prefix}{RAPID_IMAGE_TAG_PREFIX}-{architecture}'
        downloaded_layers = []
        if layers and packagetype == ZIP:
            downloaded_layers = self.layer_downloader.download_all(layers, self.force_image_build)
            docker_image_version = self._generate_docker_image_version(downloaded_layers, runtime_image_tag)
            rapid_image = f'{self._SAM_CLI_REPO_NAME}-{docker_image_version}'
        image_not_found = False
        try:
            self.docker_client.images.get(rapid_image)
            self._check_base_image_is_current(base_image)
        except docker.errors.ImageNotFound:
            LOG.info('Local image was not found.')
            image_not_found = True
        except docker.errors.APIError as e:
            if e.__class__ is docker.errors.NotFound:
                LOG.warning('Unknown 404 - Unable to check if base image is current.\n\nPossible incompatible Docker engine clone employed. Consider `--skip-pull-image` for improved speed, the tradeoff being not running the latest image.')
                image_not_found = True
            else:
                raise DockerDistributionAPIError(str(e)) from e
        if image_not_found and rapid_image == f'{image_repo}:{tag_prefix}{RAPID_IMAGE_TAG_PREFIX}-{architecture}':
            if tag_prefix:
                self._remove_rapid_images(f'{self._SAM_INVOKE_REPO_PREFIX}-{runtime}')
            else:
                self._remove_rapid_images(image_repo)
        if self.force_image_build or image_not_found or any((layer.is_defined_within_template for layer in downloaded_layers)) or (not runtime):
            stream_writer = stream or StreamWriter(sys.stderr)
            stream_writer.write_str('Building image...')
            stream_writer.flush()
            self._build_image(image if image else base_image, rapid_image, downloaded_layers, architecture, stream=stream_writer)
        return rapid_image

    def get_config(self, image_tag):
        if False:
            print('Hello World!')
        config = {}
        try:
            image = self.docker_client.images.get(image_tag)
            return image.attrs.get('Config')
        except docker.errors.ImageNotFound:
            return config

    @staticmethod
    def _generate_docker_image_version(layers, runtime_image_tag):
        if False:
            return 10
        '\n        Generate the Docker TAG that will be used to create the image\n\n        Parameters\n        ----------\n        layers list(samcli.commands.local.lib.provider.Layer)\n            List of the layers\n\n        runtime_image_tag str\n            Runtime version format to generate image name and tag (including architecture, e.g. "python:3.7-x86_64")\n\n        Returns\n        -------\n        str\n            String representing the TAG to be attached to the image\n        '
        return runtime_image_tag + '-' + hashlib.sha256('-'.join([layer.name for layer in layers]).encode('utf-8')).hexdigest()[0:25]

    def _build_image(self, base_image, docker_tag, layers, architecture, stream=None):
        if False:
            i = 10
            return i + 15
        '\n        Builds the image\n\n        Parameters\n        ----------\n        base_image str\n            Base Image to use for the new image\n        docker_tag str\n            Docker tag (REPOSITORY:TAG) to use when building the image\n        layers list(samcli.commands.local.lib.provider.Layer)\n            List of Layers to be use to mount in the image\n        architecture str\n            Architecture, either x86_64 or arm64\n        stream samcli.lib.utils.stream_writer.StreamWriter\n            Stream to write the build output\n\n        Raises\n        ------\n        samcli.commands.local.cli_common.user_exceptions.ImageBuildException\n            When docker fails to build the image\n        '
        dockerfile_content = self._generate_dockerfile(base_image, layers, architecture)
        dockerfile_name = 'dockerfile_' + str(uuid.uuid4())
        full_dockerfile_path = Path(self.layer_downloader.layer_cache, dockerfile_name)
        stream_writer = stream or StreamWriter(sys.stderr)
        try:
            with open(str(full_dockerfile_path), 'w') as dockerfile:
                dockerfile.write(dockerfile_content)
            tar_paths = {str(full_dockerfile_path): 'Dockerfile', self._RAPID_SOURCE_PATH: '/' + get_rapid_name(architecture)}
            for layer in layers:
                tar_paths[layer.codeuri] = '/' + layer.name

            def set_item_permission(tar_info):
                if False:
                    return 10
                tar_info.mode = 320
                return tar_info
            tar_filter = set_item_permission if platform.system().lower() == 'windows' else None
            with create_tarball(tar_paths, tar_filter=tar_filter) as tarballfile:
                try:
                    resp_stream = self.docker_client.api.build(fileobj=tarballfile, custom_context=True, rm=True, tag=docker_tag, pull=not self.skip_pull_image, decode=True, platform=get_docker_platform(architecture))
                    for log in resp_stream:
                        stream_writer.write_str('.')
                        stream_writer.flush()
                        if 'error' in log:
                            stream_writer.write_str(os.linesep)
                            LOG.exception('Failed to build Docker Image')
                            raise ImageBuildException('Error building docker image: {}'.format(log['error']))
                    stream_writer.write_str(os.linesep)
                except (docker.errors.BuildError, docker.errors.APIError) as ex:
                    stream_writer.write_str(os.linesep)
                    LOG.exception('Failed to build Docker Image')
                    raise ImageBuildException('Building Image failed.') from ex
        finally:
            if full_dockerfile_path.exists():
                full_dockerfile_path.unlink()

    @staticmethod
    def _generate_dockerfile(base_image, layers, architecture):
        if False:
            for i in range(10):
                print('nop')
        '\n        FROM public.ecr.aws/lambda/python:3.9-x86_64\n\n        ADD aws-lambda-rie /var/rapid\n\n        ADD layer1 /opt\n        ADD layer2 /opt\n\n        Parameters\n        ----------\n        base_image : str\n            Base Image to use for the new image\n        layers : list\n            List of Layers to be use to mount in the image\n        architecture : str\n            Architecture type either x86_64 or arm64 on AWS lambda\n\n        Returns\n        -------\n        str\n            String representing the Dockerfile contents for the image\n        '
        rie_name = get_rapid_name(architecture)
        rie_path = '/var/rapid/'
        dockerfile_content = f'FROM {base_image}\n' + f'ADD {rie_name} {rie_path}\n' + f'RUN mv {rie_path}{rie_name} {rie_path}aws-lambda-rie && chmod +x {rie_path}aws-lambda-rie\n'
        for layer in layers:
            dockerfile_content = dockerfile_content + f'ADD {layer.name} {LambdaImage._LAYERS_DIR}\n'
        return dockerfile_content

    def _remove_rapid_images(self, repo: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Remove all rapid images for given repo\n\n        Parameters\n        ----------\n        repo string\n            Repo for which rapid images will be removed\n        '
        LOG.info('Removing rapid images for repo %s', repo)
        try:
            for image in self.docker_client.images.list(name=repo):
                for tag in image.tags:
                    if self.is_rapid_image(tag) and (not self.is_rapid_image_current(tag)):
                        try:
                            self.docker_client.images.remove(image.id)
                        except docker.errors.APIError as ex:
                            LOG.warning('Failed to remove rapid image with ID: %s', image.id, exc_info=ex)
                        break
        except docker.errors.APIError as ex:
            LOG.warning('Failed getting images from repo %s', repo, exc_info=ex)

    @staticmethod
    def is_rapid_image(image_name: str) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Is the image tagged as a RAPID clone?\n\n        Parameters\n        ----------\n        image_name : str\n            Name of the image\n\n        Returns\n        -------\n        bool\n            True if the image tag starts with the rapid prefix or contains it in between. False, otherwise\n        '
        try:
            tag = image_name.split(':')[1]
            return tag.startswith(f'{RAPID_IMAGE_TAG_PREFIX}-') or f'-{RAPID_IMAGE_TAG_PREFIX}-' in tag
        except (IndexError, AttributeError):
            return False

    @staticmethod
    def is_rapid_image_current(image_name: str) -> bool:
        if False:
            print('Hello World!')
        "\n        Verify if an image has the latest format.\n        The current format doesn't include the SAM version and has the RAPID prefix between dashes.\n\n        Parameters\n        ----------\n        image_name : str\n            name the image\n\n        Returns\n        -------\n        bool\n            return True if it is current and vice versa\n        "
        return f'-{RAPID_IMAGE_TAG_PREFIX}-' in image_name

    def _check_base_image_is_current(self, image_name: str) -> None:
        if False:
            return 10
        '\n        Check if the existing base image is up-to-date and update modifier parameters\n        (skip_pull_image, force_image_build) accordingly, printing an informative\n        message depending on the case.\n\n        Parameters\n        ----------\n        image_name : str\n            Base image name to check\n        '
        if self.skip_pull_image or self.force_image_build:
            return
        if self.is_base_image_current(image_name):
            self.skip_pull_image = True
            LOG.info('Local image is up-to-date')
        else:
            self.force_image_build = True
            LOG.info('Local image is out of date and will be updated to the latest runtime. To skip this, pass in the parameter --skip-pull-image')

    def is_base_image_current(self, image_name: str) -> bool:
        if False:
            print('Hello World!')
        '\n        Return True if the base image is up-to-date with the remote environment by comparing the image digests\n\n        Parameters\n        ----------\n        image_name : str\n            Base image name to check\n\n        Returns\n        -------\n        bool\n            True if local image digest is the same as the remote image digest\n        '
        return self.get_local_image_digest(image_name) == self.get_remote_image_digest(image_name)

    def get_remote_image_digest(self, image_name: str) -> Optional[str]:
        if False:
            while True:
                i = 10
        '\n        Get the digest of the remote version of an image\n\n        Parameters\n        ----------\n        image_name : str\n            Name of the image to get the digest\n\n        Returns\n        -------\n        str\n            Image digest, including `sha256:` prefix\n        '
        remote_info = self.docker_client.images.get_registry_data(image_name)
        digest: Optional[str] = remote_info.attrs.get('Descriptor', {}).get('digest')
        return digest

    def get_local_image_digest(self, image_name: str) -> Optional[str]:
        if False:
            while True:
                i = 10
        '\n        Get the digest of the local version of an image\n\n        Parameters\n        ----------\n        image_name : str\n            Name of the image to get the digest\n\n        Returns\n        -------\n        str\n            Image digest, including `sha256:` prefix\n        '
        image_info = self.docker_client.images.get(image_name)
        try:
            full_digest: str = image_info.attrs.get('RepoDigests', [None])[0]
            return full_digest.split('@')[1]
        except (AttributeError, IndexError):
            return None