import hashlib
import os
import shutil
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import TYPE_CHECKING, Generator, Iterable, List, Optional, TextIO, Tuple, Type, Union
from urllib.parse import urlsplit
import pendulum
from typing_extensions import Self
import prefect
from prefect.utilities.importtools import lazy_import
from prefect.utilities.slugify import slugify

def python_version_minor() -> str:
    if False:
        for i in range(10):
            print('nop')
    return f'{sys.version_info.major}.{sys.version_info.minor}'

def python_version_micro() -> str:
    if False:
        print('Hello World!')
    return f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'

def get_prefect_image_name(prefect_version: str=None, python_version: str=None, flavor: str=None) -> str:
    if False:
        print('Hello World!')
    "\n    Get the Prefect image name matching the current Prefect and Python versions.\n\n    Args:\n        prefect_version: An optional override for the Prefect version.\n        python_version: An optional override for the Python version; must be at the\n            minor level e.g. '3.9'.\n        flavor: An optional alternative image flavor to build, like 'conda'\n    "
    parsed_version = (prefect_version or prefect.__version__).split('+')
    is_prod_build = len(parsed_version) == 1
    prefect_version = parsed_version[0] if is_prod_build else 'sha-' + prefect.__version_info__['full-revisionid'][:7]
    python_version = python_version or python_version_minor()
    tag = slugify(f'{prefect_version}-python{python_version}' + (f'-{flavor}' if flavor else ''), lowercase=False, max_length=128, regex_pattern='[^a-zA-Z0-9_.-]+')
    image = 'prefect' if is_prod_build else 'prefect-dev'
    return f'prefecthq/{image}:{tag}'

@contextmanager
def silence_docker_warnings() -> Generator[None, None, None]:
    if False:
        for i in range(10):
            print('nop')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='distutils Version classes are deprecated.*', category=DeprecationWarning)
        warnings.filterwarnings('ignore', message='The distutils package is deprecated and slated for removal.*', category=DeprecationWarning)
        yield
with silence_docker_warnings():
    if TYPE_CHECKING:
        import docker
        from docker import DockerClient
    else:
        docker = lazy_import('docker')

@contextmanager
def docker_client() -> Generator['DockerClient', None, None]:
    if False:
        return 10
    'Get the environmentally-configured Docker client'
    with silence_docker_warnings():
        client = docker.DockerClient.from_env()
    try:
        yield client
    finally:
        client.close()

class BuildError(Exception):
    """Raised when a Docker build fails"""
IMAGE_LABELS = {'io.prefect.version': prefect.__version__}

@silence_docker_warnings()
def build_image(context: Path, dockerfile: str='Dockerfile', tag: Optional[str]=None, pull: bool=False, platform: str=None, stream_progress_to: Optional[TextIO]=None, **kwargs) -> str:
    if False:
        print('Hello World!')
    'Builds a Docker image, returning the image ID\n\n    Args:\n        context: the root directory for the Docker build context\n        dockerfile: the path to the Dockerfile, relative to the context\n        tag: the tag to give this image\n        pull: True to pull the base image during the build\n        stream_progress_to: an optional stream (like sys.stdout, or an io.TextIO) that\n            will collect the build output as it is reported by Docker\n\n    Returns:\n        The image ID\n    '
    if not context:
        raise ValueError('context required to build an image')
    if not Path(context).exists():
        raise ValueError(f'Context path {context} does not exist')
    kwargs = {key: kwargs[key] for key in kwargs if key not in ['decode', 'labels']}
    image_id = None
    with docker_client() as client:
        events = client.api.build(path=str(context), tag=tag, dockerfile=dockerfile, pull=pull, decode=True, labels=IMAGE_LABELS, platform=platform, **kwargs)
        try:
            for event in events:
                if 'stream' in event:
                    if not stream_progress_to:
                        continue
                    stream_progress_to.write(event['stream'])
                    stream_progress_to.flush()
                elif 'aux' in event:
                    image_id = event['aux']['ID']
                elif 'error' in event:
                    raise BuildError(event['error'])
                elif 'message' in event:
                    raise BuildError(event['message'])
        except docker.errors.APIError as e:
            raise BuildError(e.explanation) from e
    assert image_id, 'The Docker daemon did not return an image ID'
    return image_id

class ImageBuilder:
    """An interface for preparing Docker build contexts and building images"""
    base_directory: Path
    context: Optional[Path]
    platform: Optional[str]
    dockerfile_lines: List[str]

    def __init__(self, base_image: str, base_directory: Path=None, platform: str=None, context: Path=None):
        if False:
            return 10
        'Create an ImageBuilder\n\n        Args:\n            base_image: the base image to use\n            base_directory: the starting point on your host for relative file locations,\n                defaulting to the current directory\n            context: use this path as the build context (if not provided, will create a\n                temporary directory for the context)\n\n        Returns:\n            The image ID\n        '
        self.base_directory = base_directory or context or Path().absolute()
        self.temporary_directory = None
        self.context = context
        self.platform = platform
        self.dockerfile_lines = []
        if self.context:
            dockerfile_path: Path = self.context / 'Dockerfile'
            if dockerfile_path.exists():
                raise ValueError(f'There is already a Dockerfile at {context}')
        self.add_line(f'FROM {base_image}')

    def __enter__(self) -> Self:
        if False:
            while True:
                i = 10
        if self.context and (not self.temporary_directory):
            return self
        self.temporary_directory = TemporaryDirectory()
        self.context = Path(self.temporary_directory.__enter__())
        return self

    def __exit__(self, exc: Type[BaseException], value: BaseException, traceback: TracebackType) -> None:
        if False:
            while True:
                i = 10
        if not self.temporary_directory:
            return
        self.temporary_directory.__exit__(exc, value, traceback)
        self.temporary_directory = None
        self.context = None

    def add_line(self, line: str) -> None:
        if False:
            return 10
        "Add a line to this image's Dockerfile"
        self.add_lines([line])

    def add_lines(self, lines: Iterable[str]) -> None:
        if False:
            return 10
        "Add lines to this image's Dockerfile"
        self.dockerfile_lines.extend(lines)

    def copy(self, source: Union[str, Path], destination: Union[str, PurePosixPath]):
        if False:
            return 10
        'Copy a file to this image'
        if not self.context:
            raise Exception('No context available')
        if not isinstance(destination, PurePosixPath):
            destination = PurePosixPath(destination)
        if not isinstance(source, Path):
            source = Path(source)
        if source.is_absolute():
            source = source.resolve().relative_to(self.base_directory)
        if self.temporary_directory:
            os.makedirs(self.context / source.parent, exist_ok=True)
            if source.is_dir():
                shutil.copytree(self.base_directory / source, self.context / source)
            else:
                shutil.copy2(self.base_directory / source, self.context / source)
        self.add_line(f'COPY {source} {destination}')

    def write_text(self, text: str, destination: Union[str, PurePosixPath]):
        if False:
            i = 10
            return i + 15
        if not self.context:
            raise Exception('No context available')
        if not isinstance(destination, PurePosixPath):
            destination = PurePosixPath(destination)
        source_hash = hashlib.sha256(text.encode()).hexdigest()
        (self.context / f'.{source_hash}').write_text(text)
        self.add_line(f'COPY .{source_hash} {destination}')

    def build(self, pull: bool=False, stream_progress_to: Optional[TextIO]=None) -> str:
        if False:
            while True:
                i = 10
        'Build the Docker image from the current state of the ImageBuilder\n\n        Args:\n            pull: True to pull the base image during the build\n            stream_progress_to: an optional stream (like sys.stdout, or an io.TextIO)\n                that will collect the build output as it is reported by Docker\n\n        Returns:\n            The image ID\n        '
        dockerfile_path: Path = self.context / 'Dockerfile'
        with dockerfile_path.open('w') as dockerfile:
            dockerfile.writelines((line + '\n' for line in self.dockerfile_lines))
        try:
            return build_image(self.context, platform=self.platform, pull=pull, stream_progress_to=stream_progress_to)
        finally:
            os.unlink(dockerfile_path)

    def assert_has_line(self, line: str) -> None:
        if False:
            i = 10
            return i + 15
        'Asserts that the given line is in the Dockerfile'
        all_lines = '\n'.join([f'  {i + 1:>3}: {line}' for (i, line) in enumerate(self.dockerfile_lines)])
        message = f'Expected {line!r} not found in Dockerfile.  Dockerfile:\n{all_lines}'
        assert line in self.dockerfile_lines, message

    def assert_line_absent(self, line: str) -> None:
        if False:
            while True:
                i = 10
        'Asserts that the given line is absent from the Dockerfile'
        if line not in self.dockerfile_lines:
            return
        i = self.dockerfile_lines.index(line)
        surrounding_lines = '\n'.join([f'  {i + 1:>3}: {line}' for (i, line) in enumerate(self.dockerfile_lines[i - 2:i + 2])])
        message = f'Unexpected {line!r} found in Dockerfile at line {i + 1}.  Surrounding lines:\n{surrounding_lines}'
        assert line not in self.dockerfile_lines, message

    def assert_line_before(self, first: str, second: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Asserts that the first line appears before the second line'
        self.assert_has_line(first)
        self.assert_has_line(second)
        first_index = self.dockerfile_lines.index(first)
        second_index = self.dockerfile_lines.index(second)
        surrounding_lines = '\n'.join([f'  {i + 1:>3}: {line}' for (i, line) in enumerate(self.dockerfile_lines[second_index - 2:first_index + 2])])
        message = f'Expected {first!r} to appear before {second!r} in the Dockerfile, but {first!r} was at line {first_index + 1} and {second!r} as at line {second_index + 1}.  Surrounding lines:\n{surrounding_lines}'
        assert first_index < second_index, message

    def assert_line_after(self, second: str, first: str) -> None:
        if False:
            return 10
        'Asserts that the second line appears after the first line'
        self.assert_line_before(first, second)

    def assert_has_file(self, source: Path, container_path: PurePosixPath) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Asserts that the given file or directory will be copied into the container\n        at the given path'
        if source.is_absolute():
            source = source.relative_to(self.base_directory)
        self.assert_has_line(f'COPY {source} {container_path}')

class PushError(Exception):
    """Raised when a Docker image push fails"""

@silence_docker_warnings()
def push_image(image_id: str, registry_url: str, name: str, tag: Optional[str]=None, stream_progress_to: Optional[TextIO]=None) -> str:
    if False:
        while True:
            i = 10
    "Pushes a local image to a Docker registry, returning the registry-qualified tag\n    for that image\n\n    This assumes that the environment's Docker daemon is already authenticated to the\n    given registry, and currently makes no attempt to authenticate.\n\n    Args:\n        image_id (str): a Docker image ID\n        registry_url (str): the URL of a Docker registry\n        name (str): the name of this image\n        tag (str): the tag to give this image (defaults to a short representation of\n            the image's ID)\n        stream_progress_to: an optional stream (like sys.stdout, or an io.TextIO) that\n            will collect the build output as it is reported by Docker\n\n    Returns:\n        A registry-qualified tag, like my-registry.example.com/my-image:abcdefg\n    "
    if not tag:
        tag = slugify(pendulum.now('utc').isoformat())
    (_, registry, _, _, _) = urlsplit(registry_url)
    repository = f'{registry}/{name}'
    with docker_client() as client:
        image: 'docker.Image' = client.images.get(image_id)
        image.tag(repository, tag=tag)
        events = client.api.push(repository, tag=tag, stream=True, decode=True)
        try:
            for event in events:
                if 'status' in event:
                    if not stream_progress_to:
                        continue
                    stream_progress_to.write(event['status'])
                    if 'progress' in event:
                        stream_progress_to.write(' ' + event['progress'])
                    stream_progress_to.write('\n')
                    stream_progress_to.flush()
                elif 'error' in event:
                    raise PushError(event['error'])
        finally:
            client.api.remove_image(f'{repository}:{tag}', noprune=True)
    return f'{repository}:{tag}'

def to_run_command(command: List[str]) -> str:
    if False:
        print('Hello World!')
    '\n    Convert a process-style list of command arguments to a single Dockerfile RUN\n    instruction.\n    '
    if not command:
        return ''
    run_command = f'RUN {command[0]}'
    if len(command) > 1:
        run_command += ' ' + ' '.join([repr(arg) for arg in command[1:]])
    return run_command

def parse_image_tag(name: str) -> Tuple[str, Optional[str]]:
    if False:
        print('Hello World!')
    "\n    Parse Docker Image String\n\n    - If a tag exists, this function parses and returns the image registry and tag,\n      separately as a tuple.\n      - Example 1: 'prefecthq/prefect:latest' -> ('prefecthq/prefect', 'latest')\n      - Example 2: 'hostname.io:5050/folder/subfolder:latest' -> ('hostname.io:5050/folder/subfolder', 'latest')\n    - Supports parsing Docker Image strings that follow Docker Image Specification v1.1.0\n      - Image building tools typically enforce this standard\n\n    Args:\n        name (str): Name of Docker Image\n\n    Return:\n        tuple: image registry, image tag\n    "
    tag = None
    name_parts = name.split('/')
    if len(name_parts) == 1:
        if ':' in name_parts[0]:
            (image_name, tag) = name_parts[0].split(':')
        else:
            image_name = name_parts[0]
    else:
        index_name = name_parts[0]
        image_path = '/'.join(name_parts[1:])
        if ':' in image_path:
            (image_path, tag) = image_path.split(':')
        image_name = f'{index_name}/{image_path}'
    return (image_name, tag)

def format_outlier_version_name(version: str):
    if False:
        for i in range(10):
            print('nop')
    '\n    Formats outlier docker version names to pass `packaging.version.parse` validation\n    - Current cases are simple, but creates stub for more complicated formatting if eventually needed.\n    - Example outlier versions that throw a parsing exception:\n      - "20.10.0-ce" (variant of community edition label)\n      - "20.10.0-ee" (variant of enterprise edition label)\n\n    Args:\n        version (str): raw docker version value\n\n    Returns:\n        str: value that can pass `packaging.version.parse` validation\n    '
    return version.replace('-ce', '').replace('-ee', '')

@contextmanager
def generate_default_dockerfile(context: Optional[Path]=None):
    if False:
        print('Hello World!')
    '\n    Generates a default Dockerfile used for deploying flows. The Dockerfile is written\n    to a temporary file and yielded. The temporary file is removed after the context\n    manager exits.\n\n    Args:\n        - context: The context to use for the Dockerfile. Defaults to\n            the current working directory.\n    '
    if not context:
        context = Path.cwd()
    lines = []
    base_image = get_prefect_image_name()
    lines.append(f'FROM {base_image}')
    dir_name = context.name
    if (context / 'requirements.txt').exists():
        lines.append(f'COPY requirements.txt /opt/prefect/{dir_name}/requirements.txt')
        lines.append(f'RUN python -m pip install -r /opt/prefect/{dir_name}/requirements.txt')
    lines.append(f'COPY . /opt/prefect/{dir_name}/')
    lines.append(f'WORKDIR /opt/prefect/{dir_name}/')
    temp_dockerfile = context / 'Dockerfile'
    if Path(temp_dockerfile).exists():
        raise RuntimeError('Failed to generate Dockerfile. Dockerfile already exists in the current directory.')
    with Path(temp_dockerfile).open('w') as f:
        f.writelines((line + '\n' for line in lines))
    try:
        yield temp_dockerfile
    finally:
        temp_dockerfile.unlink()