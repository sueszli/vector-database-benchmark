import inspect
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from typing_extensions import Self
from lightning.app.utilities.app_helpers import Logger
from lightning.app.utilities.packaging.cloud_compute import CloudCompute
if TYPE_CHECKING:
    from lightning.app.core.work import LightningWork
logger = Logger(__name__)

def load_requirements(path_dir: str, file_name: str='base.txt', comment_char: str='#', unfreeze: bool=True) -> List[str]:
    if False:
        while True:
            i = 10
    'Load requirements from a file.'
    path = os.path.join(path_dir, file_name)
    if not os.path.isfile(path):
        return []
    with open(path) as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        comment = ''
        if comment_char in ln:
            comment = ln[ln.index(comment_char):]
            ln = ln[:ln.index(comment_char)]
        req = ln.strip()
        if not req or req.startswith('http') or '@http' in req:
            continue
        if unfreeze and '<' in req and ('strict' not in comment):
            req = re.sub(',? *<=? *[\\d\\.\\*]+', '', req).strip()
        reqs.append(req)
    return reqs

@dataclass
class _Dockerfile:
    path: str
    data: List[str]

@dataclass
class BuildConfig:
    """The Build Configuration describes how the environment a LightningWork runs in should be set up.

    Arguments:
        requirements: List of requirements or list of paths to requirement files. If not passed, they will be
            automatically extracted from a `requirements.txt` if it exists.
        dockerfile: The path to a dockerfile to be used to build your container.
            You need to add those lines to ensure your container works in the cloud.

            .. warning:: This feature isn't supported yet, but coming soon.

            Example::

                WORKDIR /gridai/project
                COPY . .
        image: The base image that the work runs on. This should be a publicly accessible image from a registry that
            doesn't enforce rate limits (such as DockerHub) to pull this image, otherwise your application will not
            start.

    """
    requirements: List[str] = field(default_factory=list)
    dockerfile: Optional[Union[str, Path, _Dockerfile]] = None
    image: Optional[str] = None

    def __post_init__(self) -> None:
        if False:
            print('Hello World!')
        current_frame = inspect.currentframe()
        co_filename = current_frame.f_back.f_back.f_code.co_filename
        self._call_dir = os.path.dirname(co_filename)
        self._prepare_requirements()
        self._prepare_dockerfile()

    def build_commands(self) -> List[str]:
        if False:
            while True:
                i = 10
        'Override to run some commands before your requirements are installed.\n\n        .. note:: If you provide your own dockerfile, this would be ignored.\n\n        Example:\n\n            from dataclasses import dataclass\n            from lightning.app import BuildConfig\n\n            @dataclass\n            class MyOwnBuildConfig(BuildConfig):\n\n                def build_commands(self):\n                    return ["apt-get install libsparsehash-dev"]\n\n            BuildConfig(requirements=["git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0"])\n\n        '
        return []

    def on_work_init(self, work: 'LightningWork', cloud_compute: Optional['CloudCompute']=None) -> None:
        if False:
            i = 10
            return i + 15
        'Override with your own logic to load the requirements or dockerfile.'
        found_requirements = self._find_requirements(work)
        if self.requirements:
            if found_requirements and self.requirements != found_requirements:
                logger.info(f"A 'requirements.txt' exists with {found_requirements} but {self.requirements} was passed to the `{type(self).__name__}` in {work.name!r}. The `requirements.txt` file will be ignored.")
        else:
            self.requirements = found_requirements
        self._prepare_requirements()
        found_dockerfile = self._find_dockerfile(work)
        if self.dockerfile:
            if found_dockerfile and self.dockerfile != found_dockerfile:
                logger.info(f'A Dockerfile exists at {found_dockerfile!r} but {self.dockerfile!r} was passed to the `{type(self).__name__}` in {work.name!r}. {found_dockerfile!r}` will be ignored.')
        else:
            self.dockerfile = found_dockerfile
        self._prepare_dockerfile()

    def _find_requirements(self, work: 'LightningWork', filename: str='requirements.txt') -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        file = _get_work_file(work)
        if file is None:
            return []
        dirname = os.path.dirname(file)
        try:
            requirements = load_requirements(dirname, filename)
        except NotADirectoryError:
            return []
        return [r for r in requirements if r != 'lightning']

    def _find_dockerfile(self, work: 'LightningWork', filename: str='Dockerfile') -> Optional[str]:
        if False:
            return 10
        file = _get_work_file(work)
        if file is None:
            return None
        dirname = os.path.dirname(file)
        dockerfile = os.path.join(dirname, filename)
        if os.path.isfile(dockerfile):
            return dockerfile
        return None

    def _prepare_requirements(self) -> None:
        if False:
            return 10
        requirements = []
        for req in self.requirements:
            path = os.path.join(self._call_dir, req)
            if os.path.isfile(path):
                try:
                    new_requirements = load_requirements(self._call_dir, req)
                except NotADirectoryError:
                    continue
                requirements.extend(new_requirements)
            else:
                requirements.append(req)
        self.requirements = requirements

    def _prepare_dockerfile(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.dockerfile, (str, Path)):
            path = os.path.join(self._call_dir, self.dockerfile)
            if os.path.exists(path):
                with open(path) as f:
                    self.dockerfile = _Dockerfile(path, f.readlines())

    def to_dict(self) -> Dict:
        if False:
            while True:
                i = 10
        return {'__build_config__': asdict(self)}

    @classmethod
    def from_dict(cls, d: Dict) -> Self:
        if False:
            i = 10
            return i + 15
        return cls(**d['__build_config__'])

def _get_work_file(work: 'LightningWork') -> Optional[str]:
    if False:
        while True:
            i = 10
    cls = work.__class__
    try:
        return inspect.getfile(cls)
    except TypeError:
        logger.debug(f"The {cls.__name__} file couldn't be found.")
        return None