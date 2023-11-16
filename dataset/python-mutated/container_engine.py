from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Optional
from rich.progress import track
from .proc import CalledProcessError
from .proc import CompletedProcess
from .proc import run_command

class ContainerEngineError(CalledProcessError):
    pass

class ContainerEngine(ABC):

    @abstractmethod
    def is_available(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @abstractmethod
    def pull(self, images: List[str], dryrun: bool, stream_output: Optional[dict]) -> List[CompletedProcess]:
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    @abstractmethod
    def save(self, images: List[str], archive_path: str, dryrun: bool) -> CompletedProcess:
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def check_returncode(self, result: CompletedProcess) -> None:
        if False:
            return 10
        try:
            result.check_returncode()
        except CalledProcessError as e:
            raise ContainerEngineError(e.returncode, e.cmd) from e

class Podman(ContainerEngine):

    def is_available(self) -> bool:
        if False:
            print('Hello World!')
        result = run_command('podman version')
        return result.returncode == 0

    def pull(self, images: List[str], dryrun: bool=False, stream_output: Optional[dict]=None) -> List[CompletedProcess]:
        if False:
            while True:
                i = 10
        results = []
        for image in track(images, description=''):
            command = f'podman pull {image} --quiet'
            result = run_command(command, stream_output=stream_output, dryrun=dryrun)
            self.check_returncode(result)
            results.append(result)
        return results

    def save(self, images: List[str], archive_path: str, dryrun: bool=False) -> CompletedProcess:
        if False:
            return 10
        images_str = ' '.join(images)
        command = f'podman save -m -o {archive_path} {images_str}'
        result = run_command(command, dryrun=dryrun)
        self.check_returncode(result)
        return result

class Docker(ContainerEngine):

    def is_available(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        result = run_command('docker version')
        return result.returncode == 0

    def pull(self, images: List[str], dryrun: bool=False, stream_output: Optional[dict]=None) -> List[CompletedProcess]:
        if False:
            i = 10
            return i + 15
        results = []
        for image in track(images, description=''):
            command = f'docker pull {image} --quiet'
            result = run_command(command, stream_output=stream_output, dryrun=dryrun)
            self.check_returncode(result)
            results.append(result)
        return results

    def save(self, images: List[str], archive_path: str, dryrun: bool=False) -> CompletedProcess:
        if False:
            for i in range(10):
                print('nop')
        images_str = ' '.join(images)
        command = f'docker save -o {archive_path} {images_str}'
        result = run_command(command, dryrun=dryrun)
        self.check_returncode(result)
        return result