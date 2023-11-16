from __future__ import annotations
import contextlib
import os
import re
import subprocess
import sys
import sysconfig
from pathlib import Path
from subprocess import CalledProcessError
from typing import TYPE_CHECKING
from typing import Any
from virtualenv.seed.wheels.embed import get_embed_wheel
from poetry.utils.env.exceptions import EnvCommandError
from poetry.utils.env.site_packages import SitePackages
from poetry.utils.helpers import get_real_windows_path
if TYPE_CHECKING:
    from packaging.tags import Tag
    from poetry.core.constraints.version import Version
    from poetry.core.version.markers import BaseMarker
    from virtualenv.seed.wheels.util import Wheel
    from poetry.utils.env.generic_env import GenericEnv

class Env:
    """
    An abstract Python environment.
    """

    def __init__(self, path: Path, base: Path | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._is_windows = sys.platform == 'win32'
        self._is_mingw = sysconfig.get_platform().startswith('mingw')
        self._is_conda = bool(os.environ.get('CONDA_DEFAULT_ENV'))
        if self._is_windows:
            path = get_real_windows_path(path)
            base = get_real_windows_path(base) if base else None
        bin_dir = 'bin' if not self._is_windows or self._is_mingw else 'Scripts'
        self._path = path
        self._bin_dir = self._path / bin_dir
        self._executable = 'python'
        self._pip_executable = 'pip'
        self.find_executables()
        self._base = base or path
        self._marker_env: dict[str, Any] | None = None
        self._pip_version: Version | None = None
        self._site_packages: SitePackages | None = None
        self._paths: dict[str, str] | None = None
        self._supported_tags: list[Tag] | None = None
        self._purelib: Path | None = None
        self._platlib: Path | None = None
        self._script_dirs: list[Path] | None = None
        self._embedded_pip_path: Path | None = None

    @property
    def path(self) -> Path:
        if False:
            i = 10
            return i + 15
        return self._path

    @property
    def base(self) -> Path:
        if False:
            for i in range(10):
                print('nop')
        return self._base

    @property
    def version_info(self) -> tuple[int, int, int, str, int]:
        if False:
            return 10
        version_info: tuple[int, int, int, str, int] = self.marker_env['version_info']
        return version_info

    @property
    def python_implementation(self) -> str:
        if False:
            print('Hello World!')
        implementation: str = self.marker_env['platform_python_implementation']
        return implementation

    @property
    def python(self) -> Path:
        if False:
            i = 10
            return i + 15
        '\n        Path to current python executable\n        '
        return Path(self._bin(self._executable))

    @property
    def marker_env(self) -> dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        if self._marker_env is None:
            self._marker_env = self.get_marker_env()
        return self._marker_env

    @property
    def parent_env(self) -> GenericEnv:
        if False:
            print('Hello World!')
        from poetry.utils.env.generic_env import GenericEnv
        return GenericEnv(self.base, child_env=self)

    def _find_python_executable(self) -> None:
        if False:
            print('Hello World!')
        bin_dir = self._bin_dir
        if self._is_windows and self._is_conda:
            bin_dir = self._path
        python_executables = sorted((p.name for p in bin_dir.glob('python*') if re.match('python(?:\\d+(?:\\.\\d+)?)?(?:\\.exe)?$', p.name)))
        if python_executables:
            executable = python_executables[0]
            if executable.endswith('.exe'):
                executable = executable[:-4]
            self._executable = executable

    def _find_pip_executable(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pip_executables = sorted((p.name for p in self._bin_dir.glob('pip*') if re.match('pip(?:\\d+(?:\\.\\d+)?)?(?:\\.exe)?$', p.name)))
        if pip_executables:
            pip_executable = pip_executables[0]
            if pip_executable.endswith('.exe'):
                pip_executable = pip_executable[:-4]
            self._pip_executable = pip_executable

    def find_executables(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._find_python_executable()
        self._find_pip_executable()

    def get_embedded_wheel(self, distribution: str) -> Path:
        if False:
            i = 10
            return i + 15
        wheel: Wheel = get_embed_wheel(distribution, f'{self.version_info[0]}.{self.version_info[1]}')
        path: Path = wheel.path
        return path

    @property
    def pip_embedded(self) -> Path:
        if False:
            for i in range(10):
                print('nop')
        if self._embedded_pip_path is None:
            self._embedded_pip_path = self.get_embedded_wheel('pip') / 'pip'
        return self._embedded_pip_path

    @property
    def pip(self) -> Path:
        if False:
            while True:
                i = 10
        '\n        Path to current pip executable\n        '
        path = Path(self._bin(self._pip_executable))
        if not path.exists():
            return self.pip_embedded
        return path

    @property
    def platform(self) -> str:
        if False:
            i = 10
            return i + 15
        return sys.platform

    @property
    def os(self) -> str:
        if False:
            print('Hello World!')
        return os.name

    @property
    def pip_version(self) -> Version:
        if False:
            return 10
        if self._pip_version is None:
            self._pip_version = self.get_pip_version()
        return self._pip_version

    @property
    def site_packages(self) -> SitePackages:
        if False:
            for i in range(10):
                print('nop')
        if self._site_packages is None:
            fallbacks = [self.usersite] if self.usersite else []
            self._site_packages = SitePackages(self.purelib, self.platlib, fallbacks, skip_write_checks=not fallbacks)
        return self._site_packages

    @property
    def usersite(self) -> Path | None:
        if False:
            for i in range(10):
                print('nop')
        if 'usersite' in self.paths:
            return Path(self.paths['usersite'])
        return None

    @property
    def userbase(self) -> Path | None:
        if False:
            while True:
                i = 10
        if 'userbase' in self.paths:
            return Path(self.paths['userbase'])
        return None

    @property
    def purelib(self) -> Path:
        if False:
            return 10
        if self._purelib is None:
            self._purelib = Path(self.paths['purelib'])
        return self._purelib

    @property
    def platlib(self) -> Path:
        if False:
            while True:
                i = 10
        if self._platlib is None:
            if 'platlib' in self.paths:
                self._platlib = Path(self.paths['platlib'])
            else:
                self._platlib = self.purelib
        return self._platlib

    def _get_lib_dirs(self) -> list[Path]:
        if False:
            i = 10
            return i + 15
        return [self.purelib, self.platlib]

    def is_path_relative_to_lib(self, path: Path) -> bool:
        if False:
            print('Hello World!')
        for lib_path in self._get_lib_dirs():
            with contextlib.suppress(ValueError):
                path.relative_to(lib_path)
                return True
        return False

    @property
    def sys_path(self) -> list[str]:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @property
    def paths(self) -> dict[str, str]:
        if False:
            i = 10
            return i + 15
        if self._paths is None:
            self._paths = self.get_paths()
            if self.is_venv():
                self._paths['include'] = str(self.path.joinpath('include', 'site', f'python{self.version_info[0]}.{self.version_info[1]}'))
        return self._paths

    @property
    def supported_tags(self) -> list[Tag]:
        if False:
            print('Hello World!')
        if self._supported_tags is None:
            self._supported_tags = self.get_supported_tags()
        return self._supported_tags

    @classmethod
    def get_base_prefix(cls) -> Path:
        if False:
            for i in range(10):
                print('nop')
        real_prefix = getattr(sys, 'real_prefix', None)
        if real_prefix is not None:
            return Path(real_prefix)
        base_prefix = getattr(sys, 'base_prefix', None)
        if base_prefix is not None:
            return Path(base_prefix)
        return Path(sys.prefix)

    def get_version_info(self) -> tuple[Any, ...]:
        if False:
            return 10
        raise NotImplementedError()

    def get_python_implementation(self) -> str:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def get_marker_env(self) -> dict[str, Any]:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def get_pip_command(self, embedded: bool=False) -> list[str]:
        if False:
            print('Hello World!')
        if embedded or not Path(self._bin(self._pip_executable)).exists():
            return [str(self.python), str(self.pip_embedded)]
        return [str(self.python), '-m', 'pip']

    def get_supported_tags(self) -> list[Tag]:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def get_pip_version(self) -> Version:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def get_paths(self) -> dict[str, str]:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def is_valid_for_marker(self, marker: BaseMarker) -> bool:
        if False:
            while True:
                i = 10
        valid: bool = marker.validate(self.marker_env)
        return valid

    def is_sane(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Checks whether the current environment is sane or not.\n        '
        return True

    def get_command_from_bin(self, bin: str) -> list[str]:
        if False:
            return 10
        if bin == 'pip':
            return self.get_pip_command()
        return [self._bin(bin)]

    def run(self, bin: str, *args: str, **kwargs: Any) -> str:
        if False:
            while True:
                i = 10
        cmd = self.get_command_from_bin(bin) + list(args)
        return self._run(cmd, **kwargs)

    def run_pip(self, *args: str, **kwargs: Any) -> str:
        if False:
            for i in range(10):
                print('nop')
        pip = self.get_pip_command()
        cmd = pip + list(args)
        return self._run(cmd, **kwargs)

    def run_python_script(self, content: str, **kwargs: Any) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.run(self._executable, '-I', '-W', 'ignore', '-c', content, stderr=subprocess.PIPE, **kwargs)

    def _run(self, cmd: list[str], **kwargs: Any) -> str:
        if False:
            print('Hello World!')
        '\n        Run a command inside the Python environment.\n        '
        call = kwargs.pop('call', False)
        env = kwargs.pop('env', dict(os.environ))
        stderr = kwargs.pop('stderr', subprocess.STDOUT)
        try:
            if call:
                assert stderr != subprocess.PIPE
                subprocess.check_call(cmd, stderr=stderr, env=env, **kwargs)
                output = ''
            else:
                output = subprocess.check_output(cmd, stderr=stderr, env=env, text=True, **kwargs)
        except CalledProcessError as e:
            raise EnvCommandError(e)
        return output

    def execute(self, bin: str, *args: str, **kwargs: Any) -> int:
        if False:
            while True:
                i = 10
        command = self.get_command_from_bin(bin) + list(args)
        env = kwargs.pop('env', dict(os.environ))
        if not self._is_windows:
            return os.execvpe(command[0], command, env=env)
        kwargs['shell'] = True
        exe = subprocess.Popen(command, env=env, **kwargs)
        exe.communicate()
        return exe.returncode

    def is_venv(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @property
    def script_dirs(self) -> list[Path]:
        if False:
            i = 10
            return i + 15
        if self._script_dirs is None:
            scripts = self.paths.get('scripts')
            self._script_dirs = [Path(scripts) if scripts is not None else self._bin_dir]
            if self.userbase:
                self._script_dirs.append(self.userbase / self._script_dirs[0].name)
        return self._script_dirs

    def _bin(self, bin: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return path to the given executable.\n        '
        if self._is_windows and (not bin.endswith('.exe')):
            bin_path = self._bin_dir / (bin + '.exe')
        else:
            bin_path = self._bin_dir / bin
        if not bin_path.exists():
            if self._is_windows:
                if not bin.endswith('.exe'):
                    bin_path = self._path / (bin + '.exe')
                else:
                    bin_path = self._path / bin
                if bin_path.exists():
                    return str(bin_path)
            return bin
        return str(bin_path)

    def __eq__(self, other: object) -> bool:
        if False:
            i = 10
            return i + 15
        if not isinstance(other, Env):
            return False
        return other.__class__ == self.__class__ and other.path == self.path

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'{self.__class__.__name__}("{self._path}")'