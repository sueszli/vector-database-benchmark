import json
import os
import shutil
import subprocess as sp
import sys
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from build import BuildBackendException, ConfigSettingsType
from build.env import DefaultIsolatedEnv
from packaging.requirements import Requirement
from . import common, pywasmcross
from .build_env import get_build_flag, get_hostsitepackages, get_pyversion, get_unisolated_packages, platform
from .io import _BuildSpecExports
from .vendor._pypabuild import _STYLES, _DefaultIsolatedEnv, _error, _handle_build_error, _ProjectBuilder
AVOIDED_REQUIREMENTS = ['cmake', 'patchelf']

def _gen_runner(cross_build_env: Mapping[str, str], isolated_build_env: DefaultIsolatedEnv) -> Callable[[Sequence[str], str | None, Mapping[str, str] | None], None]:
    if False:
        i = 10
        return i + 15
    "\n    This returns a slightly modified version of default subprocess runner that pypa/build uses.\n    pypa/build prepends the virtual environment's bin directory to the PATH environment variable.\n    This is problematic because it shadows the pywasmcross compiler wrappers for cmake, meson, etc.\n\n    This function prepends the compiler wrapper directory to the PATH again so that our compiler wrappers\n    are searched first.\n\n    Parameters\n    ----------\n    cross_build_env\n        The cross build environment for pywasmcross.\n    isolated_build_env\n        The isolated build environment created by pypa/build.\n    "

    def _runner(cmd, cwd=None, extra_environ=None):
        if False:
            for i in range(10):
                print('nop')
        env = os.environ.copy()
        if extra_environ:
            env.update(extra_environ)
        env['BUILD_ENV_SCRIPTS_DIR'] = isolated_build_env._scripts_dir
        env['PATH'] = f"{cross_build_env['COMPILER_WRAPPER_DIR']}:{env['PATH']}"
        sp.check_call(cmd, cwd=cwd, env=env)
    return _runner

def symlink_unisolated_packages(env: DefaultIsolatedEnv) -> None:
    if False:
        i = 10
        return i + 15
    pyversion = get_pyversion()
    site_packages_path = f'lib/{pyversion}/site-packages'
    env_site_packages = Path(env.path) / site_packages_path
    sysconfigdata_name = get_build_flag('SYSCONFIG_NAME')
    sysconfigdata_path = Path(get_build_flag('TARGETINSTALLDIR')) / f'sysconfigdata/{sysconfigdata_name}.py'
    env_site_packages.mkdir(parents=True, exist_ok=True)
    shutil.copy(sysconfigdata_path, env_site_packages)
    host_site_packages = Path(get_hostsitepackages())
    for name in get_unisolated_packages():
        for path in chain(host_site_packages.glob(f'{name}*'), host_site_packages.glob(f'_{name}*')):
            (env_site_packages / path.name).unlink(missing_ok=True)
            (env_site_packages / path.name).symlink_to(path)

def remove_avoided_requirements(requires: set[str], avoided_requirements: set[str] | list[str]) -> set[str]:
    if False:
        print('Hello World!')
    for reqstr in list(requires):
        req = Requirement(reqstr)
        for avoid_name in set(avoided_requirements):
            if avoid_name in req.name.lower():
                requires.remove(reqstr)
    return requires

def install_reqs(env: DefaultIsolatedEnv, reqs: set[str]) -> None:
    if False:
        for i in range(10):
            print('nop')
    env.install(remove_avoided_requirements(reqs, get_unisolated_packages() + AVOIDED_REQUIREMENTS))

def _build_in_isolated_env(build_env: Mapping[str, str], srcdir: Path, outdir: str, distribution: str, config_settings: ConfigSettingsType) -> str:
    if False:
        while True:
            i = 10
    with _DefaultIsolatedEnv() as env:
        builder = _ProjectBuilder.from_isolated_env(env, srcdir, runner=_gen_runner(build_env, env))
        symlink_unisolated_packages(env)
        install_reqs(env, builder.build_system_requires)
        installed_requires_for_build = False
        try:
            build_reqs = builder.get_requires_for_build(distribution)
        except BuildBackendException:
            pass
        else:
            install_reqs(env, build_reqs)
            installed_requires_for_build = True
        with common.replace_env(build_env):
            if not installed_requires_for_build:
                build_reqs = builder.get_requires_for_build(distribution, config_settings)
                install_reqs(env, build_reqs)
            return builder.build(distribution, outdir, config_settings)

def parse_backend_flags(backend_flags: str) -> ConfigSettingsType:
    if False:
        i = 10
        return i + 15
    config_settings: dict[str, str | list[str]] = {}
    for arg in backend_flags.split():
        (setting, _, value) = arg.partition('=')
        if setting not in config_settings:
            config_settings[setting] = value
            continue
        cur_value = config_settings[setting]
        if isinstance(cur_value, str):
            config_settings[setting] = [cur_value, value]
        else:
            cur_value.append(value)
    return config_settings

def make_command_wrapper_symlinks(symlink_dir: Path) -> dict[str, str]:
    if False:
        i = 10
        return i + 15
    '\n    Create symlinks that make pywasmcross look like a compiler.\n\n    Parameters\n    ----------\n    symlink_dir\n        The directory where the symlinks will be created.\n\n    Returns\n    -------\n    The dictionary of compiler environment variables that points to the symlinks.\n    '
    pywasmcross_exe = symlink_dir / 'pywasmcross.py'
    shutil.copy2(pywasmcross.__file__, pywasmcross_exe)
    pywasmcross_exe.chmod(493)
    env = {}
    for symlink in pywasmcross.SYMLINKS:
        symlink_path = symlink_dir / symlink
        if os.path.lexists(symlink_path) and (not symlink_path.exists()):
            symlink_path.unlink()
        symlink_path.symlink_to(pywasmcross_exe)
        if symlink == 'c++':
            var = 'CXX'
        elif symlink == 'gfortran':
            var = 'FC'
        else:
            var = symlink.upper()
        env[var] = str(symlink_path)
    return env

@contextmanager
def get_build_env(env: dict[str, str], *, pkgname: str, cflags: str, cxxflags: str, ldflags: str, target_install_dir: str, exports: _BuildSpecExports) -> Iterator[dict[str, str]]:
    if False:
        print('Hello World!')
    '\n    Returns a dict of environment variables that should be used when building\n    a package with pypa/build.\n    '
    kwargs = dict(pkgname=pkgname, cflags=cflags, cxxflags=cxxflags, ldflags=ldflags, target_install_dir=target_install_dir)
    args = common.environment_substitute_args(kwargs, env)
    args['builddir'] = str(Path('.').absolute())
    args['exports'] = exports
    env = env.copy()
    with TemporaryDirectory() as symlink_dir_str:
        symlink_dir = Path(symlink_dir_str)
        env.update(make_command_wrapper_symlinks(symlink_dir))
        sysconfig_dir = Path(get_build_flag('TARGETINSTALLDIR')) / 'sysconfigdata'
        args['PYTHONPATH'] = sys.path + [str(sysconfig_dir)]
        args['orig__name__'] = __name__
        args['pythoninclude'] = get_build_flag('PYTHONINCLUDE')
        args['PATH'] = env['PATH']
        pywasmcross_env = json.dumps(args)
        env['PYWASMCROSS_ARGS'] = pywasmcross_env
        (symlink_dir / 'pywasmcross_env.json').write_text(pywasmcross_env)
        env['_PYTHON_HOST_PLATFORM'] = platform()
        env['_PYTHON_SYSCONFIGDATA_NAME'] = get_build_flag('SYSCONFIG_NAME')
        env['PYTHONPATH'] = str(sysconfig_dir)
        env['COMPILER_WRAPPER_DIR'] = str(symlink_dir)
        yield env

def build(srcdir: Path, outdir: Path, build_env: Mapping[str, str], backend_flags: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    distribution = 'wheel'
    config_settings = parse_backend_flags(backend_flags)
    try:
        with _handle_build_error():
            built = _build_in_isolated_env(build_env, srcdir, str(outdir), distribution, config_settings)
            print('{bold}{green}Successfully built {}{reset}'.format(built, **_STYLES))
            return built
    except Exception as e:
        tb = traceback.format_exc().strip('\n')
        print('\n{dim}{}{reset}\n'.format(tb, **_STYLES))
        _error(str(e))
        sys.exit(1)