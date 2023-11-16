import contextlib
import os
import re
import llnl.util.filesystem as fs
import llnl.util.lang
import llnl.util.tty as tty
from spack.util.environment import get_path
from spack.util.module_cmd import module
from .linux_distro import LinuxDistro

@contextlib.contextmanager
def unload_programming_environment():
    if False:
        for i in range(10):
            print('nop')
    'Context manager that unloads Cray Programming Environments.'
    env_bu = None
    if 'PE_ENV' in os.environ:
        env_bu = os.environ.copy()
        prg_env = 'PrgEnv-' + os.environ['PE_ENV'].lower()
        module('unload', prg_env)
    yield
    if env_bu is not None:
        os.environ.clear()
        os.environ.update(env_bu)

class CrayFrontend(LinuxDistro):
    """Represents OS that runs on login and service nodes of the Cray platform.
    It acts as a regular Linux without Cray-specific modules and compiler
    wrappers."""

    @property
    def compiler_search_paths(self):
        if False:
            for i in range(10):
                print('nop')
        "Calls the default function but unloads Cray's programming\n        environments first.\n\n        This prevents from detecting Cray compiler wrappers and avoids\n        possible false detections.\n        "
        import spack.compilers
        with unload_programming_environment():
            search_paths = get_path('PATH')
        extract_path_re = re.compile('prepend-path[\\s]*PATH[\\s]*([/\\w\\.:-]*)')
        for compiler_cls in spack.compilers.all_compiler_types():
            prg_env = getattr(compiler_cls, 'PrgEnv', None)
            compiler_module = getattr(compiler_cls, 'PrgEnv_compiler', None)
            if not (prg_env and compiler_module):
                continue
            output = module('avail', compiler_cls.PrgEnv_compiler)
            version_regex = '({0})/([\\d\\.]+[\\d]-?[\\w]*)'.format(compiler_cls.PrgEnv_compiler)
            matches = re.findall(version_regex, output)
            versions = tuple((version for (_, version) in matches if 'classic' not in version))
            msg = '[CRAY FE] Detected FE compiler [name={0}, versions={1}]'
            tty.debug(msg.format(compiler_module, versions))
            for v in versions:
                try:
                    current_module = compiler_module + '/' + v
                    out = module('show', current_module)
                    match = extract_path_re.search(out)
                    search_paths += match.group(1).split(':')
                except Exception as e:
                    msg = '[CRAY FE] An unexpected error occurred while detecting FE compiler [compiler={0},  version={1}, error={2}]'
                    tty.debug(msg.format(compiler_cls.name, v, str(e)))
        search_paths = list(llnl.util.lang.dedupe(search_paths))
        return fs.search_paths_for_executables(*search_paths)