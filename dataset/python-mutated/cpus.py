import multiprocessing
import os
from typing import Optional
import spack.config

def cpus_available():
    if False:
        i = 10
        return i + 15
    '\n    Returns the number of CPUs available for the current process, or the number\n    of phyiscal CPUs when that information cannot be retrieved. The number\n    of available CPUs might differ from the number of physical CPUs when\n    using spack through Slurm or container runtimes.\n    '
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return multiprocessing.cpu_count()

def determine_number_of_jobs(*, parallel: bool=False, max_cpus: int=cpus_available(), config: Optional['spack.config.Configuration']=None) -> int:
    if False:
        while True:
            i = 10
    '\n    Packages that require sequential builds need 1 job. Otherwise we use the\n    number of jobs set on the command line. If not set, then we use the config\n    defaults (which is usually set through the builtin config scope), but we\n    cap to the number of CPUs available to avoid oversubscription.\n\n    Parameters:\n        parallel: true when package supports parallel builds\n        max_cpus: maximum number of CPUs to use (defaults to cpus_available())\n        config: configuration object (defaults to global config)\n    '
    if not parallel:
        return 1
    cfg = config or spack.config.CONFIG
    try:
        command_line = cfg.get('config:build_jobs', default=None, scope='command_line')
        if command_line is not None:
            return command_line
    except ValueError:
        pass
    return min(max_cpus, cfg.get('config:build_jobs', 16))