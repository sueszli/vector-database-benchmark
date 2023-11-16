import logging
from typing import List, Optional
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
import subprocess
default_logger = logging.getLogger(__name__)

class MPIPlugin(RuntimeEnvPlugin):
    """This plugin enable a MPI cluster to run on top of ray.

    To use this, "mpi" need to be added to the runtime env like following

    @ray.remote(
        runtime_env={
            "mpi": {
                "args": ["-n", "4"],
                "worker_entry": worker_entry,
            }
        }
    )
    def calc_pi():
      ...

    Here worker_entry should be function for the MPI worker to run.
    For example, it should be `'py_module.worker_func'`. The module should be able to
    be imported in the runtime.

    In the mpi worker with rank==0, it'll be the normal ray function or actor.
    For the worker with rank > 0, it'll just run `worker_func`.
    """
    priority = 90
    name = 'mpi'

    def modify_context(self, uris: List[str], runtime_env: 'RuntimeEnv', context: RuntimeEnvContext, logger: Optional[logging.Logger]=default_logger) -> None:
        if False:
            return 10
        mpi_config = runtime_env.mpi()
        if mpi_config is None:
            return
        try:
            proc = subprocess.run(['mpirun', '--version'], capture_output=True, check=True)
        except subprocess.CalledProcessError:
            logger.exception('Failed to run mpi run. Please make sure mpi has been installed')
            raise
        logger.info(f'Running MPI plugin\n {proc.stdout.decode()}')
        worker_entry = mpi_config.get('worker_entry')
        assert worker_entry is not None, '`worker_entry` must be setup in the runtime env.'
        cmds = ['mpirun'] + mpi_config.get('args', []) + [context.py_executable, '-m', 'ray._private.runtime_env.mpi_runner', worker_entry]
        context.py_executable = ' '.join(cmds)