import logging
import os
import re
import shutil
import signal
import sys
from typing import Optional
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.fabric.utilities.imports import _IS_WINDOWS
from lightning.fabric.utilities.rank_zero import rank_zero_warn
from lightning.fabric.utilities.warnings import PossibleUserWarning
log = logging.getLogger(__name__)

class SLURMEnvironment(ClusterEnvironment):
    """Cluster environment for training on a cluster managed by SLURM.

    You can configure the `main_address` and `main_port` properties via the env variables `MASTER_ADDR` and
    `MASTER_PORT`, respectively.

    Args:
        auto_requeue: Whether automatic job resubmission is enabled or not. How and under which conditions a job gets
            rescheduled gets determined by the owner of this plugin.
        requeue_signal: The signal that SLURM will send to indicate that the job should be requeued. Defaults to
            SIGUSR1 on Unix.

    """

    def __init__(self, auto_requeue: bool=True, requeue_signal: Optional[signal.Signals]=None) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.auto_requeue = auto_requeue
        if requeue_signal is None and (not _IS_WINDOWS):
            requeue_signal = signal.SIGUSR1
        self.requeue_signal = requeue_signal
        self._validate_srun_used()
        self._validate_srun_variables()

    @property
    def creates_processes_externally(self) -> bool:
        if False:
            return 10
        return True

    @property
    def main_address(self) -> str:
        if False:
            i = 10
            return i + 15
        root_node = os.environ.get('MASTER_ADDR')
        if root_node is None:
            nodelist = os.environ.get('SLURM_NODELIST', '127.0.0.1')
            root_node = self.resolve_root_node_address(nodelist)
            os.environ['MASTER_ADDR'] = root_node
        log.debug(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        return root_node

    @property
    def main_port(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        job_id = os.environ.get('SLURM_JOB_ID')
        if job_id is not None:
            default_port = job_id[-4:]
            default_port = int(default_port) + 15000
        else:
            default_port = 12910
        if 'MASTER_PORT' in os.environ:
            default_port = int(os.environ['MASTER_PORT'])
        else:
            os.environ['MASTER_PORT'] = str(default_port)
        log.debug(f"MASTER_PORT: {os.environ['MASTER_PORT']}")
        return default_port

    @staticmethod
    def detect() -> bool:
        if False:
            for i in range(10):
                print('nop')
        "Returns ``True`` if the current process was launched on a SLURM cluster.\n\n        It is possible to use the SLURM scheduler to request resources and then launch processes manually using a\n        different environment. For this, the user can set the job name in SLURM to 'bash' or 'interactive' (srun --job-\n        name=interactive). This will then avoid the detection of ``SLURMEnvironment`` and another environment can be\n        detected automatically.\n\n        "
        SLURMEnvironment._validate_srun_used()
        return _is_srun_used()

    @staticmethod
    def job_name() -> Optional[str]:
        if False:
            return 10
        return os.environ.get('SLURM_JOB_NAME')

    @staticmethod
    def job_id() -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        if _is_slurm_interactive_mode():
            return None
        job_id = os.environ.get('SLURM_JOB_ID')
        if job_id is None:
            return None
        try:
            return int(job_id)
        except ValueError:
            return None

    def world_size(self) -> int:
        if False:
            print('Hello World!')
        return int(os.environ['SLURM_NTASKS'])

    def set_world_size(self, size: int) -> None:
        if False:
            print('Hello World!')
        log.debug('SLURMEnvironment.set_world_size was called, but setting world size is not allowed. Ignored.')

    def global_rank(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return int(os.environ['SLURM_PROCID'])

    def set_global_rank(self, rank: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        log.debug('SLURMEnvironment.set_global_rank was called, but setting global rank is not allowed. Ignored.')

    def local_rank(self) -> int:
        if False:
            i = 10
            return i + 15
        return int(os.environ['SLURM_LOCALID'])

    def node_rank(self) -> int:
        if False:
            return 10
        return int(os.environ['SLURM_NODEID'])

    def validate_settings(self, num_devices: int, num_nodes: int) -> None:
        if False:
            while True:
                i = 10
        if _is_slurm_interactive_mode():
            return
        ntasks_per_node = os.environ.get('SLURM_NTASKS_PER_NODE')
        if ntasks_per_node is not None and int(ntasks_per_node) != num_devices:
            raise ValueError(f'You set `devices={num_devices}` in Lightning, but the number of tasks per node configured in SLURM `--ntasks-per-node={ntasks_per_node}` does not match. HINT: Set `devices={ntasks_per_node}`.')
        nnodes = os.environ.get('SLURM_NNODES')
        if nnodes is not None and int(nnodes) != num_nodes:
            raise ValueError(f'You set `num_nodes={num_nodes}` in Lightning, but the number of nodes configured in SLURM `--nodes={nnodes}` does not match. HINT: Set `num_nodes={nnodes}`.')

    @staticmethod
    def resolve_root_node_address(nodes: str) -> str:
        if False:
            while True:
                i = 10
        "The node selection format in SLURM supports several formats.\n\n        This function selects the first host name from\n\n        - a space-separated list of host names, e.g., 'host0 host1 host3' yields 'host0' as the root\n        - a comma-separated list of host names, e.g., 'host0,host1,host3' yields 'host0' as the root\n        - the range notation with brackets, e.g., 'host[5-9]' yields 'host5' as the root\n\n        "
        nodes = re.sub('\\[(.*?)[,-].*\\]', '\\1', nodes)
        nodes = re.sub('\\[(.*?)\\]', '\\1', nodes)
        return nodes.split(' ')[0].split(',')[0]

    @staticmethod
    def _validate_srun_used() -> None:
        if False:
            for i in range(10):
                print('nop')
        'Checks if the `srun` command is available and used.\n\n        Parallel jobs (multi-GPU, multi-node) in SLURM are launched by prepending `srun` in front of the Python command.\n        Not doing so will result in processes hanging, which is a frequent user error. Lightning will emit a warning if\n        `srun` is found but not used.\n\n        '
        if _IS_WINDOWS:
            return
        srun_exists = shutil.which('srun') is not None
        if srun_exists and (not _is_srun_used()):
            hint = ' '.join(['srun', os.path.basename(sys.executable), *sys.argv])[:64]
            rank_zero_warn(f'The `srun` command is available on your system but is not used. HINT: If your intention is to run Lightning on SLURM, prepend your python command with `srun` like so: {hint} ...', category=PossibleUserWarning)

    @staticmethod
    def _validate_srun_variables() -> None:
        if False:
            for i in range(10):
                print('nop')
        'Checks for conflicting or incorrectly set variables set through `srun` and raises a useful error message.\n\n        Right now, we only check for the most common user errors. See\n        `the srun docs <https://slurm.schedmd.com/srun.html>`_\n        for a complete list of supported srun variables.\n\n        '
        ntasks = int(os.environ.get('SLURM_NTASKS', '1'))
        if ntasks > 1 and 'SLURM_NTASKS_PER_NODE' not in os.environ:
            raise RuntimeError(f'You set `--ntasks={ntasks}` in your SLURM bash script, but this variable is not supported. HINT: Use `--ntasks-per-node={ntasks}` instead.')

def _is_srun_used() -> bool:
    if False:
        return 10
    return 'SLURM_NTASKS' in os.environ and (not _is_slurm_interactive_mode())

def _is_slurm_interactive_mode() -> bool:
    if False:
        for i in range(10):
            print('nop')
    return SLURMEnvironment.job_name() in ('bash', 'interactive')