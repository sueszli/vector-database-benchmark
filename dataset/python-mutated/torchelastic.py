import logging
import os
import torch.distributed
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.fabric.utilities.rank_zero import rank_zero_warn
log = logging.getLogger(__name__)

class TorchElasticEnvironment(ClusterEnvironment):
    """Environment for fault-tolerant and elastic training with `torchelastic <https://pytorch.org/elastic/>`_"""

    @property
    def creates_processes_externally(self) -> bool:
        if False:
            print('Hello World!')
        return True

    @property
    def main_address(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        if 'MASTER_ADDR' not in os.environ:
            rank_zero_warn('MASTER_ADDR environment variable is not defined. Set as localhost')
            os.environ['MASTER_ADDR'] = '127.0.0.1'
        log.debug(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        return os.environ['MASTER_ADDR']

    @property
    def main_port(self) -> int:
        if False:
            print('Hello World!')
        if 'MASTER_PORT' not in os.environ:
            rank_zero_warn('MASTER_PORT environment variable is not defined. Set as 12910')
            os.environ['MASTER_PORT'] = '12910'
        log.debug(f"MASTER_PORT: {os.environ['MASTER_PORT']}")
        return int(os.environ['MASTER_PORT'])

    @staticmethod
    def detect() -> bool:
        if False:
            print('Hello World!')
        'Returns ``True`` if the current process was launched using the torchelastic command.'
        return torch.distributed.is_available() and torch.distributed.is_torchelastic_launched()

    def world_size(self) -> int:
        if False:
            i = 10
            return i + 15
        return int(os.environ['WORLD_SIZE'])

    def set_world_size(self, size: int) -> None:
        if False:
            while True:
                i = 10
        log.debug('TorchElasticEnvironment.set_world_size was called, but setting world size is not allowed. Ignored.')

    def global_rank(self) -> int:
        if False:
            print('Hello World!')
        return int(os.environ['RANK'])

    def set_global_rank(self, rank: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        log.debug('TorchElasticEnvironment.set_global_rank was called, but setting global rank is not allowed. Ignored.')

    def local_rank(self) -> int:
        if False:
            print('Hello World!')
        return int(os.environ['LOCAL_RANK'])

    def node_rank(self) -> int:
        if False:
            i = 10
            return i + 15
        return int(os.environ.get('GROUP_RANK', 0))

    def validate_settings(self, num_devices: int, num_nodes: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        if num_devices * num_nodes != self.world_size():
            raise ValueError(f'You set `devices={num_devices}` and `num_nodes={num_nodes}` in Lightning, but the product ({num_devices} * {num_nodes}) does not match the world size ({self.world_size()}).')