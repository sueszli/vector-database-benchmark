import os
from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.utilities import rank_zero_only

class RayEnvironment(ClusterEnvironment):
    """Environment for PTL training on a Ray cluster."""

    def __init__(self, world_size):
        if False:
            print('Hello World!')
        'Create a Ray environment.'
        self.set_world_size(world_size)
        self._global_rank = 0
        self._is_remote = False
        self._main_port = -1

    @property
    def creates_processes_externally(self) -> bool:
        if False:
            while True:
                i = 10
        'Whether the environment creates the subprocesses or not.'
        return False

    @property
    def main_address(self) -> str:
        if False:
            return 10
        'The main address through which all processes connect and communicate.'
        return os.environ.get('MASTER_ADDR', '127.0.0.1')

    @property
    def main_port(self) -> int:
        if False:
            while True:
                i = 10
        'An open and configured port in the main node through which all processes communicate.'
        if self._main_port == -1:
            self._main_port = int(os.environ.get('MASTER_PORT', 0))
        return self._main_port

    @staticmethod
    def detect() -> bool:
        if False:
            return 10
        'Detects the environment settings and returns `True` if they match.'
        return True

    def world_size(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'The number of processes across all devices and nodes.'
        return self._world_size

    def set_world_size(self, size: int) -> None:
        if False:
            return 10
        'Set world size.'
        self._world_size = size

    def global_rank(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'The rank (index) of the currently running process across all nodes and devices.'
        return self._global_rank

    def set_global_rank(self, rank: int) -> None:
        if False:
            while True:
                i = 10
        'Set global rank.'
        self._global_rank = rank
        rank_zero_only.rank = rank

    def set_remote_execution(self, is_remote: bool) -> None:
        if False:
            return 10
        'Set remote or not.'
        self._is_remote = is_remote

    def is_remote(self) -> bool:
        if False:
            while True:
                i = 10
        'Whether execute the codes remotely.'
        return self._is_remote

    def local_rank(self) -> int:
        if False:
            i = 10
            return i + 15
        'The rank (index) of the currently running process inside of the current node.'
        return int(os.environ.get('LOCAL_RANK', 0))

    def node_rank(self) -> int:
        if False:
            while True:
                i = 10
        'The rank (index) of the node on which the current process runs.'
        group_rank = os.environ.get('GROUP_RANK', 0)
        return int(os.environ.get('NODE_RANK', group_rank))