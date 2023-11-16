from abc import ABC, abstractmethod

class ClusterEnvironment(ABC):
    """Specification of a cluster environment."""

    @property
    @abstractmethod
    def creates_processes_externally(self) -> bool:
        if False:
            while True:
                i = 10
        'Whether the environment creates the subprocesses or not.'

    @property
    @abstractmethod
    def main_address(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'The main address through which all processes connect and communicate.'

    @property
    @abstractmethod
    def main_port(self) -> int:
        if False:
            return 10
        'An open and configured port in the main node through which all processes communicate.'

    @staticmethod
    @abstractmethod
    def detect() -> bool:
        if False:
            print('Hello World!')
        'Detects the environment settings corresponding to this cluster and returns ``True`` if they match.'

    @abstractmethod
    def world_size(self) -> int:
        if False:
            print('Hello World!')
        'The number of processes across all devices and nodes.'

    @abstractmethod
    def set_world_size(self, size: int) -> None:
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def global_rank(self) -> int:
        if False:
            print('Hello World!')
        'The rank (index) of the currently running process across all nodes and devices.'

    @abstractmethod
    def set_global_rank(self, rank: int) -> None:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def local_rank(self) -> int:
        if False:
            i = 10
            return i + 15
        'The rank (index) of the currently running process inside of the current node.'

    @abstractmethod
    def node_rank(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'The rank (index) of the node on which the current process runs.'

    def validate_settings(self, num_devices: int, num_nodes: int) -> None:
        if False:
            i = 10
            return i + 15
        'Validates settings configured in the script against the environment, and raises an exception if there is an\n        inconsistency.'
        pass

    def teardown(self) -> None:
        if False:
            while True:
                i = 10
        'Clean up any state set after execution finishes.'
        pass