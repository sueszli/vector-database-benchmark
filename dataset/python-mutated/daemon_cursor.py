from abc import abstractmethod
from typing import Mapping, Set

class DaemonCursorStorage:

    @abstractmethod
    def get_cursor_values(self, keys: Set[str]) -> Mapping[str, str]:
        if False:
            return 10
        'Retrieve the value for a given key in the current deployment.'

    @abstractmethod
    def set_cursor_values(self, pairs: Mapping[str, str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the value for a given key in the current deployment.'