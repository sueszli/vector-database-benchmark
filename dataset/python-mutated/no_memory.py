from typing import Optional, List, Any, Dict
from haystack.agents.memory import Memory

class NoMemory(Memory):
    """
    A memory class that doesn't store any data.
    """

    def load(self, keys: Optional[List[str]]=None, **kwargs) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Load an empty dictionary.\n\n        :param keys: Optional list of keys (ignored in this implementation).\n        :return: An empty str.\n        '
        return ''

    def save(self, data: Dict[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Save method that does nothing.\n\n        :param data: A dictionary containing the data to save (ignored in this implementation).\n        '
        pass

    def clear(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Clear method that does nothing.\n        '
        pass