from abc import ABC, abstractmethod
from typing import Any

class Pager(ABC):
    """Base class for a pager."""

    @abstractmethod
    def show(self, content: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Show content in pager.\n\n        Args:\n            content (str): Content to be displayed.\n        '

class SystemPager(Pager):
    """Uses the pager installed on the system."""

    def _pager(self, content: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        return __import__('pydoc').pager(content)

    def show(self, content: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Use the same pager used by pydoc.'
        self._pager(content)
if __name__ == '__main__':
    from .__main__ import make_test_card
    from .console import Console
    console = Console()
    with console.pager(styles=True):
        console.print(make_test_card())