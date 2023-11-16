from ._compat import Protocol
from typing import Any, Dict, Iterator, List, TypeVar, Union
_T = TypeVar('_T')

class PackageMetadata(Protocol):

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        ...

    def __contains__(self, item: str) -> bool:
        if False:
            print('Hello World!')
        ...

    def __getitem__(self, key: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        ...

    def __iter__(self) -> Iterator[str]:
        if False:
            return 10
        ...

    def get_all(self, name: str, failobj: _T=...) -> Union[List[Any], _T]:
        if False:
            i = 10
            return i + 15
        '\n        Return all values associated with a possibly multi-valued key.\n        '

    @property
    def json(self) -> Dict[str, Union[str, List[str]]]:
        if False:
            return 10
        '\n        A JSON-compatible form of the metadata.\n        '

class SimplePath(Protocol):
    """
    A minimal subset of pathlib.Path required by PathDistribution.
    """

    def joinpath(self) -> 'SimplePath':
        if False:
            return 10
        ...

    def __truediv__(self) -> 'SimplePath':
        if False:
            for i in range(10):
                print('nop')
        ...

    def parent(self) -> 'SimplePath':
        if False:
            for i in range(10):
                print('nop')
        ...

    def read_text(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        ...