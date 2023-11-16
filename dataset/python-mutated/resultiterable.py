from typing import TypeVar, TYPE_CHECKING, Iterator, Iterable
if TYPE_CHECKING:
    from pyspark._typing import SizedIterable
__all__ = ['ResultIterable']
T = TypeVar('T')

class ResultIterable(Iterable[T]):
    """
    A special result iterable. This is used because the standard
    iterator can not be pickled
    """

    def __init__(self, data: 'SizedIterable[T]'):
        if False:
            while True:
                i = 10
        self.data: 'SizedIterable[T]' = data
        self.index: int = 0
        self.maxindex: int = len(data)

    def __iter__(self) -> Iterator[T]:
        if False:
            i = 10
            return i + 15
        return iter(self.data)

    def __len__(self) -> int:
        if False:
            return 10
        return len(self.data)