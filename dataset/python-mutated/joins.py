"""Join strategies."""
from typing import Any, Optional, Tuple
from .types import EventT, FieldDescriptorT, JoinT, JoinableT
__all__ = ['Join', 'RightJoin', 'LeftJoin', 'InnerJoin', 'OuterJoin']

class Join(JoinT):
    """Base class for join strategies."""

    def __init__(self, *, stream: JoinableT, fields: Tuple[FieldDescriptorT, ...]) -> None:
        if False:
            return 10
        self.fields = {field.model: field for field in fields}
        self.stream = stream

    async def process(self, event: EventT) -> Optional[EventT]:
        """Process event to be joined with another event."""
        raise NotImplementedError()

    def __eq__(self, other: Any) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, type(self)):
            return other.fields == self.fields and other.stream is self.stream
        return False

    def __ne__(self, other: Any) -> bool:
        if False:
            print('Hello World!')
        return not self.__eq__(other)

class RightJoin(Join):
    """Right-join strategy."""

class LeftJoin(Join):
    """Left-join strategy."""

class InnerJoin(Join):
    """Inner-join strategy."""

class OuterJoin(Join):
    """Outer-join strategy."""