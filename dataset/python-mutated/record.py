from typing import Any, Mapping

class Record:
    """
    Represents a record read from a stream.
    """

    def __init__(self, data: Mapping[str, Any]):
        if False:
            for i in range(10):
                print('nop')
        self.data = data

    def __eq__(self, other: Any) -> bool:
        if False:
            print('Hello World!')
        if not isinstance(other, Record):
            return False
        return self.data == other.data

    def __repr__(self) -> str:
        if False:
            return 10
        return f'Record(data={self.data})'