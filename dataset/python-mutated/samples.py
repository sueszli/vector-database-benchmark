from typing import Dict, NamedTuple, Optional, Union

class Timestamp:
    """A nanosecond-resolution timestamp."""

    def __init__(self, sec: float, nsec: float) -> None:
        if False:
            return 10
        if nsec < 0 or nsec >= 1000000000.0:
            raise ValueError(f'Invalid value for nanoseconds in Timestamp: {nsec}')
        if sec < 0:
            nsec = -nsec
        self.sec: int = int(sec)
        self.nsec: int = int(nsec)

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        return f'{self.sec}.{self.nsec:09d}'

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'Timestamp({self.sec}, {self.nsec})'

    def __float__(self) -> float:
        if False:
            while True:
                i = 10
        return float(self.sec) + float(self.nsec) / 1000000000.0

    def __eq__(self, other: object) -> bool:
        if False:
            i = 10
            return i + 15
        return isinstance(other, Timestamp) and self.sec == other.sec and (self.nsec == other.nsec)

    def __ne__(self, other: object) -> bool:
        if False:
            while True:
                i = 10
        return not self == other

    def __gt__(self, other: 'Timestamp') -> bool:
        if False:
            return 10
        return self.sec > other.sec or self.nsec > other.nsec

    def __lt__(self, other: 'Timestamp') -> bool:
        if False:
            while True:
                i = 10
        return self.sec < other.sec or self.nsec < other.nsec

class Exemplar(NamedTuple):
    labels: Dict[str, str]
    value: float
    timestamp: Optional[Union[float, Timestamp]] = None

class Sample(NamedTuple):
    name: str
    labels: Dict[str, str]
    value: float
    timestamp: Optional[Union[float, Timestamp]] = None
    exemplar: Optional[Exemplar] = None