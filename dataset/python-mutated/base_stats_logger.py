from __future__ import annotations
from typing import TYPE_CHECKING, Any
from airflow.metrics.protocols import Timer
from airflow.typing_compat import Protocol
if TYPE_CHECKING:
    from airflow.metrics.protocols import DeltaType, TimerProtocol

class StatsLogger(Protocol):
    """This class is only used for TypeChecking (for IDEs, mypy, etc)."""
    instance: StatsLogger | NoStatsLogger | None = None

    @classmethod
    def incr(cls, stat: str, count: int=1, rate: int | float=1, *, tags: dict[str, Any] | None=None) -> None:
        if False:
            return 10
        'Increment stat.'

    @classmethod
    def decr(cls, stat: str, count: int=1, rate: int | float=1, *, tags: dict[str, Any] | None=None) -> None:
        if False:
            print('Hello World!')
        'Decrement stat.'

    @classmethod
    def gauge(cls, stat: str, value: float, rate: int | float=1, delta: bool=False, *, tags: dict[str, Any] | None=None) -> None:
        if False:
            return 10
        'Gauge stat.'

    @classmethod
    def timing(cls, stat: str, dt: DeltaType | None, *, tags: dict[str, Any] | None=None) -> None:
        if False:
            print('Hello World!')
        'Stats timing.'

    @classmethod
    def timer(cls, *args, **kwargs) -> TimerProtocol:
        if False:
            return 10
        'Timer metric that can be cancelled.'
        raise NotImplementedError()

class NoStatsLogger:
    """If no StatsLogger is configured, NoStatsLogger is used as a fallback."""

    @classmethod
    def incr(cls, stat: str, count: int=1, rate: int=1, *, tags: dict[str, str] | None=None) -> None:
        if False:
            i = 10
            return i + 15
        'Increment stat.'

    @classmethod
    def decr(cls, stat: str, count: int=1, rate: int=1, *, tags: dict[str, str] | None=None) -> None:
        if False:
            print('Hello World!')
        'Decrement stat.'

    @classmethod
    def gauge(cls, stat: str, value: int, rate: int=1, delta: bool=False, *, tags: dict[str, str] | None=None) -> None:
        if False:
            return 10
        'Gauge stat.'

    @classmethod
    def timing(cls, stat: str, dt: DeltaType, *, tags: dict[str, str] | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Stats timing.'

    @classmethod
    def timer(cls, *args, **kwargs) -> TimerProtocol:
        if False:
            return 10
        'Timer metric that can be cancelled.'
        return Timer()