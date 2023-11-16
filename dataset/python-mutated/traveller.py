from __future__ import annotations
from typing import TYPE_CHECKING
from typing import cast
from pendulum.datetime import DateTime
from pendulum.utils._compat import PYPY
if TYPE_CHECKING:
    from types import TracebackType
    from typing_extensions import Self

class BaseTraveller:

    def __init__(self, datetime_class: type[DateTime]=DateTime) -> None:
        if False:
            while True:
                i = 10
        self._datetime_class: type[DateTime] = datetime_class

    def freeze(self) -> Self:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def travel_back(self) -> Self:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def travel(self, years: int=0, months: int=0, weeks: int=0, days: int=0, hours: int=0, minutes: int=0, seconds: int=0, microseconds: int=0) -> Self:
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def travel_to(self, dt: DateTime, *, freeze: bool=False) -> Self:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def __enter__(self) -> Self:
        if False:
            i = 10
            return i + 15
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType) -> None:
        if False:
            i = 10
            return i + 15
        ...
if not PYPY:
    import time_machine

    class Traveller(BaseTraveller):

        def __init__(self, datetime_class: type[DateTime]=DateTime) -> None:
            if False:
                for i in range(10):
                    print('nop')
            super().__init__(datetime_class)
            self._started: bool = False
            self._traveller: time_machine.travel | None = None
            self._coordinates: time_machine.Coordinates | None = None

        def freeze(self) -> Self:
            if False:
                for i in range(10):
                    print('nop')
            if self._started:
                cast(time_machine.Coordinates, self._coordinates).move_to(self._datetime_class.now(), tick=False)
            else:
                self._start(freeze=True)
            return self

        def travel_back(self) -> Self:
            if False:
                for i in range(10):
                    print('nop')
            if not self._started:
                return self
            cast(time_machine.travel, self._traveller).stop()
            self._coordinates = None
            self._traveller = None
            self._started = False
            return self

        def travel(self, years: int=0, months: int=0, weeks: int=0, days: int=0, hours: int=0, minutes: int=0, seconds: int=0, microseconds: int=0, *, freeze: bool=False) -> Self:
            if False:
                for i in range(10):
                    print('nop')
            self._start(freeze=freeze)
            cast(time_machine.Coordinates, self._coordinates).move_to(self._datetime_class.now().add(years=years, months=months, weeks=weeks, days=days, hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds))
            return self

        def travel_to(self, dt: DateTime, *, freeze: bool=False) -> Self:
            if False:
                i = 10
                return i + 15
            self._start(freeze=freeze)
            cast(time_machine.Coordinates, self._coordinates).move_to(dt)
            return self

        def _start(self, freeze: bool=False) -> None:
            if False:
                return 10
            if self._started:
                return
            if not self._traveller:
                self._traveller = time_machine.travel(self._datetime_class.now(), tick=not freeze)
            self._coordinates = self._traveller.start()
            self._started = True

        def __enter__(self) -> Self:
            if False:
                for i in range(10):
                    print('nop')
            self._start()
            return self

        def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType) -> None:
            if False:
                while True:
                    i = 10
            self.travel_back()
else:

    class Traveller(BaseTraveller):
        ...