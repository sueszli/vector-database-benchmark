from datetime import datetime
from typing import Optional
from .scheduler import UTC_ZERO
from .virtualtimescheduler import VirtualTimeScheduler

class HistoricalScheduler(VirtualTimeScheduler):
    """Provides a virtual time scheduler that uses datetime for absolute time
    and timedelta for relative time."""

    def __init__(self, initial_clock: Optional[datetime]=None) -> None:
        if False:
            print('Hello World!')
        'Creates a new historical scheduler with the specified initial clock\n        value.\n\n        Args:\n            initial_clock: Initial value for the clock.\n        '
        super().__init__(initial_clock or UTC_ZERO)