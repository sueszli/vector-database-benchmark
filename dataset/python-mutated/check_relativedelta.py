from __future__ import annotations
from dateutil import relativedelta

class Calendar:

    def __init__(self, week_start: relativedelta.weekday=relativedelta.MO) -> None:
        if False:
            i = 10
            return i + 15
        self.week_start = week_start