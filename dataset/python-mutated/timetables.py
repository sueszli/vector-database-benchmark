from __future__ import annotations
from airflow import settings
from airflow.timetables.base import DataInterval, Timetable
from airflow.timetables.interval import CronDataIntervalTimetable, DeltaDataIntervalTimetable

def cron_timetable(expr: str) -> CronDataIntervalTimetable:
    if False:
        print('Hello World!')
    return CronDataIntervalTimetable(expr, settings.TIMEZONE)

def delta_timetable(delta) -> DeltaDataIntervalTimetable:
    if False:
        return 10
    return DeltaDataIntervalTimetable(delta)

class CustomSerializationTimetable(Timetable):

    def __init__(self, value: str):
        if False:
            while True:
                i = 10
        self.value = value

    @classmethod
    def deserialize(cls, data):
        if False:
            i = 10
            return i + 15
        return cls(data['value'])

    def __eq__(self, other) -> bool:
        if False:
            while True:
                i = 10
        'Only for testing purposes.'
        if not isinstance(other, CustomSerializationTimetable):
            return False
        return self.value == other.value

    def serialize(self):
        if False:
            print('Hello World!')
        return {'value': self.value}

    @property
    def summary(self):
        if False:
            while True:
                i = 10
        return f'{type(self).__name__}({self.value!r})'

    def infer_manual_data_interval(self, *, run_after):
        if False:
            print('Hello World!')
        raise DataInterval.exact(run_after)

    def next_dagrun_info(self, *, last_automated_data_interval, restriction):
        if False:
            return 10
        return None