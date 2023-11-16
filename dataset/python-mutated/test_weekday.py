from __future__ import annotations
from enum import Enum
import pytest
from airflow.utils.weekday import WeekDay

class TestWeekDay:

    def test_weekday_enum_length(self):
        if False:
            while True:
                i = 10
        assert len(WeekDay) == 7

    def test_weekday_name_value(self):
        if False:
            i = 10
            return i + 15
        weekdays = 'MONDAY TUESDAY WEDNESDAY THURSDAY FRIDAY SATURDAY SUNDAY'
        weekdays = weekdays.split()
        for (i, weekday) in enumerate(weekdays, start=1):
            weekday_enum = WeekDay(i)
            assert weekday_enum == i
            assert int(weekday_enum) == i
            assert weekday_enum.name == weekday
            assert weekday_enum in WeekDay
            assert 0 < weekday_enum < 8
            assert isinstance(weekday_enum, WeekDay)
            assert isinstance(weekday_enum, int)
            assert isinstance(weekday_enum, Enum)

    @pytest.mark.parametrize('weekday, expected', [('Monday', 1), (WeekDay.MONDAY, 1)], ids=['with-string', 'with-enum'])
    def test_convert(self, weekday, expected):
        if False:
            return 10
        result = WeekDay.convert(weekday)
        assert result == expected

    def test_convert_with_incorrect_input(self):
        if False:
            print('Hello World!')
        invalid = 'Sun'
        error_message = f'Invalid Week Day passed: "{invalid}"'
        with pytest.raises(AttributeError, match=error_message):
            WeekDay.convert(invalid)

    @pytest.mark.parametrize('weekday, expected', [('Monday', {WeekDay.MONDAY}), (WeekDay.MONDAY, {WeekDay.MONDAY}), ({'Thursday': '1'}, {WeekDay.THURSDAY}), (['Thursday'], {WeekDay.THURSDAY}), (['Thursday', WeekDay.MONDAY], {WeekDay.MONDAY, WeekDay.THURSDAY})], ids=['with-string', 'with-enum', 'with-dict', 'with-list', 'with-mix'])
    def test_validate_week_day(self, weekday, expected):
        if False:
            i = 10
            return i + 15
        result = WeekDay.validate_week_day(weekday)
        assert expected == result

    def test_validate_week_day_with_invalid_type(self):
        if False:
            for i in range(10):
                print('nop')
        invalid_week_day = 5
        with pytest.raises(TypeError, match=f'Unsupported Type for week_day parameter: {type(invalid_week_day)}.Input should be iterable type:str, set, list, dict or Weekday enum type'):
            WeekDay.validate_week_day(invalid_week_day)