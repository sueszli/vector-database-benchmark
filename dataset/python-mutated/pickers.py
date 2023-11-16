""" Various kinds of date, time and date/time picker widgets.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ...core.enums import CalendarPosition
from ...core.has_props import HasProps, abstract
from ...core.properties import Bool, Date, Datetime, Either, Enum, Int, List, Nullable, Override, Positive, String, Time, Tuple
from .inputs import InputWidget
__all__ = ('DatePicker', 'DateRangePicker', 'DatetimePicker', 'DatetimeRangePicker', 'MultipleDatePicker', 'MultipleDatetimePicker', 'TimePicker')

@abstract
class PickerBase(InputWidget):
    """ Base class for various kinds of picker widgets. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    position = Enum(CalendarPosition, default='auto', help='\n    Where the calendar is rendered relative to the input when ``inline`` is False.\n    ')
    inline = Bool(default=False, help='\n    Whether the calendar sholud be displayed inline.\n    ')

@abstract
class TimeCommon(HasProps):
    """ Common properties for time-like picker widgets. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    hour_increment = Positive(Int)(default=1, help='\n    Defines the granularity of hour value incremements in the UI.\n    ')
    minute_increment = Positive(Int)(default=1, help='\n    Defines the granularity of minute value incremements in the UI.\n    ')
    second_increment = Positive(Int)(default=1, help='\n    Defines the granularity of second value incremements in the UI.\n    ')
    seconds = Bool(default=False, help='\n    Allows to select seconds. By default only hours and minuts are\n    selectable, and AM/PM depending on ``clock`` option.\n    ')
    clock = Enum('12h', '24h', default='24h', help='\n    Whether to use 12 hour or 24 hour clock.\n    ')

class TimePicker(PickerBase, TimeCommon):
    """ Widget for picking time. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    value = Nullable(Time, default=None, help='\n    The initial or picked time.\n    ')
    time_format = String(default='H:i', help='\n    Formatting specification for the display of the picked date.\n\n    +---+------------------------------------+------------+\n    | H | Hours (24 hours)                   | 00 to 23   |\n    | h | Hours                              | 1 to 12    |\n    | G | Hours, 2 digits with leading zeros | 1 to 12    |\n    | i | Minutes                            | 00 to 59   |\n    | S | Seconds, 2 digits                  | 00 to 59   |\n    | s | Seconds                            | 0, 1 to 59 |\n    | K | AM/PM                              | AM or PM   |\n    +---+------------------------------------+------------+\n\n    See also https://flatpickr.js.org/formatting/#date-formatting-tokens.\n    ')
    min_time = Nullable(Time)(default=None, help='\n    Optional earliest allowable time.\n    ')
    max_time = Nullable(Time)(default=None, help='\n    Optional latest allowable time.\n    ')

@abstract
class DateCommon(HasProps):
    """ Common properties for date-like picker widgets. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    disabled_dates = Nullable(List(Either(Date, Tuple(Date, Date))), default=None, help='\n    A list of dates of ``(start, end)`` date ranges to make unavailable for\n    selection. All other dates will be avalable.\n\n    .. note::\n        Only one of ``disabled_dates`` and ``enabled_dates`` should be specified.\n    ')
    enabled_dates = Nullable(List(Either(Date, Tuple(Date, Date))), default=None, help='\n    A list of dates of ``(start, end)`` date ranges to make available for\n    selection. All other dates will be unavailable.\n\n    .. note::\n        Only one of ``disabled_dates`` and ``enabled_dates`` should be specified.\n    ')
    date_format = String(default='Y-m-d', help='\n    Formatting specification for the display of the picked date.\n\n    +---+-----------------------------------------------------------+-----------------------------------------+\n    | d | Day of the month, 2 digits with leading zeros             | 01 to 31                                |\n    | D | A textual representation of a day                         | Mon through Sun                         |\n    | l | A full textual representation of the day of the week      | Sunday through Saturday                 |\n    | j | Day of the month without leading zeros                    | 1 to 31                                 |\n    | J | Day of the month without leading zeros and ordinal suffix | 1st, 2nd, to 31st                       |\n    | w | Numeric representation of the day of the week             | 0 (for Sunday) through 6 (for Saturday) |\n    | W | Numeric representation of the week                        | 0 through 52                            |\n    | F | A full textual representation of a month                  | January through December                |\n    | m | Numeric representation of a month, with leading zero      | 01 through 12                           |\n    | n | Numeric representation of a month, without leading zeros  | 1 through 12                            |\n    | M | A short textual representation of a month                 | Jan through Dec                         |\n    | U | The number of seconds since the Unix Epoch                | 1413704993                              |\n    | y | A two digit representation of a year                      | 99 or 03                                |\n    | Y | A full numeric representation of a year, 4 digits         | 1999 or 2003                            |\n    | Z | ISO Date format                                           | 2017-03-04T01:23:43.000Z                |\n    +---+-----------------------------------------------------------+-----------------------------------------+\n\n    See also https://flatpickr.js.org/formatting/#date-formatting-tokens.\n    ')

@abstract
class BaseDatePicker(PickerBase, DateCommon):
    """ Bases for various calendar-based date picker widgets.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    min_date = Nullable(Date, default=None, help='\n    Optional earliest allowable date.\n    ')
    max_date = Nullable(Date, default=None, help='\n    Optional latest allowable date.\n    ')

class DatePicker(BaseDatePicker):
    """ Calendar-based date picker widget.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    value = Nullable(Date, default=None, help='\n    The initial or picked date.\n    ')

class DateRangePicker(BaseDatePicker):
    """ Calendar-based picker of date ranges. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    value = Nullable(Tuple(Date, Date), default=None, help='\n    The initial or picked date range.\n    ')

class MultipleDatePicker(BaseDatePicker):
    """ Calendar-based picker of dates. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    value = List(Date, default=[], help='\n    The initial or picked dates.\n    ')
    separator = String(default=', ', help='\n    The separator between displayed dates.\n    ')

@abstract
class BaseDatetimePicker(PickerBase, DateCommon, TimeCommon):
    """ Bases for various calendar-based datetime picker widgets.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    min_date = Nullable(Either(Datetime, Date), default=None, help='\n    Optional earliest allowable date and time.\n    ')
    max_date = Nullable(Either(Datetime, Date), default=None, help='\n    Optional latest allowable date and time.\n    ')
    date_format = Override(default='Y-m-d H:i')

class DatetimePicker(BaseDatetimePicker):
    """ Calendar-based date and time picker widget.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    value = Nullable(Datetime, default=None, help='\n    The initial or picked date and time.\n    ')

class DatetimeRangePicker(BaseDatetimePicker):
    """ Calendar-based picker of date and time ranges. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    value = Nullable(Tuple(Datetime, Datetime), default=None, help='\n    The initial or picked date and time range.\n    ')

class MultipleDatetimePicker(BaseDatetimePicker):
    """ Calendar-based picker of dates and times. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    value = List(Datetime, default=[], help='\n    The initial or picked dates and times.\n    ')
    separator = String(default=', ', help='\n    The separator between displayed dates and times.\n    ')