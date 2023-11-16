"""Factory for HolidayCalendar implementations."""
import numpy as np
import tensorflow.compat.v2 as tf
from tf_quant_finance.datetime import bounded_holiday_calendar
from tf_quant_finance.datetime import unbounded_holiday_calendar

def create_holiday_calendar(weekend_mask=None, holidays=None, start_year=None, end_year=None):
    if False:
        return 10
    'Creates a holiday calendar.\n\n  Each instance should be used in the context of only one graph. E.g. one can\'t\n  create a HolidayCalendar in one tf.function and reuse it in another.\n\n  Note: providing bounds for the calendar, i.e. `holidays` and/or `start_year`,\n  `end_year` yields a better-performing calendar.\n\n  Args:\n    weekend_mask: Boolean `Tensor` of 7 elements one for each day of the week\n      starting with Monday at index 0. A `True` value indicates the day is\n      considered a weekend day and a `False` value implies a week day.\n      Default value: None which means no weekends are applied.\n    holidays: Defines the holidays that are added to the weekends defined by\n      `weekend_mask`. An instance of `dates.DateTensor` or an object\n      convertible to `DateTensor`.\n      Default value: None which means no holidays other than those implied by\n      the weekends (if any).\n      Note that it is necessary to provide holidays for each year, and also\n      adjust the holidays that fall on the weekends if required, e.g.\n      2021-12-25 to 2021-12-24. To avoid doing this manually one can use\n      AbstractHolidayCalendar from Pandas:\n\n      ```python\n      from pandas.tseries.holiday import AbstractHolidayCalendar\n      from pandas.tseries.holiday import Holiday\n      from pandas.tseries.holiday import nearest_workday\n\n      class MyCalendar(AbstractHolidayCalendar):\n          rules = [\n              Holiday(\'NewYear\', month=1, day=1, observance=nearest_workday),\n              Holiday(\'Christmas\', month=12, day=25,\n                       observance=nearest_workday)\n          ]\n\n      calendar = MyCalendar()\n      holidays_index = calendar.holidays(\n          start=datetime.date(2020, 1, 1),\n          end=datetime.date(2030, 12, 31))\n      holidays = np.array(holidays_index.to_pydatetime(), dtype="<M8[D]")\n      ```\n\n    start_year: Integer giving the earliest year this calendar includes. If\n      `holidays` is specified, then `start_year` and `end_year` are ignored,\n      and the boundaries are derived from `holidays`.\n      Default value: None which means start year is inferred from `holidays`, if\n      present.\n    end_year: Integer giving the latest year this calendar includes. If\n      `holidays` is specified, then `start_year` and `end_year` are ignored,\n      and the boundaries are derived from `holidays`.\n      Default value: None which means start year is inferred from `holidays`, if\n      present.\n\n  Returns:\n    A HolidayCalendar instance.\n  '
    is_bounded = _tensor_is_not_empty(holidays) or (start_year is not None and end_year is not None)
    if is_bounded:
        return bounded_holiday_calendar.BoundedHolidayCalendar(weekend_mask, holidays, start_year, end_year)
    return unbounded_holiday_calendar.UnboundedHolidayCalendar(weekend_mask, holidays)

def _tensor_is_not_empty(t):
    if False:
        return 10
    'Returns whether t is definitely not empty.'
    if t is None:
        return False
    if isinstance(t, np.ndarray):
        return t.size > 0
    if isinstance(t, tf.Tensor):
        num_elem = t.shape.num_elements
        return num_elem is not None and num_elem > 0
    return bool(t)