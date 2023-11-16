import pandas as pd
from dateutil.relativedelta import relativedelta

class Timedelta(object):
    """Represents differences in time.

    Timedeltas can be defined in multiple units. Supported units:

    - "ms" : milliseconds
    - "s" : seconds
    - "h" : hours
    - "m" : minutes
    - "d" : days
    - "o"/"observations" : number of individual events
    - "mo" : months
    - "Y" : years

    Timedeltas can also be defined in terms of observations. In this case, the
    Timedelta represents the period spanned by `value`.

    For observation timedeltas:
    >>> three_observations_log = Timedelta(3, "observations")
    >>> three_observations_log.get_name()
    '3 Observations'
    """
    _Observations = 'o'
    _absolute_units = ['ms', 's', 'h', 'm', 'd', 'w']
    _relative_units = ['mo', 'Y']
    _readable_units = {'ms': 'Milliseconds', 's': 'Seconds', 'h': 'Hours', 'm': 'Minutes', 'd': 'Days', 'o': 'Observations', 'w': 'Weeks', 'Y': 'Years', 'mo': 'Months'}
    _readable_to_unit = {v.lower(): k for (k, v) in _readable_units.items()}

    def __init__(self, value, unit=None, delta_obj=None):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            value (float, str, dict) : Value of timedelta, string providing\n                both unit and value, or a dictionary of units and times.\n            unit (str) : Unit of time delta.\n            delta_obj (pd.Timedelta or pd.DateOffset) : A time object used\n                internally to do time operations. If None is provided, one will\n                be created using the provided value and unit.\n        '
        self.check_value(value, unit)
        self.times = self.fix_units()
        if delta_obj is not None:
            self.delta_obj = delta_obj
        else:
            self.delta_obj = self.get_unit_type()

    @classmethod
    def from_dictionary(cls, dictionary):
        if False:
            for i in range(10):
                print('nop')
        dict_units = dictionary['unit']
        dict_values = dictionary['value']
        if isinstance(dict_units, str) and isinstance(dict_values, (int, float)):
            return cls({dict_units: dict_values})
        else:
            all_units = dict()
            for i in range(len(dict_units)):
                all_units[dict_units[i]] = dict_values[i]
            return cls(all_units)

    @classmethod
    def make_singular(cls, s):
        if False:
            return 10
        if len(s) > 1 and s.endswith('s'):
            return s[:-1]
        return s

    @classmethod
    def _check_unit_plural(cls, s):
        if False:
            for i in range(10):
                print('nop')
        if len(s) > 2 and (not s.endswith('s')):
            return (s + 's').lower()
        elif len(s) > 1:
            return s.lower()
        return s

    def get_value(self, unit=None):
        if False:
            while True:
                i = 10
        if unit is not None:
            return self.times[unit]
        elif len(self.times.values()) == 1:
            return list(self.times.values())[0]
        else:
            return self.times

    def get_units(self):
        if False:
            while True:
                i = 10
        return list(self.times.keys())

    def get_unit_type(self):
        if False:
            print('Hello World!')
        all_units = self.get_units()
        if self._Observations in all_units:
            return None
        elif self.is_absolute() and self.has_multiple_units() is False:
            return pd.Timedelta(self.times[all_units[0]], all_units[0])
        else:
            readable_times = self.lower_readable_times()
            return relativedelta(**readable_times)

    def check_value(self, value, unit):
        if False:
            while True:
                i = 10
        if isinstance(value, str):
            from featuretools.utils.wrangle import _check_timedelta
            td = _check_timedelta(value)
            self.times = td.times
        elif isinstance(value, dict):
            self.times = value
        else:
            self.times = {unit: value}

    def fix_units(self):
        if False:
            for i in range(10):
                print('nop')
        fixed_units = dict()
        for (unit, value) in self.times.items():
            unit = self._check_unit_plural(unit)
            if unit in self._readable_to_unit:
                unit = self._readable_to_unit[unit]
            fixed_units[unit] = value
        return fixed_units

    def lower_readable_times(self):
        if False:
            while True:
                i = 10
        readable_times = dict()
        for (unit, value) in self.times.items():
            readable_unit = self._readable_units[unit].lower()
            readable_times[readable_unit] = value
        return readable_times

    def get_name(self):
        if False:
            return 10
        all_units = self.get_units()
        if self.has_multiple_units() is False:
            return '{} {}'.format(self.times[all_units[0]], self._readable_units[all_units[0]])
        final_str = ''
        for (unit, value) in self.times.items():
            if value == 1:
                unit = self.make_singular(unit)
            final_str += '{} {} '.format(value, self._readable_units[unit])
        return final_str[:-1]

    def get_arguments(self):
        if False:
            for i in range(10):
                print('nop')
        units = list()
        values = list()
        for (unit, value) in self.times.items():
            units.append(unit)
            values.append(value)
        if len(units) == 1:
            return {'unit': units[0], 'value': values[0]}
        else:
            return {'unit': units, 'value': values}

    def is_absolute(self):
        if False:
            for i in range(10):
                print('nop')
        for unit in self.get_units():
            if unit not in self._absolute_units:
                return False
        return True

    def has_no_observations(self):
        if False:
            i = 10
            return i + 15
        for unit in self.get_units():
            if unit in self._Observations:
                return False
        return True

    def has_multiple_units(self):
        if False:
            while True:
                i = 10
        if len(self.get_units()) > 1:
            return True
        else:
            return False

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, Timedelta):
            return False
        return self.times == other.times

    def __neg__(self):
        if False:
            return 10
        'Negate the timedelta'
        new_times = dict()
        for (unit, value) in self.times.items():
            new_times[unit] = -value
        if self.delta_obj is not None:
            return Timedelta(new_times, delta_obj=-self.delta_obj)
        else:
            return Timedelta(new_times)

    def __radd__(self, time):
        if False:
            while True:
                i = 10
        'Add the Timedelta to a timestamp value'
        if self._Observations not in self.get_units():
            return time + self.delta_obj
        else:
            raise Exception('Invalid unit')

    def __rsub__(self, time):
        if False:
            i = 10
            return i + 15
        'Subtract the Timedelta from a timestamp value'
        if self._Observations not in self.get_units():
            return time - self.delta_obj
        else:
            raise Exception('Invalid unit')