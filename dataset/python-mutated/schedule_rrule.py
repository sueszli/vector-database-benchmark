from __future__ import absolute_import, division, print_function
__metaclass__ = type
DOCUMENTATION = "\n    name: schedule_rrule\n    author: John Westcott IV (@john-westcott-iv)\n    short_description: Generate an rrule string which can be used for Schedules\n    requirements:\n      - pytz\n      - python-dateutil >= 2.7.0\n    description:\n      - Returns a string based on criteria which represents an rrule\n    options:\n      _terms:\n        description:\n          - The frequency of the schedule\n          - none - Run this schedule once\n          - minute - Run this schedule every x minutes\n          - hour - Run this schedule every x hours\n          - day - Run this schedule every x days\n          - week - Run this schedule weekly\n          - month - Run this schedule monthly\n        required: True\n        choices: ['none', 'minute', 'hour', 'day', 'week', 'month']\n      start_date:\n        description:\n          - The date to start the rule\n          - Used for all frequencies\n          - Format should be YYYY-MM-DD [HH:MM:SS]\n        type: str\n      timezone:\n        description:\n          - The timezone to use for this rule\n          - Used for all frequencies\n          - Format should be as US/Eastern\n          - Defaults to America/New_York\n        type: str\n      every:\n        description:\n          - The repetition in months, weeks, days hours or minutes\n          - Used for all types except none\n        type: int\n      end_on:\n        description:\n          - How to end this schedule\n          - If this is not defined, this schedule will never end\n          - If this is a positive integer, this schedule will end after this number of occurences\n          - If this is a date in the format YYYY-MM-DD [HH:MM:SS], this schedule ends after this date\n          - Used for all types except none\n        type: str\n      on_days:\n        description:\n          - The days to run this schedule on\n          - A comma-separated list which can contain values sunday, monday, tuesday, wednesday, thursday, friday\n          - Used for week type schedules\n      month_day_number:\n        description:\n          - The day of the month this schedule will run on (0-31)\n          - Used for month type schedules\n          - Cannot be used with on_the parameter\n        type: int\n      on_the:\n        description:\n          - A description on when this schedule will run\n          - Two strings separated by a space\n          - First string is one of first, second, third, fourth, last\n          - Second string is one of sunday, monday, tuesday, wednesday, thursday, friday\n          - Used for month type schedules\n          - Cannot be used with month_day_number parameters\n"
EXAMPLES = '\n    - name: Create a string for a schedule\n      debug:\n        msg: "{{ query(\'awx.awx.schedule_rrule\', \'none\', start_date=\'1979-09-13 03:45:07\') }}"\n'
RETURN = '\n_raw:\n  description:\n    - String in the rrule format\n  type: string\n'
import re
from ansible.module_utils.six import raise_from
from ansible.plugins.lookup import LookupBase
from ansible.errors import AnsibleError
from datetime import datetime
try:
    import pytz
    from dateutil import rrule
except ImportError as imp_exc:
    LIBRARY_IMPORT_ERROR = imp_exc
else:
    LIBRARY_IMPORT_ERROR = None

class LookupModule(LookupBase):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if LIBRARY_IMPORT_ERROR:
            raise_from(AnsibleError('{0}'.format(LIBRARY_IMPORT_ERROR)), LIBRARY_IMPORT_ERROR)
        super().__init__(*args, **kwargs)
        self.frequencies = {'none': rrule.DAILY, 'minute': rrule.MINUTELY, 'hour': rrule.HOURLY, 'day': rrule.DAILY, 'week': rrule.WEEKLY, 'month': rrule.MONTHLY}
        self.weekdays = {'monday': rrule.MO, 'tuesday': rrule.TU, 'wednesday': rrule.WE, 'thursday': rrule.TH, 'friday': rrule.FR, 'saturday': rrule.SA, 'sunday': rrule.SU}
        self.set_positions = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'last': -1}

    @staticmethod
    def parse_date_time(date_string):
        if False:
            print('Hello World!')
        try:
            return datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return datetime.strptime(date_string, '%Y-%m-%d')

    def run(self, terms, variables=None, **kwargs):
        if False:
            while True:
                i = 10
        if len(terms) != 1:
            raise AnsibleError('You may only pass one schedule type in at a time')
        frequency = terms[0].lower()
        return self.get_rrule(frequency, kwargs)

    def get_rrule(self, frequency, kwargs):
        if False:
            i = 10
            return i + 15
        if frequency not in self.frequencies:
            raise AnsibleError('Frequency of {0} is invalid'.format(frequency))
        rrule_kwargs = {'freq': self.frequencies[frequency], 'interval': kwargs.get('every', 1)}
        if 'start_date' in kwargs:
            try:
                rrule_kwargs['dtstart'] = LookupModule.parse_date_time(kwargs['start_date'])
            except Exception as e:
                raise_from(AnsibleError('Parameter start_date must be in the format YYYY-MM-DD [HH:MM:SS]'), e)
        if frequency == 'none':
            rrule_kwargs['count'] = 1
        else:
            if 'end_on' in kwargs:
                end_on = kwargs['end_on']
                if re.match('^\\d+$', end_on):
                    rrule_kwargs['count'] = end_on
                else:
                    try:
                        rrule_kwargs['until'] = LookupModule.parse_date_time(end_on)
                    except Exception as e:
                        raise_from(AnsibleError('Parameter end_on must either be an integer or in the format YYYY-MM-DD [HH:MM:SS]'), e)
            if frequency == 'week' and 'on_days' in kwargs:
                days = []
                for day in kwargs['on_days'].split(','):
                    day = day.strip()
                    if day not in self.weekdays:
                        raise AnsibleError('Parameter on_days must only contain values {0}'.format(', '.join(self.weekdays.keys())))
                    days.append(self.weekdays[day])
                rrule_kwargs['byweekday'] = days
            if frequency == 'month':
                if 'month_day_number' in kwargs and 'on_the' in kwargs:
                    raise AnsibleError('Month based frequencies can have month_day_number or on_the but not both')
                if 'month_day_number' in kwargs:
                    try:
                        my_month_day = int(kwargs['month_day_number'])
                        if my_month_day < 1 or my_month_day > 31:
                            raise Exception()
                    except Exception as e:
                        raise_from(AnsibleError('month_day_number must be between 1 and 31'), e)
                    rrule_kwargs['bymonthday'] = my_month_day
                if 'on_the' in kwargs:
                    try:
                        (occurance, weekday) = kwargs['on_the'].split(' ')
                    except Exception as e:
                        raise_from(AnsibleError('on_the parameter must be two words separated by a space'), e)
                    if weekday not in self.weekdays:
                        raise AnsibleError('Weekday portion of on_the parameter is not valid')
                    if occurance not in self.set_positions:
                        raise AnsibleError('The first string of the on_the parameter is not valid')
                    rrule_kwargs['byweekday'] = self.weekdays[weekday]
                    rrule_kwargs['bysetpos'] = self.set_positions[occurance]
        my_rule = rrule.rrule(**rrule_kwargs)
        timezone = 'America/New_York'
        if 'timezone' in kwargs:
            if kwargs['timezone'] not in pytz.all_timezones:
                raise AnsibleError('Timezone parameter is not valid')
            timezone = kwargs['timezone']
        return_rrule = str(my_rule).replace('\n', ' ').replace('DTSTART:', 'DTSTART;TZID={0}:'.format(timezone))
        if kwargs.get('every', 1) == 1:
            return_rrule = '{0};INTERVAL=1'.format(return_rrule)
        return return_rrule