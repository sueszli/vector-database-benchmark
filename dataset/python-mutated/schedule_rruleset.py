from __future__ import absolute_import, division, print_function
__metaclass__ = type
DOCUMENTATION = "\n    name: schedule_rruleset\n    author: John Westcott IV (@john-westcott-iv)\n    short_description: Generate an rruleset string\n    requirements:\n      - pytz\n      - python-dateutil >= 2.7.0\n    description:\n      - Returns a string based on criteria which represents an rrule\n    options:\n      _terms:\n        description:\n          - The start date of the ruleset\n          - Used for all frequencies\n          - Format should be YYYY-MM-DD [HH:MM:SS]\n        required: True\n        type: str\n      timezone:\n        description:\n          - The timezone to use for this rule\n          - Used for all frequencies\n          - Format should be as US/Eastern\n          - Defaults to America/New_York\n        type: str\n      rules:\n        description:\n          - Array of rules in the rruleset\n        type: list\n        elements: dict\n        required: True\n        suboptions:\n          frequency:\n            description:\n              - The frequency of the schedule\n              - none - Run this schedule once\n              - minute - Run this schedule every x minutes\n              - hour - Run this schedule every x hours\n              - day - Run this schedule every x days\n              - week - Run this schedule weekly\n              - month - Run this schedule monthly\n            required: True\n            choices: ['none', 'minute', 'hour', 'day', 'week', 'month']\n          interval:\n            description:\n              - The repetition in months, weeks, days hours or minutes\n              - Used for all types except none\n            type: int\n          end_on:\n            description:\n              - How to end this schedule\n              - If this is not defined, this schedule will never end\n              - If this is a positive integer, this schedule will end after this number of occurrences\n              - If this is a date in the format YYYY-MM-DD [HH:MM:SS], this schedule ends after this date\n              - Used for all types except none\n            type: str\n          bysetpos:\n            description:\n              - Specify an occurrence number, corresponding to the nth occurrence of the rule inside the frequency period.\n              - A comma-separated list of positions (first, second, third, forth or last)\n            type: string\n          bymonth:\n            description:\n              - The months this schedule will run on\n              - A comma-separated list which can contain values 0-12\n            type: string\n          bymonthday:\n            description:\n              - The day of the month this schedule will run on\n              - A comma-separated list which can contain values 0-31\n            type: string\n          byyearday:\n            description:\n              - The year day numbers to run this schedule on\n              - A comma-separated list which can contain values 0-366\n            type: string\n          byweekno:\n            description:\n              - The week numbers to run this schedule on\n              - A comma-separated list which can contain values as described in ISO8601\n            type: string\n          byweekday:\n            description:\n              - The days to run this schedule on\n              - A comma-separated list which can contain values sunday, monday, tuesday, wednesday, thursday, friday\n            type: string\n          byhour:\n            description:\n              - The hours to run this schedule on\n              - A comma-separated list which can contain values 0-23\n            type: string\n          byminute:\n            description:\n              - The minutes to run this schedule on\n              - A comma-separated list which can contain values 0-59\n            type: string\n          include:\n            description:\n              - If this rule should be included (RRULE) or excluded (EXRULE)\n            type: bool\n            default: True\n"
EXAMPLES = '\n    - name: Create a ruleset for everyday except Sundays\n      set_fact:\n        complex_rule: "{{ query(awx.awx.schedule_rruleset, \'2022-04-30 10:30:45\', rules=rrules, timezone=\'UTC\' ) }}"\n      vars:\n        rrules:\n          - frequency: \'day\'\n            interval: 1\n          - frequency: \'day\'\n            interval: 1\n            byweekday: \'sunday\'\n            include: False\n'
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
            for i in range(10):
                print('nop')
        if LIBRARY_IMPORT_ERROR:
            raise_from(AnsibleError('{0}'.format(LIBRARY_IMPORT_ERROR)), LIBRARY_IMPORT_ERROR)
        super().__init__(*args, **kwargs)
        self.frequencies = {'none': rrule.DAILY, 'minute': rrule.MINUTELY, 'hour': rrule.HOURLY, 'day': rrule.DAILY, 'week': rrule.WEEKLY, 'month': rrule.MONTHLY}
        self.weekdays = {'monday': rrule.MO, 'tuesday': rrule.TU, 'wednesday': rrule.WE, 'thursday': rrule.TH, 'friday': rrule.FR, 'saturday': rrule.SA, 'sunday': rrule.SU}
        self.set_positions = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'last': -1}

    @staticmethod
    def parse_date_time(date_string):
        if False:
            i = 10
            return i + 15
        try:
            return datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return datetime.strptime(date_string, '%Y-%m-%d')

    def process_integer(self, field_name, rule, min_value, max_value, rule_number):
        if False:
            i = 10
            return i + 15
        return_values = []
        if isinstance(rule[field_name], int):
            rule[field_name] = [rule[field_name]]
        if not isinstance(rule[field_name], list):
            rule[field_name] = rule[field_name].split(',')
        for value in rule[field_name]:
            if isinstance(value, str):
                value = value.strip()
            if not re.match('^\\d+$', str(value)) or int(value) < min_value or int(value) > max_value:
                raise AnsibleError('In rule {0} {1} must be between {2} and {3}'.format(rule_number, field_name, min_value, max_value))
            return_values.append(int(value))
        return return_values

    def process_list(self, field_name, rule, valid_list, rule_number):
        if False:
            print('Hello World!')
        return_values = []
        if not isinstance(rule[field_name], list):
            rule[field_name] = rule[field_name].split(',')
        for value in rule[field_name]:
            value = value.strip().lower()
            if value not in valid_list:
                raise AnsibleError('In rule {0} {1} must only contain values in {2}'.format(rule_number, field_name, ', '.join(valid_list.keys())))
            return_values.append(valid_list[value])
        return return_values

    def run(self, terms, variables=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if len(terms) != 1:
            raise AnsibleError('You may only pass one schedule type in at a time')
        try:
            start_date = LookupModule.parse_date_time(terms[0])
        except Exception as e:
            raise_from(AnsibleError('The start date must be in the format YYYY-MM-DD [HH:MM:SS]'), e)
        if not kwargs.get('rules', None):
            raise AnsibleError('You must include rules to be in the ruleset via the rules parameter')
        timezone = 'America/New_York'
        if 'timezone' in kwargs:
            if kwargs['timezone'] not in pytz.all_timezones:
                raise AnsibleError('Timezone parameter is not valid')
            timezone = kwargs['timezone']
        rules = []
        got_at_least_one_rule = False
        for rule_index in range(0, len(kwargs['rules'])):
            rule = kwargs['rules'][rule_index]
            rule_number = rule_index + 1
            valid_options = ['frequency', 'interval', 'end_on', 'bysetpos', 'bymonth', 'bymonthday', 'byyearday', 'byweekno', 'byweekday', 'byhour', 'byminute', 'include']
            invalid_options = list(set(rule.keys()) - set(valid_options))
            if invalid_options:
                raise AnsibleError('Rule {0} has invalid options: {1}'.format(rule_number, ', '.join(invalid_options)))
            frequency = rule.get('frequency', None)
            if not frequency:
                raise AnsibleError('Rule {0} is missing a frequency'.format(rule_number))
            if frequency not in self.frequencies:
                raise AnsibleError('Frequency of rule {0} is invalid {1}'.format(rule_number, frequency))
            rrule_kwargs = {'freq': self.frequencies[frequency], 'interval': rule.get('interval', 1), 'dtstart': start_date}
            if frequency == 'none':
                rrule_kwargs['count'] = 1
            elif 'end_on' in rule:
                end_on = rule['end_on']
                if re.match('^\\d+$', end_on):
                    rrule_kwargs['count'] = end_on
                else:
                    try:
                        rrule_kwargs['until'] = LookupModule.parse_date_time(end_on)
                    except Exception as e:
                        raise_from(AnsibleError('In rule {0} end_on must either be an integer or in the format YYYY-MM-DD [HH:MM:SS]'.format(rule_number)), e)
            if 'bysetpos' in rule:
                rrule_kwargs['bysetpos'] = self.process_list('bysetpos', rule, self.set_positions, rule_number)
            if 'bymonth' in rule:
                rrule_kwargs['bymonth'] = self.process_integer('bymonth', rule, 1, 12, rule_number)
            if 'bymonthday' in rule:
                rrule_kwargs['bymonthday'] = self.process_integer('bymonthday', rule, 1, 31, rule_number)
            if 'byyearday' in rule:
                rrule_kwargs['byyearday'] = self.process_integer('byyearday', rule, 1, 366, rule_number)
            if 'byweekno' in rule:
                rrule_kwargs['byweekno'] = self.process_integer('byweekno', rule, 1, 52, rule_number)
            if 'byweekday' in rule:
                rrule_kwargs['byweekday'] = self.process_list('byweekday', rule, self.weekdays, rule_number)
            if 'byhour' in rule:
                rrule_kwargs['byhour'] = self.process_integer('byhour', rule, 0, 23, rule_number)
            if 'byminute' in rule:
                rrule_kwargs['byminute'] = self.process_integer('byminute', rule, 0, 59, rule_number)
            try:
                generated_rule = str(rrule.rrule(**rrule_kwargs))
            except Exception as e:
                raise_from(AnsibleError('Failed to parse rrule for rule {0} {1}: {2}'.format(rule_number, str(rrule_kwargs), e)), e)
            if rule.get('interval', 1) == 1:
                generated_rule = '{0};INTERVAL=1'.format(generated_rule)
            if rule_index == 0:
                generated_rule = generated_rule.replace('\n', ' ').replace('DTSTART:', 'DTSTART;TZID={0}:'.format(timezone))
            else:
                generated_rule = generated_rule.split('\n')[1]
            if not rule.get('include', True):
                generated_rule = generated_rule.replace('RRULE', 'EXRULE')
            else:
                got_at_least_one_rule = True
            rules.append(generated_rule)
        if not got_at_least_one_rule:
            raise AnsibleError('A ruleset must contain at least one RRULE')
        rruleset_str = ' '.join(rules)
        try:
            rules = rrule.rrulestr(rruleset_str)
        except Exception as e:
            raise_from(AnsibleError('Failed to parse generated rule set via rruleset {0}'.format(e)), e)
        return rruleset_str