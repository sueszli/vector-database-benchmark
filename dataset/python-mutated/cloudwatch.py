from . import exactly_one, json_checker

def dict_or_string(x):
    if False:
        print('Hello World!')
    '\n    Property: Dashboard.DashboardBody\n    '
    if isinstance(x, (dict, str)):
        return x
    raise TypeError(f'Value {x} of type {type(x)} must be either dict or str')

def validate_unit(unit):
    if False:
        while True:
            i = 10
    '\n    Validate Units\n    Property: MetricStat.Unit\n    '
    VALID_UNITS = ('Seconds', 'Microseconds', 'Milliseconds', 'Bytes', 'Kilobytes', 'Megabytes', 'Gigabytes', 'Terabytes', 'Bits', 'Kilobits', 'Megabits', 'Gigabits', 'Terabits', 'Percent', 'Count', 'Bytes/Second', 'Kilobytes/Second', 'Megabytes/Second', 'Gigabytes/Second', 'Terabytes/Second', 'Bits/Second', 'Kilobits/Second', 'Megabits/Second', 'Gigabits/Second', 'Terabits/Second', 'Count/Second', 'None')
    if unit not in VALID_UNITS:
        raise ValueError('MetricStat Unit must be one of: %s' % ', '.join(VALID_UNITS))
    return unit

def validate_treat_missing_data(value):
    if False:
        i = 10
        return i + 15
    '\n    Validate TreatMissingData\n    Property: Alarm.TreatMissingData\n    '
    VALID_TREAT_MISSING_DATA_TYPES = ('breaching', 'notBreaching', 'ignore', 'missing')
    if value not in VALID_TREAT_MISSING_DATA_TYPES:
        raise ValueError('Alarm TreatMissingData must be one of: %s' % ', '.join(VALID_TREAT_MISSING_DATA_TYPES))
    return value

def validate_alarm(self):
    if False:
        print('Hello World!')
    '\n    Class: Alarm\n    '
    conds = ['ExtendedStatistic', 'Metrics', 'Statistic']
    exactly_one(self.__class__.__name__, self.properties, conds)

def validate_dashboard(self):
    if False:
        return 10
    '\n    Class: Dashboard\n    '
    name = 'DashboardBody'
    if name in self.properties:
        dashboard_body = self.properties.get(name)
        self.properties[name] = json_checker(dashboard_body)