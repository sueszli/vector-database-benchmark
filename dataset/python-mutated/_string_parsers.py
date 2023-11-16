import datetime
import re

class Frequencies:

    @staticmethod
    def hourly(t):
        if False:
            return 10
        dt = t + datetime.timedelta(hours=1)
        return dt.replace(minute=0, second=0, microsecond=0)

    @staticmethod
    def daily(t):
        if False:
            for i in range(10):
                print('nop')
        dt = t + datetime.timedelta(days=1)
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def weekly(t):
        if False:
            print('Hello World!')
        dt = t + datetime.timedelta(days=7 - t.weekday())
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def monthly(t):
        if False:
            return 10
        if t.month == 12:
            (y, m) = (t.year + 1, 1)
        else:
            (y, m) = (t.year, t.month + 1)
        return t.replace(year=y, month=m, day=1, hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def yearly(t):
        if False:
            print('Hello World!')
        y = t.year + 1
        return t.replace(year=y, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

def parse_size(size):
    if False:
        while True:
            i = 10
    size = size.strip()
    reg = re.compile('([e\\+\\-\\.\\d]+)\\s*([kmgtpezy])?(i)?(b)', flags=re.I)
    match = reg.fullmatch(size)
    if not match:
        return None
    (s, u, i, b) = match.groups()
    try:
        s = float(s)
    except ValueError as e:
        raise ValueError("Invalid float value while parsing size: '%s'" % s) from e
    u = 'kmgtpezy'.index(u.lower()) + 1 if u else 0
    i = 1024 if i else 1000
    b = {'b': 8, 'B': 1}[b] if b else 1
    size = s * i ** u / b
    return size

def parse_duration(duration):
    if False:
        while True:
            i = 10
    duration = duration.strip()
    reg = '(?:([e\\+\\-\\.\\d]+)\\s*([a-z]+)[\\s\\,]*)'
    units = [('y|years?', 31536000), ('months?', 2628000), ('w|weeks?', 604800), ('d|days?', 86400), ('h|hours?', 3600), ('min(?:ute)?s?', 60), ('s|sec(?:ond)?s?', 1), ('ms|milliseconds?', 0.001), ('us|microseconds?', 1e-06)]
    if not re.fullmatch(reg + '+', duration, flags=re.I):
        return None
    seconds = 0
    for (value, unit) in re.findall(reg, duration, flags=re.I):
        try:
            value = float(value)
        except ValueError as e:
            raise ValueError("Invalid float value while parsing duration: '%s'" % value) from e
        try:
            unit = next((u for (r, u) in units if re.fullmatch(r, unit, flags=re.I)))
        except StopIteration:
            raise ValueError("Invalid unit value while parsing duration: '%s'" % unit) from None
        seconds += value * unit
    return datetime.timedelta(seconds=seconds)

def parse_frequency(frequency):
    if False:
        while True:
            i = 10
    frequencies = {'hourly': Frequencies.hourly, 'daily': Frequencies.daily, 'weekly': Frequencies.weekly, 'monthly': Frequencies.monthly, 'yearly': Frequencies.yearly}
    frequency = frequency.strip().lower()
    return frequencies.get(frequency, None)

def parse_day(day):
    if False:
        for i in range(10):
            print('nop')
    days = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6}
    day = day.strip().lower()
    if day in days:
        return days[day]
    elif day.startswith('w') and day[1:].isdigit():
        day = int(day[1:])
        if not 0 <= day < 7:
            raise ValueError("Invalid weekday value while parsing day (expected [0-6]): '%d'" % day)
    else:
        day = None
    return day

def parse_time(time):
    if False:
        for i in range(10):
            print('nop')
    time = time.strip()
    reg = re.compile('^[\\d\\.\\:]+\\s*(?:[ap]m)?$', flags=re.I)
    if not reg.match(time):
        return None
    formats = ['%H', '%H:%M', '%H:%M:%S', '%H:%M:%S.%f', '%I %p', '%I:%M %S', '%I:%M:%S %p', '%I:%M:%S.%f %p']
    for format_ in formats:
        try:
            dt = datetime.datetime.strptime(time, format_)
        except ValueError:
            pass
        else:
            return dt.time()
    raise ValueError("Unrecognized format while parsing time: '%s'" % time)

def parse_daytime(daytime):
    if False:
        while True:
            i = 10
    daytime = daytime.strip()
    reg = re.compile('^(.*?)\\s+at\\s+(.*)$', flags=re.I)
    match = reg.match(daytime)
    if match:
        (day, time) = match.groups()
    else:
        day = time = daytime
    try:
        day = parse_day(day)
        if match and day is None:
            raise ValueError
    except ValueError as e:
        raise ValueError("Invalid day while parsing daytime: '%s'" % day) from e
    try:
        time = parse_time(time)
        if match and time is None:
            raise ValueError
    except ValueError as e:
        raise ValueError("Invalid time while parsing daytime: '%s'" % time) from e
    if day is None and time is None:
        return None
    return (day, time)