from datetime import datetime

def get_time_difference(timestamp1, timestamp2):
    if False:
        while True:
            i = 10
    time_format = '%Y-%m-%d %H:%M:%S.%f'
    parsed_timestamp1 = datetime.strptime(str(timestamp1), time_format)
    parsed_timestamp2 = datetime.strptime(timestamp2, time_format)
    time_difference = parsed_timestamp2 - parsed_timestamp1
    total_seconds = int(time_difference.total_seconds())
    (years, seconds_remainder) = divmod(total_seconds, 365 * 24 * 60 * 60)
    (months, seconds_remainder) = divmod(seconds_remainder, 30 * 24 * 60 * 60)
    (days, seconds_remainder) = divmod(seconds_remainder, 24 * 60 * 60)
    (hours, seconds_remainder) = divmod(seconds_remainder, 60 * 60)
    (minutes, _) = divmod(seconds_remainder, 60)
    time_difference_dict = {'years': years, 'months': months, 'days': days, 'hours': hours, 'minutes': minutes}
    return time_difference_dict

def parse_interval_to_seconds(interval: str) -> int:
    if False:
        print('Hello World!')
    units = {'Minutes': 60, 'Hours': 3600, 'Days': 86400, 'Weeks': 604800, 'Months': 2592000}
    interval = ' '.join(interval.split())
    (value, unit) = interval.split(' ')
    return int(value) * units[unit]