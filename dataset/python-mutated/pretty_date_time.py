from __future__ import print_function
from __future__ import division
from datetime import datetime

def pretty_date_time(date_time):
    if False:
        while True:
            i = 10
    "Print a pretty datetime similar to what's seen on Hacker News.\n\n    Gets a datetime object or a int() Epoch timestamp and return a\n    pretty string like 'an hour ago', 'Yesterday', '3 months ago',\n    'just now', etc.\n\n    Adapted from: http://stackoverflow.com/questions/1551382/user-friendly-time-format-in-python  # NOQA\n\n    :type foo: :class:`datetime.datetime`\n    :param foo: An instance of `datetime.datetime`.\n\n    :rtype: str\n    :return: the pretty datetime.\n    "
    now = datetime.now()
    if type(date_time) is int:
        diff = now - datetime.fromtimestamp(date_time)
    elif isinstance(date_time, datetime):
        diff = now - date_time
    elif not date_time:
        diff = now - now
    second_diff = diff.seconds
    day_diff = diff.days
    if day_diff < 0:
        return ''
    if day_diff == 0:
        if second_diff < 10:
            return 'just now'
        if second_diff < 60:
            return str(second_diff) + ' seconds ago'
        if second_diff < 120:
            return '1 minute ago'
        if second_diff < 3600:
            return str(second_diff // 60) + ' minutes ago'
        if second_diff < 7200:
            return '1 hour ago'
        if second_diff < 86400:
            return str(second_diff // 3600) + ' hours ago'
    if day_diff == 1:
        return 'Yesterday'
    if day_diff < 7:
        return str(day_diff) + ' days ago'
    if day_diff < 31:
        return str(day_diff // 7) + ' week(s) ago'
    if day_diff < 365:
        return str(day_diff // 30) + ' month(s) ago'
    return str(day_diff // 365) + ' year(s) ago'