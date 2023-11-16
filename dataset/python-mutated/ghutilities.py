"""This file contains a list of utilities for working with GitHub data."""
from datetime import datetime
import re
GITHUB_DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%SZ'

def datetimeFromGHTimeStr(text):
    if False:
        i = 10
        return i + 15
    'Parse GitHub time format into datetime structure.'
    return datetime.strptime(text, GITHUB_DATETIME_FORMAT)

def datetimeToGHTimeStr(timestamp):
    if False:
        return 10
    'Convert datetime to GitHub datetime string'
    return timestamp.strftime(GITHUB_DATETIME_FORMAT)

def findMentions(text):
    if False:
        print('Hello World!')
    'Returns all mentions in text. Skips "username".'
    matches = re.findall('@(\\w+)', text)
    return list(filter(lambda x: x != 'username' and x != '', matches))