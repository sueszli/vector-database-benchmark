""" This a collection of tools for SentryReporter and SentryScrubber aimed to
simplify work with several data structures.
"""
import re
from dataclasses import dataclass
from typing import Optional
from faker import Faker
_re_search_exception = re.compile('^(\\S+)\\s*:\\s*(.+)')
_re_remove_sentry = re.compile('Sentry is attempting.*')

@dataclass
class LastCoreException:
    type: str
    message: str

def parse_last_core_output(text: str) -> Optional[LastCoreException]:
    if False:
        while True:
            i = 10
    ' This function tries to find an Exception type and the Exception message in the raw core output\n    '

    def _clean_up(s: str):
        if False:
            i = 10
            return i + 15
        return _re_remove_sentry.sub('', s).strip()
    for line in reversed(text.split('\n')):
        if (m := _re_search_exception.match(line)):
            return LastCoreException(type=_clean_up(m.group(1)), message=_clean_up(m.group(2)))
    return None

def get_first_item(items, default=None):
    if False:
        while True:
            i = 10
    return items[0] if items else default

def get_last_item(items, default=None):
    if False:
        print('Hello World!')
    return items[-1] if items else default

def delete_item(d, key):
    if False:
        i = 10
        return i + 15
    if not d:
        return d
    if key in d:
        del d[key]
    return d

def get_value(d, key, default=None):
    if False:
        return 10
    return d.get(key, default) if d else default

def extract_dict(d, regex_key_pattern):
    if False:
        while True:
            i = 10
    if not d or not regex_key_pattern:
        return dict()
    matched_keys = [key for key in d if re.match(regex_key_pattern, key)]
    return {key: d[key] for key in matched_keys}

def modify_value(d, key, function):
    if False:
        while True:
            i = 10
    if not d or not key or (not function):
        return d
    if key in d:
        d[key] = function(d[key])
    return d

def distinct_by(list_of_dict, key):
    if False:
        while True:
            i = 10
    'This function removes all duplicates from a list of dictionaries. A duplicate\n    here is a dictionary that have the same value of the given key.\n\n    If no key field is presented in the item, then the item will not be considered\n    as a duplicate.\n\n    Args:\n        list_of_dict: list of dictionaries\n        key: a field key that will be used for items comparison\n\n    Returns:\n        Array of distinct items\n    '
    if not list_of_dict or not key:
        return list_of_dict
    values_viewed = set()
    result = []
    for item in list_of_dict:
        value = get_value(item, key, None)
        if value is None:
            result.append(item)
            continue
        if value not in values_viewed:
            result.append(item)
        values_viewed.add(value)
    return result

def format_version(version: Optional[str]) -> Optional[str]:
    if False:
        i = 10
        return i + 15
    if not version:
        return version
    if 'GIT' in version:
        return 'dev'
    parts = version.split('-', maxsplit=2)
    if len(parts) < 2:
        return version
    if parts[1].isdigit():
        return parts[0]
    return f'{parts[0]}-{parts[1]}'

def obfuscate_string(s: str, part_of_speech: str='noun') -> str:
    if False:
        print('Hello World!')
    'Obfuscate string by replacing it with random word.\n\n    The same random words will be generated for the same given strings.\n    '
    if not s:
        return s
    faker = Faker(locale='en_US')
    faker.seed_instance(s)
    return faker.word(part_of_speech=part_of_speech)