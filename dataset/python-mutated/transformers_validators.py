from .overrides import Enum
from typing import Union, Any, Type
NAME_PATTERN = '^[a-z,A-Z,0-9,\\-,é,è,à,ç, ,|,&,\\/,\\\\,_,.,#]*$'

def transform_email(email: str) -> str:
    if False:
        while True:
            i = 10
    return email.lower().strip() if isinstance(email, str) else email

def remove_whitespace(value: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    return ' '.join(value.split()) if isinstance(value, str) else value

def remove_duplicate_values(value: list) -> list:
    if False:
        return 10
    if value is not None and isinstance(value, list):
        if len(value) > 0 and (isinstance(value[0], int) or isinstance(value[0], dict)):
            return value
        value = list(set(value))
    return value

def single_to_list(value: Union[list, Any]) -> list:
    if False:
        while True:
            i = 10
    if value is not None and (not isinstance(value, list)):
        value = [value]
    return value

def force_is_event(events_enum: list[Type[Enum]]):
    if False:
        while True:
            i = 10

    def fn(value: list):
        if False:
            i = 10
            return i + 15
        if value is not None and isinstance(value, list):
            for v in value:
                r = False
                for en in events_enum:
                    if en.has_value(v['type']) or en.has_value(v['type'].lower()):
                        r = True
                        break
                v['isEvent'] = r
        return value
    return fn