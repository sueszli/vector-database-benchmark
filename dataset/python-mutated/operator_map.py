import operator
from collections.abc import Callable
import frappe
from frappe.database.utils import NestedSetHierarchy
from frappe.model.db_query import get_timespan_date_range
from frappe.query_builder import Field

def like(key: Field, value: str) -> frappe.qb:
    if False:
        i = 10
        return i + 15
    'Wrapper method for `LIKE`\n\n\tArgs:\n\t        key (str): field\n\t        value (str): criterion\n\n\tReturns:\n\t        frappe.qb: `frappe.qb object with `LIKE`\n\t'
    return key.like(value)

def func_in(key: Field, value: list | tuple) -> frappe.qb:
    if False:
        while True:
            i = 10
    'Wrapper method for `IN`\n\n\tArgs:\n\t        key (str): field\n\t        value (Union[int, str]): criterion\n\n\tReturns:\n\t        frappe.qb: `frappe.qb object with `IN`\n\t'
    if isinstance(value, str):
        value = value.split(',')
    return key.isin(value)

def not_like(key: Field, value: str) -> frappe.qb:
    if False:
        return 10
    'Wrapper method for `NOT LIKE`\n\n\tArgs:\n\t        key (str): field\n\t        value (str): criterion\n\n\tReturns:\n\t        frappe.qb: `frappe.qb object with `NOT LIKE`\n\t'
    return key.not_like(value)

def func_not_in(key: Field, value: list | tuple | str):
    if False:
        i = 10
        return i + 15
    'Wrapper method for `NOT IN`\n\n\tArgs:\n\t        key (str): field\n\t        value (Union[int, str]): criterion\n\n\tReturns:\n\t        frappe.qb: `frappe.qb object with `NOT IN`\n\t'
    if isinstance(value, str):
        value = value.split(',')
    return key.notin(value)

def func_regex(key: Field, value: str) -> frappe.qb:
    if False:
        i = 10
        return i + 15
    'Wrapper method for `REGEX`\n\n\tArgs:\n\t        key (str): field\n\t        value (str): criterion\n\n\tReturns:\n\t        frappe.qb: `frappe.qb object with `REGEX`\n\t'
    return key.regex(value)

def func_between(key: Field, value: list | tuple) -> frappe.qb:
    if False:
        while True:
            i = 10
    'Wrapper method for `BETWEEN`\n\n\tArgs:\n\t        key (str): field\n\t        value (Union[int, str]): criterion\n\n\tReturns:\n\t        frappe.qb: `frappe.qb object with `BETWEEN`\n\t'
    return key[slice(*value)]

def func_is(key, value):
    if False:
        return 10
    'Wrapper for IS'
    return key.isnotnull() if value.lower() == 'set' else key.isnull()

def func_timespan(key: Field, value: str) -> frappe.qb:
    if False:
        i = 10
        return i + 15
    'Wrapper method for `TIMESPAN`\n\n\tArgs:\n\t        key (str): field\n\t        value (str): criterion\n\n\tReturns:\n\t        frappe.qb: `frappe.qb object with `TIMESPAN`\n\t'
    return func_between(key, get_timespan_date_range(value))
OPERATOR_MAP: dict[str, Callable] = {'+': operator.add, '=': operator.eq, '-': operator.sub, '!=': operator.ne, '<': operator.lt, '>': operator.gt, '<=': operator.le, '=<': operator.le, '>=': operator.ge, '=>': operator.ge, '/': operator.truediv, '*': operator.mul, 'in': func_in, 'not in': func_not_in, 'like': like, 'not like': not_like, 'regex': func_regex, 'between': func_between, 'is': func_is, 'timespan': func_timespan, 'nested_set': NestedSetHierarchy}