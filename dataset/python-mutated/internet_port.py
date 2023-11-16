import itertools
from typing import Any
from sqlalchemy import Column
from superset.advanced_data_type.types import AdvancedDataType, AdvancedDataTypeRequest, AdvancedDataTypeResponse
from superset.utils.core import FilterOperator, FilterStringOperators
port_conversion_dict: dict[str, list[int]] = {'http': [80], 'ssh': [22], 'https': [443], 'ftp': [20, 21], 'ftps': [989, 990], 'telnet': [23], 'telnets': [992], 'smtp': [25], 'submissions': [465], 'kerberos': [88], 'kerberos-adm': [749], 'poperator3': [110], 'poperator3s': [995], 'nntp': [119], 'nntps': [563], 'ntp': [123], 'snmp': [161], 'ldap': [389], 'ldaps': [636], 'imap2': [143], 'imaps': [993]}

def port_translation_func(req: AdvancedDataTypeRequest) -> AdvancedDataTypeResponse:
    if False:
        print('Hello World!')
    '\n    Convert a passed in AdvancedDataTypeRequest to a AdvancedDataTypeResponse\n    '
    resp: AdvancedDataTypeResponse = {'values': [], 'error_message': '', 'display_value': '', 'valid_filter_operators': [FilterStringOperators.EQUALS, FilterStringOperators.GREATER_THAN_OR_EQUAL, FilterStringOperators.GREATER_THAN, FilterStringOperators.IN, FilterStringOperators.LESS_THAN, FilterStringOperators.LESS_THAN_OR_EQUAL]}
    if req['values'] == ['']:
        resp['values'].append([''])
        return resp
    for val in req['values']:
        string_value = str(val)
        try:
            if string_value.isnumeric():
                if not 1 <= int(string_value) <= 65535:
                    raise ValueError
            resp['values'].append([int(string_value)] if string_value.isnumeric() else port_conversion_dict[string_value])
        except (KeyError, ValueError):
            resp['error_message'] = str(f"'{string_value}' does not appear to be a port name or number")
            break
        else:
            resp['display_value'] = ', '.join(map(lambda x: f"{x['start']} - {x['end']}" if isinstance(x, dict) else str(x), resp['values']))
    return resp

def port_translate_filter_func(col: Column, operator: FilterOperator, values: list[Any]) -> Any:
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert a passed in column, FilterOperator\n    and list of values into an sqlalchemy expression\n    '
    return_expression: Any
    if operator in (FilterOperator.IN, FilterOperator.NOT_IN):
        vals_list = itertools.chain.from_iterable(values)
        if operator == FilterOperator.IN.value:
            cond = col.in_(vals_list)
        elif operator == FilterOperator.NOT_IN.value:
            cond = ~col.in_(vals_list)
        return_expression = cond
    if len(values) == 1:
        value = values[0]
        value.sort()
        if operator == FilterOperator.EQUALS.value:
            return_expression = col.in_(value)
        if operator == FilterOperator.GREATER_THAN_OR_EQUALS.value:
            return_expression = col >= value[0]
        if operator == FilterOperator.GREATER_THAN.value:
            return_expression = col > value[0]
        if operator == FilterOperator.LESS_THAN.value:
            return_expression = col < value[-1]
        if operator == FilterOperator.LESS_THAN_OR_EQUALS.value:
            return_expression = col <= value[-1]
        if operator == FilterOperator.NOT_EQUALS.value:
            return_expression = ~col.in_(value)
    return return_expression
internet_port: AdvancedDataType = AdvancedDataType(verbose_name='port', description='represents of a port', valid_data_types=['int'], translate_filter=port_translate_filter_func, translate_type=port_translation_func)