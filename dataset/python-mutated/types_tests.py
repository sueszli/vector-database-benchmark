"""Unit tests for Superset"""
import sqlalchemy
from sqlalchemy import Column, Integer
from superset.advanced_data_type.types import AdvancedDataTypeRequest, AdvancedDataTypeResponse
from superset.utils.core import FilterOperator, FilterStringOperators
from superset.advanced_data_type.plugins.internet_address import internet_address
from superset.advanced_data_type.plugins.internet_port import internet_port as port

def test_ip_func_valid_ip():
    if False:
        return 10
    'Test to see if the cidr_func behaves as expected when a valid IP is passed in'
    cidr_request: AdvancedDataTypeRequest = {'advanced_data_type': 'cidr', 'values': ['1.1.1.1']}
    cidr_response: AdvancedDataTypeResponse = {'values': [16843009], 'error_message': '', 'display_value': '16843009', 'valid_filter_operators': [FilterStringOperators.EQUALS, FilterStringOperators.GREATER_THAN_OR_EQUAL, FilterStringOperators.GREATER_THAN, FilterStringOperators.IN, FilterStringOperators.LESS_THAN, FilterStringOperators.LESS_THAN_OR_EQUAL]}
    assert internet_address.translate_type(cidr_request) == cidr_response

def test_cidr_func_invalid_ip():
    if False:
        while True:
            i = 10
    'Test to see if the cidr_func behaves as expected when an invalid IP is passed in'
    cidr_request: AdvancedDataTypeRequest = {'advanced_data_type': 'cidr', 'values': ['abc']}
    cidr_response: AdvancedDataTypeResponse = {'values': [], 'error_message': "'abc' does not appear to be an IPv4 or IPv6 network", 'display_value': '', 'valid_filter_operators': [FilterStringOperators.EQUALS, FilterStringOperators.GREATER_THAN_OR_EQUAL, FilterStringOperators.GREATER_THAN, FilterStringOperators.IN, FilterStringOperators.LESS_THAN, FilterStringOperators.LESS_THAN_OR_EQUAL]}
    assert internet_address.translate_type(cidr_request) == cidr_response

def test_cidr_func_empty_ip():
    if False:
        return 10
    'Test to see if the cidr_func behaves as expected when no IP is passed in'
    cidr_request: AdvancedDataTypeRequest = {'advanced_data_type': 'cidr', 'values': ['']}
    cidr_response: AdvancedDataTypeResponse = {'values': [''], 'error_message': '', 'display_value': '', 'valid_filter_operators': [FilterStringOperators.EQUALS, FilterStringOperators.GREATER_THAN_OR_EQUAL, FilterStringOperators.GREATER_THAN, FilterStringOperators.IN, FilterStringOperators.LESS_THAN, FilterStringOperators.LESS_THAN_OR_EQUAL]}
    assert internet_address.translate_type(cidr_request) == cidr_response

def test_port_translation_func_valid_port_number():
    if False:
        return 10
    'Test to see if the port_translation_func behaves as expected when a valid port number\n    is passed in'
    port_request: AdvancedDataTypeRequest = {'advanced_data_type': 'port', 'values': ['80']}
    port_response: AdvancedDataTypeResponse = {'values': [[80]], 'error_message': '', 'display_value': '[80]', 'valid_filter_operators': [FilterStringOperators.EQUALS, FilterStringOperators.GREATER_THAN_OR_EQUAL, FilterStringOperators.GREATER_THAN, FilterStringOperators.IN, FilterStringOperators.LESS_THAN, FilterStringOperators.LESS_THAN_OR_EQUAL]}
    assert port.translate_type(port_request) == port_response

def test_port_translation_func_valid_port_name():
    if False:
        i = 10
        return i + 15
    'Test to see if the port_translation_func behaves as expected when a valid port name\n    is passed in'
    port_request: AdvancedDataTypeRequest = {'advanced_data_type': 'port', 'values': ['https']}
    port_response: AdvancedDataTypeResponse = {'values': [[443]], 'error_message': '', 'display_value': '[443]', 'valid_filter_operators': [FilterStringOperators.EQUALS, FilterStringOperators.GREATER_THAN_OR_EQUAL, FilterStringOperators.GREATER_THAN, FilterStringOperators.IN, FilterStringOperators.LESS_THAN, FilterStringOperators.LESS_THAN_OR_EQUAL]}
    assert port.translate_type(port_request) == port_response

def test_port_translation_func_invalid_port_name():
    if False:
        return 10
    'Test to see if the port_translation_func behaves as expected when an invalid port name\n    is passed in'
    port_request: AdvancedDataTypeRequest = {'advanced_data_type': 'port', 'values': ['abc']}
    port_response: AdvancedDataTypeResponse = {'values': [], 'error_message': "'abc' does not appear to be a port name or number", 'display_value': '', 'valid_filter_operators': [FilterStringOperators.EQUALS, FilterStringOperators.GREATER_THAN_OR_EQUAL, FilterStringOperators.GREATER_THAN, FilterStringOperators.IN, FilterStringOperators.LESS_THAN, FilterStringOperators.LESS_THAN_OR_EQUAL]}
    assert port.translate_type(port_request) == port_response

def test_port_translation_func_invalid_port_number():
    if False:
        return 10
    'Test to see if the port_translation_func behaves as expected when an invalid port\n    number is passed in'
    port_request: AdvancedDataTypeRequest = {'advanced_data_type': 'port', 'values': ['123456789']}
    port_response: AdvancedDataTypeResponse = {'values': [], 'error_message': "'123456789' does not appear to be a port name or number", 'display_value': '', 'valid_filter_operators': [FilterStringOperators.EQUALS, FilterStringOperators.GREATER_THAN_OR_EQUAL, FilterStringOperators.GREATER_THAN, FilterStringOperators.IN, FilterStringOperators.LESS_THAN, FilterStringOperators.LESS_THAN_OR_EQUAL]}
    assert port.translate_type(port_request) == port_response

def test_port_translation_func_empty_port_number():
    if False:
        while True:
            i = 10
    'Test to see if the port_translation_func behaves as expected when no port\n    number is passed in'
    port_request: AdvancedDataTypeRequest = {'advanced_data_type': 'port', 'values': ['']}
    port_response: AdvancedDataTypeResponse = {'values': [['']], 'error_message': '', 'display_value': '', 'valid_filter_operators': [FilterStringOperators.EQUALS, FilterStringOperators.GREATER_THAN_OR_EQUAL, FilterStringOperators.GREATER_THAN, FilterStringOperators.IN, FilterStringOperators.LESS_THAN, FilterStringOperators.LESS_THAN_OR_EQUAL]}
    assert port.translate_type(port_request) == port_response

def test_cidr_translate_filter_func_equals():
    if False:
        return 10
    'Test to see if the cidr_translate_filter_func behaves as expected when the EQUALS\n    operator is used'
    input_column = Column('user_ip', Integer)
    input_operation = FilterOperator.EQUALS
    input_values = [16843009]
    cidr_translate_filter_response = input_column == input_values[0]
    assert internet_address.translate_filter(input_column, input_operation, input_values).compare(cidr_translate_filter_response)

def test_cidr_translate_filter_func_not_equals():
    if False:
        for i in range(10):
            print('nop')
    'Test to see if the cidr_translate_filter_func behaves as expected when the NOT_EQUALS\n    operator is used'
    input_column = Column('user_ip', Integer)
    input_operation = FilterOperator.NOT_EQUALS
    input_values = [16843009]
    cidr_translate_filter_response = input_column != input_values[0]
    assert internet_address.translate_filter(input_column, input_operation, input_values).compare(cidr_translate_filter_response)

def test_cidr_translate_filter_func_greater_than_or_equals():
    if False:
        i = 10
        return i + 15
    'Test to see if the cidr_translate_filter_func behaves as expected when the\n    GREATER_THAN_OR_EQUALS operator is used'
    input_column = Column('user_ip', Integer)
    input_operation = FilterOperator.GREATER_THAN_OR_EQUALS
    input_values = [16843009]
    cidr_translate_filter_response: sqlalchemy.sql.expression.BinaryExpression = input_column >= input_values[0]
    assert internet_address.translate_filter(input_column, input_operation, input_values).compare(cidr_translate_filter_response)

def test_cidr_translate_filter_func_greater_than():
    if False:
        i = 10
        return i + 15
    'Test to see if the cidr_translate_filter_func behaves as expected when the\n    GREATER_THAN operator is used'
    input_column = Column('user_ip', Integer)
    input_operation = FilterOperator.GREATER_THAN
    input_values = [16843009]
    cidr_translate_filter_response: sqlalchemy.sql.expression.BinaryExpression = input_column > input_values[0]
    assert internet_address.translate_filter(input_column, input_operation, input_values).compare(cidr_translate_filter_response)

def test_cidr_translate_filter_func_less_than():
    if False:
        for i in range(10):
            print('nop')
    'Test to see if the cidr_translate_filter_func behaves as expected when the LESS_THAN\n    operator is used'
    input_column = Column('user_ip', Integer)
    input_operation = FilterOperator.LESS_THAN
    input_values = [16843009]
    cidr_translate_filter_response: sqlalchemy.sql.expression.BinaryExpression = input_column < input_values[0]
    assert internet_address.translate_filter(input_column, input_operation, input_values).compare(cidr_translate_filter_response)

def test_cidr_translate_filter_func_less_than_or_equals():
    if False:
        i = 10
        return i + 15
    'Test to see if the cidr_translate_filter_func behaves as expected when the\n    LESS_THAN_OR_EQUALS operator is used'
    input_column = Column('user_ip', Integer)
    input_operation = FilterOperator.LESS_THAN_OR_EQUALS
    input_values = [16843009]
    cidr_translate_filter_response: sqlalchemy.sql.expression.BinaryExpression = input_column <= input_values[0]
    assert internet_address.translate_filter(input_column, input_operation, input_values).compare(cidr_translate_filter_response)

def test_cidr_translate_filter_func_in_single():
    if False:
        return 10
    'Test to see if the cidr_translate_filter_func behaves as expected when the IN operator\n    is used with a single IP'
    input_column = Column('user_ip', Integer)
    input_operation = FilterOperator.IN
    input_values = [16843009]
    cidr_translate_filter_response: sqlalchemy.sql.expression.BinaryExpression = input_column.in_(input_values)
    assert internet_address.translate_filter(input_column, input_operation, input_values).compare(cidr_translate_filter_response)

def test_cidr_translate_filter_func_in_double():
    if False:
        for i in range(10):
            print('nop')
    "Test to see if the cidr_translate_filter_func behaves as expected when the IN operator\n    is used with two IP's"
    input_column = Column('user_ip', Integer)
    input_operation = FilterOperator.IN
    input_values = [{'start': 16843009, 'end': 33686018}]
    input_condition = input_column.in_([])
    cidr_translate_filter_response: sqlalchemy.sql.expression.BinaryExpression = input_condition | (input_column <= 33686018) & (input_column >= 16843009)
    assert internet_address.translate_filter(input_column, input_operation, input_values).compare(cidr_translate_filter_response)

def test_cidr_translate_filter_func_not_in_single():
    if False:
        print('Hello World!')
    'Test to see if the cidr_translate_filter_func behaves as expected when the NOT_IN\n    operator is used with a single IP'
    input_column = Column('user_ip', Integer)
    input_operation = FilterOperator.NOT_IN
    input_values = [16843009]
    cidr_translate_filter_response: sqlalchemy.sql.expression.BinaryExpression = ~input_column.in_(input_values)
    assert internet_address.translate_filter(input_column, input_operation, input_values).compare(cidr_translate_filter_response)

def test_cidr_translate_filter_func_not_in_double():
    if False:
        for i in range(10):
            print('nop')
    "Test to see if the cidr_translate_filter_func behaves as expected when the NOT_IN\n    operator is used with two IP's"
    input_column = Column('user_ip', Integer)
    input_operation = FilterOperator.NOT_IN
    input_values = [{'start': 16843009, 'end': 33686018}]
    input_condition = ~input_column.in_([])
    cidr_translate_filter_response: sqlalchemy.sql.expression.BinaryExpression = input_condition & (input_column > 33686018) & (input_column < 16843009)
    assert internet_address.translate_filter(input_column, input_operation, input_values).compare(cidr_translate_filter_response)

def test_port_translate_filter_func_equals():
    if False:
        i = 10
        return i + 15
    'Test to see if the port_translate_filter_func behaves as expected when the EQUALS\n    operator is used'
    input_column = Column('user_ip', Integer)
    input_operation = FilterOperator.EQUALS
    input_values = [[443]]
    port_translate_filter_response: sqlalchemy.sql.expression.BinaryExpression = input_column.in_(input_values[0])
    assert port.translate_filter(input_column, input_operation, input_values).compare(port_translate_filter_response)

def test_port_translate_filter_func_not_equals():
    if False:
        return 10
    'Test to see if the port_translate_filter_func behaves as expected when the NOT_EQUALS\n    operator is used'
    input_column = Column('user_ip', Integer)
    input_operation = FilterOperator.NOT_EQUALS
    input_values = [[443]]
    port_translate_filter_response: sqlalchemy.sql.expression.BinaryExpression = ~input_column.in_(input_values[0])
    assert port.translate_filter(input_column, input_operation, input_values).compare(port_translate_filter_response)

def test_port_translate_filter_func_greater_than_or_equals():
    if False:
        while True:
            i = 10
    'Test to see if the port_translate_filter_func behaves as expected when the\n    GREATER_THAN_OR_EQUALS operator is used'
    input_column = Column('user_ip', Integer)
    input_operation = FilterOperator.GREATER_THAN_OR_EQUALS
    input_values = [[443]]
    port_translate_filter_response: sqlalchemy.sql.expression.BinaryExpression = input_column >= input_values[0][0]
    assert port.translate_filter(input_column, input_operation, input_values).compare(port_translate_filter_response)

def test_port_translate_filter_func_greater_than():
    if False:
        return 10
    'Test to see if the port_translate_filter_func behaves as expected when the\n    GREATER_THAN operator is used'
    input_column = Column('user_ip', Integer)
    input_operation = FilterOperator.GREATER_THAN
    input_values = [[443]]
    port_translate_filter_response: sqlalchemy.sql.expression.BinaryExpression = input_column > input_values[0][0]
    assert port.translate_filter(input_column, input_operation, input_values).compare(port_translate_filter_response)

def test_port_translate_filter_func_less_than_or_equals():
    if False:
        print('Hello World!')
    'Test to see if the port_translate_filter_func behaves as expected when the\n    LESS_THAN_OR_EQUALS operator is used'
    input_column = Column('user_ip', Integer)
    input_operation = FilterOperator.LESS_THAN_OR_EQUALS
    input_values = [[443]]
    port_translate_filter_response: sqlalchemy.sql.expression.BinaryExpression = input_column <= input_values[0][0]
    assert port.translate_filter(input_column, input_operation, input_values).compare(port_translate_filter_response)

def test_port_translate_filter_func_less_than():
    if False:
        return 10
    'Test to see if the port_translate_filter_func behaves as expected when the LESS_THAN\n    operator is used'
    input_column = Column('user_ip', Integer)
    input_operation = FilterOperator.LESS_THAN
    input_values = [[443]]
    port_translate_filter_response: sqlalchemy.sql.expression.BinaryExpression = input_column < input_values[0][0]
    assert port.translate_filter(input_column, input_operation, input_values).compare(port_translate_filter_response)

def test_port_translate_filter_func_in_single():
    if False:
        return 10
    'Test to see if the port_translate_filter_func behaves as expected when the IN operator\n    is used with a single port'
    input_column = Column('user_ip', Integer)
    input_operation = FilterOperator.IN
    input_values = [[443]]
    port_translate_filter_response: sqlalchemy.sql.expression.BinaryExpression = input_column.in_(input_values[0])
    assert port.translate_filter(input_column, input_operation, input_values).compare(port_translate_filter_response)

def test_port_translate_filter_func_in_double():
    if False:
        for i in range(10):
            print('nop')
    'Test to see if the port_translate_filter_func behaves as expected when the IN operator\n    is used with two ports'
    input_column = Column('user_ip', Integer)
    input_operation = FilterOperator.IN
    input_values = [[443, 80]]
    port_translate_filter_response: sqlalchemy.sql.expression.BinaryExpression = input_column.in_(input_values[0])
    assert port.translate_filter(input_column, input_operation, input_values).compare(port_translate_filter_response)

def test_port_translate_filter_func_not_in_single():
    if False:
        return 10
    'Test to see if the port_translate_filter_func behaves as expected when the NOT_IN\n    operator is used with a single port'
    input_column = Column('user_ip', Integer)
    input_operation = FilterOperator.NOT_IN
    input_values = [[443]]
    port_translate_filter_response: sqlalchemy.sql.expression.BinaryExpression = ~input_column.in_(input_values[0])
    assert port.translate_filter(input_column, input_operation, input_values).compare(port_translate_filter_response)

def test_port_translate_filter_func_not_in_double():
    if False:
        for i in range(10):
            print('nop')
    'Test to see if the port_translate_filter_func behaves as expected when the NOT_IN\n    operator is used with two ports'
    input_column = Column('user_ip', Integer)
    input_operation = FilterOperator.NOT_IN
    input_values = [[443, 80]]
    port_translate_filter_response: sqlalchemy.sql.expression.BinaryExpression = ~input_column.in_(input_values[0])
    assert port.translate_filter(input_column, input_operation, input_values).compare(port_translate_filter_response)