import re
from collections import namedtuple
from boto3.exceptions import DynamoDBNeedsConditionError, DynamoDBNeedsKeyConditionError, DynamoDBOperationNotSupportedError
ATTR_NAME_REGEX = re.compile('[^.\\[\\]]+(?![^\\[]*\\])')

class ConditionBase:
    expression_format = ''
    expression_operator = ''
    has_grouped_values = False

    def __init__(self, *values):
        if False:
            i = 10
            return i + 15
        self._values = values

    def __and__(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, ConditionBase):
            raise DynamoDBOperationNotSupportedError('AND', other)
        return And(self, other)

    def __or__(self, other):
        if False:
            return 10
        if not isinstance(other, ConditionBase):
            raise DynamoDBOperationNotSupportedError('OR', other)
        return Or(self, other)

    def __invert__(self):
        if False:
            print('Hello World!')
        return Not(self)

    def get_expression(self):
        if False:
            for i in range(10):
                print('nop')
        return {'format': self.expression_format, 'operator': self.expression_operator, 'values': self._values}

    def __eq__(self, other):
        if False:
            return 10
        if isinstance(other, type(self)):
            if self._values == other._values:
                return True
        return False

    def __ne__(self, other):
        if False:
            return 10
        return not self.__eq__(other)

class AttributeBase:

    def __init__(self, name):
        if False:
            print('Hello World!')
        self.name = name

    def __and__(self, value):
        if False:
            for i in range(10):
                print('nop')
        raise DynamoDBOperationNotSupportedError('AND', self)

    def __or__(self, value):
        if False:
            for i in range(10):
                print('nop')
        raise DynamoDBOperationNotSupportedError('OR', self)

    def __invert__(self):
        if False:
            while True:
                i = 10
        raise DynamoDBOperationNotSupportedError('NOT', self)

    def eq(self, value):
        if False:
            print('Hello World!')
        'Creates a condition where the attribute is equal to the value.\n\n        :param value: The value that the attribute is equal to.\n        '
        return Equals(self, value)

    def lt(self, value):
        if False:
            return 10
        'Creates a condition where the attribute is less than the value.\n\n        :param value: The value that the attribute is less than.\n        '
        return LessThan(self, value)

    def lte(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Creates a condition where the attribute is less than or equal to the\n           value.\n\n        :param value: The value that the attribute is less than or equal to.\n        '
        return LessThanEquals(self, value)

    def gt(self, value):
        if False:
            i = 10
            return i + 15
        'Creates a condition where the attribute is greater than the value.\n\n        :param value: The value that the attribute is greater than.\n        '
        return GreaterThan(self, value)

    def gte(self, value):
        if False:
            print('Hello World!')
        'Creates a condition where the attribute is greater than or equal to\n           the value.\n\n        :param value: The value that the attribute is greater than or equal to.\n        '
        return GreaterThanEquals(self, value)

    def begins_with(self, value):
        if False:
            while True:
                i = 10
        'Creates a condition where the attribute begins with the value.\n\n        :param value: The value that the attribute begins with.\n        '
        return BeginsWith(self, value)

    def between(self, low_value, high_value):
        if False:
            return 10
        'Creates a condition where the attribute is greater than or equal\n        to the low value and less than or equal to the high value.\n\n        :param low_value: The value that the attribute is greater than or equal to.\n        :param high_value: The value that the attribute is less than or equal to.\n        '
        return Between(self, low_value, high_value)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return isinstance(other, type(self)) and self.name == other.name

    def __ne__(self, other):
        if False:
            print('Hello World!')
        return not self.__eq__(other)

class ConditionAttributeBase(ConditionBase, AttributeBase):
    """This base class is for conditions that can have attribute methods.

    One example is the Size condition. To complete a condition, you need
    to apply another AttributeBase method like eq().
    """

    def __init__(self, *values):
        if False:
            i = 10
            return i + 15
        ConditionBase.__init__(self, *values)
        AttributeBase.__init__(self, values[0].name)

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return ConditionBase.__eq__(self, other) and AttributeBase.__eq__(self, other)

    def __ne__(self, other):
        if False:
            print('Hello World!')
        return not self.__eq__(other)

class ComparisonCondition(ConditionBase):
    expression_format = '{0} {operator} {1}'

class Equals(ComparisonCondition):
    expression_operator = '='

class NotEquals(ComparisonCondition):
    expression_operator = '<>'

class LessThan(ComparisonCondition):
    expression_operator = '<'

class LessThanEquals(ComparisonCondition):
    expression_operator = '<='

class GreaterThan(ComparisonCondition):
    expression_operator = '>'

class GreaterThanEquals(ComparisonCondition):
    expression_operator = '>='

class In(ComparisonCondition):
    expression_operator = 'IN'
    has_grouped_values = True

class Between(ConditionBase):
    expression_operator = 'BETWEEN'
    expression_format = '{0} {operator} {1} AND {2}'

class BeginsWith(ConditionBase):
    expression_operator = 'begins_with'
    expression_format = '{operator}({0}, {1})'

class Contains(ConditionBase):
    expression_operator = 'contains'
    expression_format = '{operator}({0}, {1})'

class Size(ConditionAttributeBase):
    expression_operator = 'size'
    expression_format = '{operator}({0})'

class AttributeType(ConditionBase):
    expression_operator = 'attribute_type'
    expression_format = '{operator}({0}, {1})'

class AttributeExists(ConditionBase):
    expression_operator = 'attribute_exists'
    expression_format = '{operator}({0})'

class AttributeNotExists(ConditionBase):
    expression_operator = 'attribute_not_exists'
    expression_format = '{operator}({0})'

class And(ConditionBase):
    expression_operator = 'AND'
    expression_format = '({0} {operator} {1})'

class Or(ConditionBase):
    expression_operator = 'OR'
    expression_format = '({0} {operator} {1})'

class Not(ConditionBase):
    expression_operator = 'NOT'
    expression_format = '({operator} {0})'

class Key(AttributeBase):
    pass

class Attr(AttributeBase):
    """Represents an DynamoDB item's attribute."""

    def ne(self, value):
        if False:
            while True:
                i = 10
        'Creates a condition where the attribute is not equal to the value\n\n        :param value: The value that the attribute is not equal to.\n        '
        return NotEquals(self, value)

    def is_in(self, value):
        if False:
            i = 10
            return i + 15
        'Creates a condition where the attribute is in the value,\n\n        :type value: list\n        :param value: The value that the attribute is in.\n        '
        return In(self, value)

    def exists(self):
        if False:
            return 10
        'Creates a condition where the attribute exists.'
        return AttributeExists(self)

    def not_exists(self):
        if False:
            print('Hello World!')
        'Creates a condition where the attribute does not exist.'
        return AttributeNotExists(self)

    def contains(self, value):
        if False:
            i = 10
            return i + 15
        'Creates a condition where the attribute contains the value.\n\n        :param value: The value the attribute contains.\n        '
        return Contains(self, value)

    def size(self):
        if False:
            return 10
        'Creates a condition for the attribute size.\n\n        Note another AttributeBase method must be called on the returned\n        size condition to be a valid DynamoDB condition.\n        '
        return Size(self)

    def attribute_type(self, value):
        if False:
            while True:
                i = 10
        'Creates a condition for the attribute type.\n\n        :param value: The type of the attribute.\n        '
        return AttributeType(self, value)
BuiltConditionExpression = namedtuple('BuiltConditionExpression', ['condition_expression', 'attribute_name_placeholders', 'attribute_value_placeholders'])

class ConditionExpressionBuilder:
    """This class is used to build condition expressions with placeholders"""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._name_count = 0
        self._value_count = 0
        self._name_placeholder = 'n'
        self._value_placeholder = 'v'

    def _get_name_placeholder(self):
        if False:
            i = 10
            return i + 15
        return '#' + self._name_placeholder + str(self._name_count)

    def _get_value_placeholder(self):
        if False:
            for i in range(10):
                print('nop')
        return ':' + self._value_placeholder + str(self._value_count)

    def reset(self):
        if False:
            i = 10
            return i + 15
        'Resets the placeholder name and values'
        self._name_count = 0
        self._value_count = 0

    def build_expression(self, condition, is_key_condition=False):
        if False:
            i = 10
            return i + 15
        "Builds the condition expression and the dictionary of placeholders.\n\n        :type condition: ConditionBase\n        :param condition: A condition to be built into a condition expression\n            string with any necessary placeholders.\n\n        :type is_key_condition: Boolean\n        :param is_key_condition: True if the expression is for a\n            KeyConditionExpression. False otherwise.\n\n        :rtype: (string, dict, dict)\n        :returns: Will return a string representing the condition with\n            placeholders inserted where necessary, a dictionary of\n            placeholders for attribute names, and a dictionary of\n            placeholders for attribute values. Here is a sample return value:\n\n            ('#n0 = :v0', {'#n0': 'myattribute'}, {':v1': 'myvalue'})\n        "
        if not isinstance(condition, ConditionBase):
            raise DynamoDBNeedsConditionError(condition)
        attribute_name_placeholders = {}
        attribute_value_placeholders = {}
        condition_expression = self._build_expression(condition, attribute_name_placeholders, attribute_value_placeholders, is_key_condition=is_key_condition)
        return BuiltConditionExpression(condition_expression=condition_expression, attribute_name_placeholders=attribute_name_placeholders, attribute_value_placeholders=attribute_value_placeholders)

    def _build_expression(self, condition, attribute_name_placeholders, attribute_value_placeholders, is_key_condition):
        if False:
            print('Hello World!')
        expression_dict = condition.get_expression()
        replaced_values = []
        for value in expression_dict['values']:
            replaced_value = self._build_expression_component(value, attribute_name_placeholders, attribute_value_placeholders, condition.has_grouped_values, is_key_condition)
            replaced_values.append(replaced_value)
        return expression_dict['format'].format(*replaced_values, operator=expression_dict['operator'])

    def _build_expression_component(self, value, attribute_name_placeholders, attribute_value_placeholders, has_grouped_values, is_key_condition):
        if False:
            i = 10
            return i + 15
        if isinstance(value, ConditionBase):
            return self._build_expression(value, attribute_name_placeholders, attribute_value_placeholders, is_key_condition)
        elif isinstance(value, AttributeBase):
            if is_key_condition and (not isinstance(value, Key)):
                raise DynamoDBNeedsKeyConditionError(f'Attribute object {value.name} is of type {type(value)}. KeyConditionExpression only supports Attribute objects of type Key')
            return self._build_name_placeholder(value, attribute_name_placeholders)
        else:
            return self._build_value_placeholder(value, attribute_value_placeholders, has_grouped_values)

    def _build_name_placeholder(self, value, attribute_name_placeholders):
        if False:
            return 10
        attribute_name = value.name
        attribute_name_parts = ATTR_NAME_REGEX.findall(attribute_name)
        placeholder_format = ATTR_NAME_REGEX.sub('%s', attribute_name)
        str_format_args = []
        for part in attribute_name_parts:
            name_placeholder = self._get_name_placeholder()
            self._name_count += 1
            str_format_args.append(name_placeholder)
            attribute_name_placeholders[name_placeholder] = part
        return placeholder_format % tuple(str_format_args)

    def _build_value_placeholder(self, value, attribute_value_placeholders, has_grouped_values=False):
        if False:
            for i in range(10):
                print('nop')
        if has_grouped_values:
            placeholder_list = []
            for v in value:
                value_placeholder = self._get_value_placeholder()
                self._value_count += 1
                placeholder_list.append(value_placeholder)
                attribute_value_placeholders[value_placeholder] = v
            return '(' + ', '.join(placeholder_list) + ')'
        else:
            value_placeholder = self._get_value_placeholder()
            self._value_count += 1
            attribute_value_placeholders[value_placeholder] = value
            return value_placeholder