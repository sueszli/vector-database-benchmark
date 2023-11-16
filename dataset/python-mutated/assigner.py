import re
from collections.abc import MutableSequence
from robot.errors import DataError, ExecutionStatus, HandlerExecutionFailed, VariableError
from robot.utils import DotDict, ErrorDetails, format_assign_message, get_error_message, is_dict_like, is_list_like, is_number, is_string, prepr, type_name
from .search import search_variable, VariableMatch

class VariableAssignment:

    def __init__(self, assignment):
        if False:
            while True:
                i = 10
        validator = AssignmentValidator()
        try:
            self.assignment = [validator.validate(var) for var in assignment]
            self.error = None
        except DataError as err:
            self.assignment = assignment
            self.error = err

    def __iter__(self):
        if False:
            return 10
        return iter(self.assignment)

    def __len__(self):
        if False:
            return 10
        return len(self.assignment)

    def validate_assignment(self):
        if False:
            return 10
        if self.error:
            raise self.error

    def assigner(self, context):
        if False:
            i = 10
            return i + 15
        self.validate_assignment()
        return VariableAssigner(self.assignment, context)

class AssignmentValidator:

    def __init__(self):
        if False:
            return 10
        self._seen_list = False
        self._seen_dict = False
        self._seen_any_var = False
        self._seen_assign_mark = False

    def validate(self, variable):
        if False:
            print('Hello World!')
        variable = self._validate_assign_mark(variable)
        self._validate_state(is_list=variable[0] == '@', is_dict=variable[0] == '&')
        return variable

    def _validate_assign_mark(self, variable):
        if False:
            for i in range(10):
                print('nop')
        if self._seen_assign_mark:
            raise DataError("Assign mark '=' can be used only with the last variable.", syntax=True)
        if variable.endswith('='):
            self._seen_assign_mark = True
            return variable[:-1].rstrip()
        return variable

    def _validate_state(self, is_list, is_dict):
        if False:
            for i in range(10):
                print('nop')
        if is_list and self._seen_list:
            raise DataError('Assignment can contain only one list variable.', syntax=True)
        if self._seen_dict or (is_dict and self._seen_any_var):
            raise DataError('Dictionary variable cannot be assigned with other variables.', syntax=True)
        self._seen_list += is_list
        self._seen_dict += is_dict
        self._seen_any_var = True

class VariableAssigner:
    _valid_extended_attr = re.compile('^[_a-zA-Z]\\w*$')

    def __init__(self, assignment, context):
        if False:
            i = 10
            return i + 15
        self._assignment = assignment
        self._context = context

    def __enter__(self):
        if False:
            print('Hello World!')
        return self

    def __exit__(self, etype, error, tb):
        if False:
            while True:
                i = 10
        if error is None:
            return
        if not isinstance(error, ExecutionStatus):
            error = HandlerExecutionFailed(ErrorDetails(error))
        if error.can_continue(self._context):
            self.assign(error.return_value)

    def assign(self, return_value):
        if False:
            while True:
                i = 10
        context = self._context
        context.output.trace(lambda : f'Return: {prepr(return_value)}', write_if_flat=False)
        resolver = ReturnValueResolver(self._assignment)
        for (name, items, value) in resolver.resolve(return_value):
            if items:
                value = self._item_assign(name, items, value, context.variables)
            elif not self._extended_assign(name, value, context.variables):
                value = self._normal_assign(name, value, context.variables)
            context.info(format_assign_message(name, value, items))

    def _extended_assign(self, name, value, variables):
        if False:
            while True:
                i = 10
        if name[0] != '$' or '.' not in name or name in variables:
            return False
        (base, attr) = [token.strip() for token in name[2:-1].rsplit('.', 1)]
        try:
            var = variables.replace_scalar(f'${{{base}}}')
        except VariableError:
            return False
        if not (self._variable_supports_extended_assign(var) and self._is_valid_extended_attribute(attr)):
            return False
        try:
            setattr(var, attr, value)
        except Exception:
            raise VariableError(f"Setting attribute '{attr}' to variable '${{{base}}}' failed: {get_error_message()}")
        return True

    def _variable_supports_extended_assign(self, var):
        if False:
            for i in range(10):
                print('nop')
        return not (is_string(var) or is_number(var))

    def _is_valid_extended_attribute(self, attr):
        if False:
            i = 10
            return i + 15
        return self._valid_extended_attr.match(attr) is not None

    def _parse_sequence_index(self, index):
        if False:
            while True:
                i = 10
        if isinstance(index, (int, slice)):
            return index
        if not is_string(index):
            raise ValueError
        if ':' not in index:
            return int(index)
        if index.count(':') > 2:
            raise ValueError
        return slice(*[int(i) if i else None for i in index.split(':')])

    def _variable_type_supports_item_assign(self, var):
        if False:
            while True:
                i = 10
        return hasattr(var, '__setitem__') and callable(var.__setitem__)

    def _raise_cannot_set_type(self, value, expected):
        if False:
            for i in range(10):
                print('nop')
        value_type = type_name(value)
        raise VariableError(f'Expected {expected}-like value, got {value_type}.')

    def _validate_item_assign(self, name, value):
        if False:
            print('Hello World!')
        if name[0] == '@':
            if not is_list_like(value):
                self._raise_cannot_set_type(value, 'list')
            value = list(value)
        if name[0] == '&':
            if not is_dict_like(value):
                self._raise_cannot_set_type(value, 'dictionary')
            value = DotDict(value)
        return value

    def _item_assign(self, name, items, value, variables):
        if False:
            return 10
        (*nested, item) = items
        decorated_nested_items = ''.join((f'[{item}]' for item in nested))
        var = variables.replace_scalar(f'${name[1:]}{decorated_nested_items}')
        if not self._variable_type_supports_item_assign(var):
            var_type = type_name(var)
            raise VariableError(f"Variable '{name}{decorated_nested_items}' is {var_type} and does not support item assignment.")
        selector = variables.replace_scalar(item)
        if isinstance(var, MutableSequence):
            try:
                selector = self._parse_sequence_index(selector)
            except ValueError:
                pass
        try:
            value = self._validate_item_assign(name, value)
            var[selector] = value
        except (IndexError, TypeError, Exception):
            var_type = type_name(var)
            raise VariableError(f"Setting value to {var_type} variable '{name}{decorated_nested_items}' at index [{item}] failed: {get_error_message()}")
        return value

    def _normal_assign(self, name, value, variables):
        if False:
            for i in range(10):
                print('nop')
        try:
            variables[name] = value
        except DataError as err:
            raise VariableError(f"Setting variable '{name}' failed: {err}")
        return value if name[0] == '$' else variables[name]

def ReturnValueResolver(assignment):
    if False:
        for i in range(10):
            print('nop')
    if not assignment:
        return NoReturnValueResolver()
    if len(assignment) == 1:
        return OneReturnValueResolver(assignment[0])
    if any((a[0] == '@' for a in assignment)):
        return ScalarsAndListReturnValueResolver(assignment)
    return ScalarsOnlyReturnValueResolver(assignment)

class NoReturnValueResolver:

    def resolve(self, return_value):
        if False:
            i = 10
            return i + 15
        return []

class OneReturnValueResolver:

    def __init__(self, assignment):
        if False:
            for i in range(10):
                print('nop')
        match: VariableMatch = search_variable(assignment)
        self._name = match.name
        self._items = match.items

    def resolve(self, return_value):
        if False:
            while True:
                i = 10
        if return_value is None:
            identifier = self._name[0]
            return_value = {'$': None, '@': [], '&': {}}[identifier]
        return [(self._name, self._items, return_value)]

class _MultiReturnValueResolver:

    def __init__(self, assignments):
        if False:
            for i in range(10):
                print('nop')
        self._names = []
        self._items = []
        for assign in assignments:
            match: VariableMatch = search_variable(assign)
            self._names.append(match.name)
            self._items.append(match.items)
        self._min_count = len(assignments)

    def resolve(self, return_value):
        if False:
            while True:
                i = 10
        return_value = self._convert_to_list(return_value)
        self._validate(len(return_value))
        return self._resolve(return_value)

    def _convert_to_list(self, return_value):
        if False:
            print('Hello World!')
        if return_value is None:
            return [None] * self._min_count
        if is_string(return_value):
            self._raise_expected_list(return_value)
        try:
            return list(return_value)
        except TypeError:
            self._raise_expected_list(return_value)

    def _raise_expected_list(self, ret):
        if False:
            return 10
        self._raise(f'Expected list-like value, got {type_name(ret)}.')

    def _raise(self, error):
        if False:
            while True:
                i = 10
        raise VariableError(f'Cannot set variables: {error}')

    def _validate(self, return_count):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def _resolve(self, return_value):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

class ScalarsOnlyReturnValueResolver(_MultiReturnValueResolver):

    def _validate(self, return_count):
        if False:
            print('Hello World!')
        if return_count != self._min_count:
            self._raise(f'Expected {self._min_count} return values, got {return_count}.')

    def _resolve(self, return_value):
        if False:
            return 10
        return list(zip(self._names, self._items, return_value))

class ScalarsAndListReturnValueResolver(_MultiReturnValueResolver):

    def __init__(self, assignments):
        if False:
            i = 10
            return i + 15
        super().__init__(assignments)
        self._min_count -= 1

    def _validate(self, return_count):
        if False:
            return 10
        if return_count < self._min_count:
            self._raise(f'Expected {self._min_count} or more return values, got {return_count}.')

    def _resolve(self, return_value):
        if False:
            return 10
        list_index = [a[0][0] for a in self._names].index('@')
        list_len = len(return_value) - len(self._names) + 1
        elements_before_list = list(zip(self._names[:list_index], self._items[:list_index], return_value[:list_index]))
        elements_after_list = list(zip(self._names[list_index + 1:], self._items[list_index + 1:], return_value[list_index + list_len:]))
        list_elements = [(self._names[list_index], self._items[list_index], return_value[list_index:list_index + list_len])]
        return elements_before_list + list_elements + elements_after_list