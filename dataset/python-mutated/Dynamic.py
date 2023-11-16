from decimal import Decimal
from robot.api.deco import keyword

class Dynamic:

    def get_keyword_names(self):
        if False:
            for i in range(10):
                print('nop')
        return [name for name in dir(self) if hasattr(getattr(self, name), 'robot_name')]

    def run_keyword(self, name, args, kwargs):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self, name)(*args, **kwargs)

    def get_keyword_arguments(self, name):
        if False:
            for i in range(10):
                print('nop')
        if name == 'default_values':
            return [('first', 1), ('first_expected', 1), ('middle', None), ('middle_expected', None), ('last', True), ('last_expected', True)]
        if name == 'kwonly_defaults':
            return [('*',), ('first', 1), ('first_expected', 1), ('last', True), ('last_expected', True)]
        if name == 'default_values_when_types_are_none':
            return [('value', True), ('expected', None)]
        return ['value', 'expected=None']

    def get_keyword_types(self, name):
        if False:
            return 10
        return getattr(self, name).robot_types

    @keyword(types=[int])
    def list_of_types(self, value, expected=None):
        if False:
            for i in range(10):
                print('nop')
        self._validate_type(value, expected)

    @keyword(types={'value': Decimal, 'return': None})
    def dict_of_types(self, value, expected=None):
        if False:
            for i in range(10):
                print('nop')
        self._validate_type(value, expected)

    @keyword(types=['bytes'])
    def list_of_aliases(self, value, expected=None):
        if False:
            i = 10
            return i + 15
        self._validate_type(value, expected)

    @keyword(types={'value': 'Dictionary'})
    def dict_of_aliases(self, value, expected=None):
        if False:
            print('Hello World!')
        self._validate_type(value, expected)

    @keyword
    def default_values(self, first=1, first_expected=1, middle=None, middle_expected=None, last=True, last_expected=True):
        if False:
            while True:
                i = 10
        self._validate_type(first, first_expected)
        self._validate_type(middle, middle_expected)
        self._validate_type(last, last_expected)

    @keyword
    def kwonly_defaults(self, first=1, first_expected=1, last=True, last_expected=True):
        if False:
            i = 10
            return i + 15
        self._validate_type(first, first_expected)
        self._validate_type(last, last_expected)

    @keyword(types=None)
    def default_values_when_types_are_none(self, value=True, expected=None):
        if False:
            return 10
        self._validate_type(value, expected)

    def _validate_type(self, argument, expected):
        if False:
            while True:
                i = 10
        if isinstance(expected, str):
            expected = eval(expected)
        if argument != expected or type(argument) != type(expected):
            raise AssertionError('%r (%s) != %r (%s)' % (argument, type(argument).__name__, expected, type(expected).__name__))