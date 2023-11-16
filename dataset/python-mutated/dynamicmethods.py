from robot.errors import DataError
from robot.utils import get_error_message, is_bytes, is_list_like, is_string, type_name
from .arguments import PythonArgumentParser

def no_dynamic_method(*args):
    if False:
        return 10
    return None

class _DynamicMethod:
    _underscore_name = NotImplemented

    def __init__(self, lib):
        if False:
            i = 10
            return i + 15
        self.method = self._get_method(lib)

    def _get_method(self, lib):
        if False:
            return 10
        for name in (self._underscore_name, self._camelCaseName):
            method = getattr(lib, name, None)
            if callable(method):
                return method
        return no_dynamic_method

    @property
    def _camelCaseName(self):
        if False:
            for i in range(10):
                print('nop')
        tokens = self._underscore_name.split('_')
        return ''.join([tokens[0]] + [t.capitalize() for t in tokens[1:]])

    @property
    def name(self):
        if False:
            i = 10
            return i + 15
        return self.method.__name__

    def __call__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self._handle_return_value(self.method(*args))
        except:
            raise DataError("Calling dynamic method '%s' failed: %s" % (self.name, get_error_message()))

    def _handle_return_value(self, value):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def _to_string(self, value, allow_tuple=False, allow_none=False):
        if False:
            for i in range(10):
                print('nop')
        if is_string(value):
            return value
        if is_bytes(value):
            return value.decode('UTF-8')
        if allow_tuple and is_list_like(value) and (len(value) > 0):
            return tuple(value)
        if allow_none and value is None:
            return value
        or_tuple = ' or a non-empty tuple' if allow_tuple else ''
        raise DataError('Return value must be a string%s, got %s.' % (or_tuple, type_name(value)))

    def _to_list(self, value):
        if False:
            print('Hello World!')
        if value is None:
            return ()
        if not is_list_like(value):
            raise DataError
        return value

    def _to_list_of_strings(self, value, allow_tuples=False):
        if False:
            while True:
                i = 10
        try:
            return [self._to_string(item, allow_tuples) for item in self._to_list(value)]
        except DataError:
            raise DataError('Return value must be a list of strings%s.' % (' or non-empty tuples' if allow_tuples else ''))

    def __bool__(self):
        if False:
            print('Hello World!')
        return self.method is not no_dynamic_method

class GetKeywordNames(_DynamicMethod):
    _underscore_name = 'get_keyword_names'

    def _handle_return_value(self, value):
        if False:
            return 10
        names = self._to_list_of_strings(value)
        return list(self._remove_duplicates(names))

    def _remove_duplicates(self, names):
        if False:
            print('Hello World!')
        seen = set()
        for name in names:
            if name not in seen:
                seen.add(name)
                yield name

class RunKeyword(_DynamicMethod):
    _underscore_name = 'run_keyword'

    @property
    def supports_kwargs(self):
        if False:
            return 10
        spec = PythonArgumentParser().parse(self.method)
        return len(spec.positional) == 3

class GetKeywordDocumentation(_DynamicMethod):
    _underscore_name = 'get_keyword_documentation'

    def _handle_return_value(self, value):
        if False:
            return 10
        return self._to_string(value or '')

class GetKeywordArguments(_DynamicMethod):
    _underscore_name = 'get_keyword_arguments'

    def __init__(self, lib):
        if False:
            for i in range(10):
                print('nop')
        _DynamicMethod.__init__(self, lib)
        self._supports_kwargs = RunKeyword(lib).supports_kwargs

    def _handle_return_value(self, value):
        if False:
            for i in range(10):
                print('nop')
        if value is None:
            if self._supports_kwargs:
                return ['*varargs', '**kwargs']
            return ['*varargs']
        return self._to_list_of_strings(value, allow_tuples=True)

class GetKeywordTypes(_DynamicMethod):
    _underscore_name = 'get_keyword_types'

    def _handle_return_value(self, value):
        if False:
            for i in range(10):
                print('nop')
        return value if self else {}

class GetKeywordTags(_DynamicMethod):
    _underscore_name = 'get_keyword_tags'

    def _handle_return_value(self, value):
        if False:
            return 10
        return self._to_list_of_strings(value)

class GetKeywordSource(_DynamicMethod):
    _underscore_name = 'get_keyword_source'

    def _handle_return_value(self, value):
        if False:
            return 10
        return self._to_string(value, allow_none=True)