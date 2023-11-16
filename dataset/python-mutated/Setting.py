import os
from collections import Iterable, OrderedDict
from coala_utils.decorators import enforce_signature, generate_repr
from coala_utils.string_processing.StringConverter import StringConverter
from coalib.bearlib.languages.Language import Language, UnknownLanguageError
from coalib.parsing.Globbing import glob_escape
from coalib.results.SourcePosition import SourcePosition

def path(obj, *args, **kwargs):
    if False:
        while True:
            i = 10
    return obj.__path__(*args, **kwargs)

def path_list(obj, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return obj.__path_list__(*args, **kwargs)

def url(obj, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    return obj.__url__(*args, **kwargs)

def glob(obj, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Creates a path in which all special glob characters in all the\n    parent directories in the given setting are properly escaped.\n\n    :param obj: The ``Setting`` object from which the key is obtained.\n    :return:    Returns a path in which special glob characters are escaped.\n    '
    return obj.__glob__(*args, **kwargs)

def glob_list(obj, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Creates a list of paths in which all special glob characters in all the\n    parent directories of all paths in the given setting are properly escaped.\n\n    :param obj: The ``Setting`` object from which the key is obtained.\n    :return:    Returns a list of paths in which special glob characters are\n                escaped.\n    '
    return obj.__glob_list__(*args, **kwargs)

def language(name):
    if False:
        i = 10
        return i + 15
    '\n    Convert a string into ``Language`` object.\n\n    :param name:        String containing language name.\n    :return:            ``Language`` object.\n    :raises ValueError: If the ``name`` contain invalid language name.\n    '
    try:
        return Language[name]
    except UnknownLanguageError as e:
        raise ValueError(e)

def typed_list(conversion_func):
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates a class that converts a setting into a list of elements each\n    converted with the given conversion function.\n\n    :param conversion_func: The conversion function that converts a string into\n                            your desired list item object.\n    :return:                An instance of the created conversion class.\n    '

    class Converter:

        def __call__(self, setting):
            if False:
                print('Hello World!')
            return [conversion_func(StringConverter(elem)) for elem in setting]

        def __repr__(self):
            if False:
                return 10
            return f'typed_list({conversion_func.__name__}) at ({hex(id(self))})'
    return Converter()
str_list = typed_list(str)
int_list = typed_list(int)
float_list = typed_list(float)
bool_list = typed_list(bool)

def typed_dict(key_type, value_type, default):
    if False:
        return 10
    '\n    Creates a class that converts a setting into a dict with the given types.\n\n    :param key_type:   The type conversion function for the keys.\n    :param value_type: The type conversion function for the values.\n    :param default:    The default value to use if no one is given by the user.\n    :return:           An instance of the created conversion class.\n    '

    class Converter:

        def __call__(self, setting):
            if False:
                print('Hello World!')
            return {key_type(StringConverter(key)): value_type(StringConverter(value)) if value != '' else default for (key, value) in dict(setting).items()}

        def __repr__(self):
            if False:
                i = 10
                return i + 15
            return f'typed_dict({key_type.__name__}, {value_type.__name__}, ' + f'default={default}) at ({hex(id(self))})'
    return Converter()

def typed_ordered_dict(key_type, value_type, default):
    if False:
        return 10
    '\n    Creates a class that converts a setting into an ordered dict with the\n    given types.\n\n    :param key_type:   The type conversion function for the keys.\n    :param value_type: The type conversion function for the values.\n    :param default:    The default value to use if no one is given by the user.\n    :return:           An instance of the created conversion class.\n    '

    class Converter:

        def __call__(self, setting):
            if False:
                print('Hello World!')
            return OrderedDict(((key_type(StringConverter(key)), value_type(StringConverter(value)) if value != '' else default) for (key, value) in OrderedDict(setting).items()))

        def __repr__(self):
            if False:
                for i in range(10):
                    print('nop')
            return f'typed_ordered_dict({key_type.__name__}, ' + f'{value_type.__name__}, default={default}) ' + f'at ({hex(id(self))})'
    return Converter()

@generate_repr('key', 'value', 'origin', 'from_cli', 'to_append')
class Setting(StringConverter):
    """
    A Setting consists mainly of a key and a value. It mainly offers many
    conversions into common data types.
    """

    @enforce_signature
    def __init__(self, key, value, origin: (str, SourcePosition)='', strip_whitespaces: bool=True, list_delimiters: Iterable=(',', ';'), from_cli: bool=False, remove_empty_iter_elements: bool=True, to_append: bool=False):
        if False:
            while True:
                i = 10
        '\n        Initializes a new Setting,\n\n        :param key:                        The key of the Setting.\n        :param value:                      The value, if you apply conversions\n                                           to this object these will be applied\n                                           to this value.\n        :param origin:                     The originating file. This will be\n                                           used for path conversions and the\n                                           last part will be stripped off. If\n                                           you want to specify a directory as\n                                           origin be sure to end it with a\n                                           directory separator.\n        :param strip_whitespaces:          Whether to strip whitespaces from\n                                           the value or not\n        :param list_delimiters:            Delimiters for list conversion\n        :param from_cli:                   True if this setting was read by the\n                                           CliParser.\n        :param remove_empty_iter_elements: Whether to remove empty elements in\n                                           iterable values.\n        :param to_append:                  The boolean value if setting value\n                                           needs to be appended to a setting in\n                                           the defaults of a section.\n        '
        self.to_append = to_append
        StringConverter.__init__(self, value, strip_whitespaces=strip_whitespaces, list_delimiters=list_delimiters, remove_empty_iter_elements=remove_empty_iter_elements)
        self.from_cli = from_cli
        self.key = key
        self._origin = origin
        self.length = 1

    def __path__(self, origin=None, glob_escape_origin=False):
        if False:
            return 10
        '\n        Determines the path of this setting.\n\n        Note: You can also use this function on strings, in that case the\n        origin argument will be taken in every case.\n\n        :param origin:             The origin file to take if no origin is\n                                   specified for the given setting. If you\n                                   want to provide a directory, make sure it\n                                   ends with a directory separator.\n        :param glob_escape_origin: When this is set to true, the origin of\n                                   this setting will be escaped with\n                                   ``glob_escape``.\n        :return:                   An absolute path.\n        :raises ValueError:        If no origin is specified in the setting\n                                   nor the given origin parameter.\n        '
        strrep = str(self).strip()
        if os.path.isabs(strrep):
            return strrep
        if hasattr(self, 'origin') and self.origin != '':
            origin = self.origin
        if origin is None:
            raise ValueError('Cannot determine path without origin.')
        origin = os.path.abspath(os.path.dirname(origin))
        if glob_escape_origin:
            origin = glob_escape(origin)
        return os.path.normpath(os.path.join(origin, strrep))

    def __glob__(self, origin=None):
        if False:
            while True:
                i = 10
        '\n        Determines the path of this setting with proper escaping of its\n        parent directories.\n\n        :param origin:      The origin file to take if no origin is specified\n                            for the given setting. If you want to provide a\n                            directory, make sure it ends with a directory\n                            separator.\n        :return:            An absolute path in which the parent directories\n                            are escaped.\n        :raises ValueError: If no origin is specified in the setting nor the\n                            given origin parameter.\n        '
        return Setting.__path__(self, origin, glob_escape_origin=True)

    def __path_list__(self):
        if False:
            print('Hello World!')
        '\n        Splits the value into a list and creates a path out of each item taking\n        the origin of the setting into account.\n\n        :return: A list of absolute paths.\n        '
        return [Setting.__path__(elem, self.origin) for elem in self]

    def __glob_list__(self):
        if False:
            i = 10
            return i + 15
        '\n        Splits the value into a list and creates a path out of each item in\n        which the special glob characters in origin are escaped.\n\n        :return: A list of absolute paths in which the special characters in\n                 the parent directories of the setting are escaped.\n        '
        return [Setting.__glob__(elem, self.origin) for elem in self]

    def __iter__(self, remove_backslashes=True):
        if False:
            i = 10
            return i + 15
        if self.to_append:
            raise ValueError('Iteration on this object is invalid because the value is incomplete. Please access the value of the setting in a section to iterate through it.')
        return StringConverter.__iter__(self, remove_backslashes)

    @property
    def key(self):
        if False:
            while True:
                i = 10
        return self._key

    @key.setter
    def key(self, key):
        if False:
            while True:
                i = 10
        newkey = str(key)
        if newkey == '':
            raise ValueError('An empty key is not allowed for a setting.')
        self._key = newkey

    @StringConverter.value.getter
    def value(self):
        if False:
            while True:
                i = 10
        if self.to_append:
            raise ValueError('This property is invalid because the value is incomplete. Please access the value of the setting in a section to get the complete value.')
        return self._value

    @property
    def origin(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the filename.\n        '
        if isinstance(self._origin, SourcePosition):
            return self._origin.filename
        else:
            return self._origin

    @property
    def line_number(self):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self._origin, SourcePosition):
            return self._origin.line
        else:
            raise TypeError("Instantiated with str 'origin' which does not have line numbers. Use SourcePosition for line numbers.")

    @property
    def end_line_number(self):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self._origin, SourcePosition):
            return self.length + self._origin.line - 1
        else:
            raise TypeError("Instantiated with str 'origin' which does not have line numbers. Use SourcePosition for line numbers.")