from collections import Iterable, namedtuple
from glob import iglob
import os.path
from coala_utils.decorators import enforce_signature, generate_eq, generate_repr
from coalib.parsing.ConfParser import ConfParser

@generate_repr()
@generate_eq('language', 'docstyle', 'markers')
class DocstyleDefinition:
    """
    The DocstyleDefinition class holds values that identify a certain type of
    documentation comment (for which language, documentation style/tool used
    etc.).
    """
    Metadata = namedtuple('Metadata', ('param_start', 'param_end', 'exception_start', 'exception_end', 'return_sep'))
    ClassPadding = namedtuple('ClassPadding', ('top_padding', 'bottom_padding'))
    FunctionPadding = namedtuple('FunctionPadding', ('top_padding', 'bottom_padding'))
    DocstringTypeRegex = namedtuple('DocstringTypeRegex', ('class_sign', 'func_sign'))

    @enforce_signature
    def __init__(self, language: str, docstyle: str, markers: (Iterable, str), metadata: Metadata, class_padding: ClassPadding, function_padding: FunctionPadding, docstring_type_regex: DocstringTypeRegex, docstring_position: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        Instantiates a new DocstyleDefinition.\n\n        :param language:\n            The case insensitive programming language of the\n            documentation comment, e.g. ``"CPP"`` for C++ or\n            ``"PYTHON3"``.\n        :param docstyle:\n            The case insensitive documentation style/tool used\n            to document code, e.g. ``"default"`` or ``"doxygen"``.\n        :param markers:\n            An iterable of marker/delimiter string iterables\n            or a single marker/delimiter string iterable that\n            identify a documentation comment. See ``markers``\n            property for more details on markers.\n        :param metadata:\n            A namedtuple consisting of certain attributes that\n            form the layout of the certain documentation comment\n            e.g. ``param_start`` defining the start symbol of\n            the parameter fields and ``param_end`` defining the\n            end.\n        :param class_padding:\n            A namedtuple consisting of values about\n            blank lines before and after the documentation of\n            ``docstring_type`` class.\n        :param function_padding:\n            A namedtuple consisting of values about\n            blank lines before and after the documentation of\n            ``docstring_type`` function.\n        :param docstring_type_regex:\n            A namedtuple consisting of regex\n            about ``class`` and ``function`` of a language, which\n            is used to determine ``docstring_type`` of\n            DocumentationComment.\n        :param docstring_position:\n            Defines the position where the regex of\n            docstring type is present(i.e. ``top`` or ``bottom``).\n        '
        self._language = language.lower()
        self._docstyle = docstyle.lower()
        markers = tuple(markers)
        if len(markers) == 3 and all((isinstance(x, str) for x in markers)):
            markers = (markers,)
        self._markers = tuple((tuple(marker_set) for marker_set in markers))
        for marker_set in self._markers:
            length = len(marker_set)
            if length != 3:
                raise ValueError('Length of a given marker set was not 3 (was actually {}).'.format(length))
        self._metadata = metadata
        self._class_padding = class_padding
        self._function_padding = function_padding
        self._docstring_type_regex = docstring_type_regex
        self._docstring_position = docstring_position

    @property
    def language(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The programming language.\n\n        :return: A lower-case string defining the programming language (i.e.\n                 "cpp" or "python").\n        '
        return self._language

    @property
    def docstyle(self):
        if False:
            i = 10
            return i + 15
        '\n        The documentation style/tool used to document code.\n\n        :return: A lower-case string defining the docstyle (i.e. "default" or\n                 "doxygen").\n        '
        return self._docstyle

    @property
    def markers(self):
        if False:
            i = 10
            return i + 15
        '\n        A tuple of marker sets that identify a documentation comment.\n\n        Marker sets consist of 3 entries where the first is the start-marker,\n        the second one the each-line marker and the last one the end-marker.\n        For example a marker tuple with a single marker set\n        ``(("/**", "*", "*/"),)`` would match following documentation comment:\n\n        ::\n\n            /**\n             * This is documentation.\n             */\n\n        It\'s also possible to supply an empty each-line marker\n        (``("/**", "", "*/")``):\n\n        ::\n\n            /**\n             This is more documentation.\n             */\n\n        Markers are matched "greedy", that means it will match as many\n        each-line markers as possible. I.e. for ``("///", "///", "///")``):\n\n        ::\n\n            /// Brief documentation.\n            ///\n            /// Detailed documentation.\n\n        :return: A tuple of marker/delimiter string tuples that identify a\n                 documentation comment.\n        '
        return self._markers

    @property
    def metadata(self):
        if False:
            return 10
        '\n        A namedtuple of certain attributes present in the documentation.\n\n        These attributes are used to define parts of the documentation.\n        '
        return self._metadata

    @property
    def class_padding(self):
        if False:
            while True:
                i = 10
        '\n        A namedtuple ``ClassPadding`` consisting of values about blank lines\n        before and after the documentation of ``docstring_type`` class.\n\n        These values are official standard of following blank lines before and\n        after the documentation of ``docstring_type`` class.\n        '
        return self._class_padding

    @property
    def function_padding(self):
        if False:
            i = 10
            return i + 15
        '\n        A namedtuple ``FunctionPadding`` consisting of values about blank\n        lines before and after the documentation of ``docstring_type``\n        function.\n\n        These values are official standard of following blank lines before and\n        after the documentation of ``docstring_type`` function.\n        '
        return self._function_padding

    @property
    def docstring_type_regex(self):
        if False:
            while True:
                i = 10
        '\n        A namedtuple ``DocstringTypeRegex`` consisting of regex about ``class``\n        and ``function`` of a language, which is used to determine\n        ``docstring_type`` of DocumentationComment.\n        '
        return self._docstring_type_regex

    @property
    def docstring_position(self):
        if False:
            i = 10
            return i + 15
        '\n        Defines the position, where the regex of docstring type is present.\n        Depending on different languages the docstrings are present below or\n        above the defined class or function. This expicitly defines where the\n        class regex or function regex is present(i.e. ``top`` or ``bottom``).\n        '
        return self._docstring_position

    @classmethod
    @enforce_signature
    def load(cls, language: str, docstyle: str, coalang_dir=None):
        if False:
            while True:
                i = 10
        '\n        Loads a ``DocstyleDefinition`` from the coala docstyle definition files.\n\n        This function considers all settings inside the according coalang-files\n        as markers, except ``param_start``, ``param_end`` and ``return_sep``\n        which are considered as special metadata markers.\n\n        .. note::\n\n            When placing new coala docstyle definition files, these must\n            consist of only lowercase letters and end with ``.coalang``!\n\n        :param language:           The case insensitive programming language of\n                                   the documentation comment as a string.\n        :param docstyle:           The case insensitive documentation\n                                   style/tool used to document code, e.g.\n                                   ``"default"`` or ``"doxygen"``.\n        :param coalang_dir:        Path to directory with coalang docstyle\n                                   definition files. This replaces the default\n                                   path if given.\n        :raises FileNotFoundError: Raised when the given docstyle was not\n                                   found.\n        :raises KeyError:          Raised when the given language is not\n                                   defined for given docstyle.\n        :return:                   The ``DocstyleDefinition`` for given language\n                                   and docstyle.\n        '
        docstyle = docstyle.lower()
        language_config_parser = ConfParser(remove_empty_iter_elements=False)
        coalang_file = os.path.join(coalang_dir or os.path.dirname(__file__), docstyle + '.coalang')
        try:
            docstyle_settings = language_config_parser.parse(coalang_file)
        except FileNotFoundError:
            raise FileNotFoundError('Docstyle definition ' + repr(docstyle) + ' not found.')
        language = language.lower()
        try:
            docstyle_settings = docstyle_settings[language]
        except KeyError:
            raise KeyError('Language {!r} is not defined for docstyle {!r}.'.format(language, docstyle))
        metadata_settings = ('param_start', 'param_end', 'exception_start', 'exception_end', 'return_sep')
        metadata = cls.Metadata(*(str(docstyle_settings.get(req_setting, '')) for req_setting in metadata_settings))
        try:
            class_padding = cls.ClassPadding(*(int(padding) for padding in tuple(docstyle_settings['class_padding'])))
        except IndexError:
            class_padding = cls.ClassPadding('', '')
        try:
            function_padding = cls.FunctionPadding(*(int(padding) for padding in tuple(docstyle_settings['function_padding'])))
        except IndexError:
            function_padding = cls.FunctionPadding('', '')
        try:
            docstring_type_regex = cls.DocstringTypeRegex(*(str(sign) for sign in tuple(docstyle_settings['docstring_type_regex'])))
        except IndexError:
            docstring_type_regex = cls.DocstringTypeRegex('', '')
        try:
            docstring_position = docstyle_settings['docstring_position'].value
        except IndexError:
            docstring_position = ''
        ignore_keys = ('class_padding', 'function_padding', 'docstring_type_regex', 'docstring_position') + metadata_settings
        marker_sets = (tuple(value) for (key, value) in docstyle_settings.contents.items() if key not in ignore_keys and (not key.startswith('comment')))
        return cls(language, docstyle, marker_sets, metadata, class_padding, function_padding, docstring_type_regex, docstring_position)

    @staticmethod
    def get_available_definitions():
        if False:
            i = 10
            return i + 15
        '\n        Returns a sequence of pairs with ``(docstyle, language)`` which are\n        available when using ``load()``.\n\n        :return: A sequence of pairs with ``(docstyle, language)``.\n        '
        pattern = os.path.join(os.path.dirname(__file__), '*.coalang')
        for coalang_file in iglob(pattern):
            docstyle = os.path.splitext(os.path.basename(coalang_file))[0]
            if docstyle.lower() == docstyle:
                parser = ConfParser(remove_empty_iter_elements=False)
                for language in parser.parse(coalang_file):
                    yield (docstyle, language.lower())