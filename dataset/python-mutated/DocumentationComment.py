from collections import namedtuple
from coala_utils.decorators import generate_eq, generate_repr
from coalib.results.TextRange import TextRange
from functools import lru_cache
SPHINX_REF = {'Attribute': (':attr:`', '`'), 'Class': (':class:`', '`'), 'Constant': (':const:`', '`'), 'Data': (':data:`', '`'), 'Exception': (':exc:`', '`'), 'Function': (':func:`', '`'), 'Method': (':meth:`', '`'), 'Module': (':mod:`', '`'), 'Object': (':obj:`', '`')}

def _find_references(line, identifiers):
    if False:
        print('Hello World!')
    '\n    Find references to another object in the given line.\n\n    :param line:\n        String to look into.\n    :param identifiers:\n        A dict mapping the type of references to a tuple of two strings with\n        which the reference might start and end respectively.\n    :return:\n        A list of two-element tuples containing the type of reference (inferred\n        from the given dict) and the corresponding index of occurence in the\n        line, sorted according to the index.\n    '
    occurences = []
    for (ref_type, identifier) in identifiers.items():
        ref = line.find(identifier[0])
        while ref != -1:
            occurences.append((ref_type, ref))
            ref = line.find(identifier[0], ref + 1)
    return sorted(occurences, key=lambda x: x[1])

@generate_repr()
@generate_eq('documentation', 'language', 'docstyle', 'indent', 'marker', 'position')
class DocumentationComment:
    """
    The DocumentationComment holds information about a documentation comment
    inside source-code, like position etc.
    """
    Parameter = namedtuple('Parameter', 'name, desc')
    ExceptionValue = namedtuple('ExceptionValue', 'name, desc')
    ReturnValue = namedtuple('ReturnValue', 'desc')
    Description = namedtuple('Description', 'desc')
    Reference = namedtuple('Reference', ['type_ref', 'ref_addr'])
    top_padding = 0
    bottom_padding = 0
    docstring_type = 'others'

    def __init__(self, documentation, docstyle_definition, indent, marker, position):
        if False:
            while True:
                i = 10
        '\n        Instantiates a new DocumentationComment.\n\n        :param documentation:\n            The documentation text.\n        :param docstyle_definition:\n            The ``DocstyleDefinition`` instance that defines what docstyle is\n            being used in the documentation.\n        :param indent:\n            The string of indentation used in front of the first marker of the\n            documentation.\n        :param marker:\n            The three-element tuple with marker strings, that identified this\n            documentation comment.\n        :param position:\n            The starting ``TextPosition`` of the documentation.\n        '
        self.documentation = documentation
        self.docstyle_definition = docstyle_definition
        self.indent = '' if indent is None else indent
        self.marker = ('', '', '') if marker is None else marker
        self.position = position
        self.range = None if position is None else TextRange.from_values(position.line, position.column, position.line + self.assemble().count('\n'), len(self.assemble()) - self.assemble().rfind('\n'))

    def __str__(self):
        if False:
            print('Hello World!')
        return self.documentation

    @property
    def language(self):
        if False:
            print('Hello World!')
        return self.docstyle_definition.language

    @property
    def docstyle(self):
        if False:
            while True:
                i = 10
        return self.docstyle_definition.docstyle

    @property
    def metadata(self):
        if False:
            while True:
                i = 10
        return self.docstyle_definition.metadata

    def parse(self):
        if False:
            return 10
        '\n        Parses documentation independent of language and docstyle.\n\n        :return:\n            The list of all the parsed sections of the documentation. Every\n            section is a namedtuple of either ``Description`` or ``Parameter``\n            or ``ReturnValue``.\n        :raises NotImplementedError:\n            When no parsing method is present for the given language and\n            docstyle.\n        '
        if self.language == 'python' and self.docstyle == 'default':
            return self._parse_documentation_with_symbols((':param ', ':'), (':raises ', ':'), ':return:', SPHINX_REF)
        elif self.language == 'python' and self.docstyle == 'doxygen':
            return self._parse_documentation_with_symbols(('@param ', ' '), ('@raises ', ' '), '@return ')
        elif self.language == 'java' and self.docstyle == 'default':
            return self._parse_documentation_with_symbols(('@param  ', ' '), ('@raises  ', ' '), '@return ')
        elif self.language == 'golang' and self.docstyle == 'golang':
            return self.documentation.splitlines(keepends=True)
        else:
            raise NotImplementedError('Documentation parsing for {0.language!r} in {0.docstyle!r} has not been implemented yet'.format(self))

    def _parse_documentation_with_symbols(self, param_identifiers, exception_identifiers, return_identifiers, ref_identifiers={}):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parses documentation based on parameter, exception and return symbols.\n\n        :param param_identifiers:\n            A tuple of two strings with which a parameter starts and ends.\n        :param exception_identifiers:\n            A tuple of two strings with which an exception starts and ends.\n        :param return_identifiers:\n            The string with which a return description starts.\n        :return:\n            The list of all the parsed sections of the documentation. Every\n            section is a named tuple of either ``Description``, ``Parameter``,\n            ``ExceptionValue`` or ``ReturnValue``.\n        '
        lines = self.documentation.splitlines(keepends=True)
        parse_mode = self.Description
        cur_param = ''
        desc = ''
        parsed = []
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith(param_identifiers[0]):
                parse_mode = self.Parameter
                param_offset = line.find(param_identifiers[0]) + len(param_identifiers[0])
                splitted = line[param_offset:].split(param_identifiers[1], 1)
                if len(splitted) == 1:
                    splitted = line[param_offset:].split(' ', 1)
                cur_param = splitted[0].strip()
                param_desc = splitted[1]
                parsed.append(self.Parameter(name=cur_param, desc=param_desc))
            elif stripped_line.startswith(exception_identifiers[0]):
                parse_mode = self.ExceptionValue
                exception_offset = line.find(exception_identifiers[0]) + len(exception_identifiers[0])
                splitted = line[exception_offset:].split(exception_identifiers[1], 1)
                if len(splitted) == 1:
                    splitted = line[exception_offset:].split(' ', 1)
                cur_exception = splitted[0].strip()
                exception_desc = splitted[1]
                parsed.append(self.ExceptionValue(name=cur_exception, desc=exception_desc))
            elif stripped_line.startswith(return_identifiers):
                parse_mode = self.ReturnValue
                return_offset = line.find(return_identifiers) + len(return_identifiers)
                retval_desc = line[return_offset:]
                parsed.append(self.ReturnValue(desc=retval_desc))
            elif _find_references(line, ref_identifiers):
                occurences = _find_references(line, ref_identifiers)
                for (ref, _) in occurences:
                    identifier = ref_identifiers[ref]
                    splitted = line.split(identifier[0], 1)[1].split(identifier[1], 1)
                    addr = splitted[0].strip()
                    line = splitted[1:][0]
                    parsed.append(self.Reference(type_ref=ref, ref_addr=addr))
            elif parse_mode == self.ReturnValue:
                retval_desc += line
                parsed.pop()
                parsed.append(self.ReturnValue(desc=retval_desc))
            elif parse_mode == self.ExceptionValue:
                exception_desc += line
                parsed.pop()
                parsed.append(self.ExceptionValue(name=cur_exception, desc=exception_desc))
            elif parse_mode == self.Parameter:
                param_desc += line
                parsed.pop()
                parsed.append(self.Parameter(name=cur_param, desc=param_desc))
            else:
                desc += line
                try:
                    parsed.pop()
                except IndexError:
                    pass
                parsed.append(self.Description(desc=desc))
        return parsed

    @classmethod
    def from_metadata(cls, doccomment, docstyle_definition, marker, indent, position):
        if False:
            while True:
                i = 10
        '\n        Assembles a list of parsed documentation comment metadata.\n\n        This function just assembles the documentation comment\n        itself, without the markers and indentation.\n\n        >>> from coalib.bearlib.languages.documentation.DocumentationComment \\\n        ...     import DocumentationComment\n        >>> from coalib.bearlib.languages.documentation.DocstyleDefinition \\\n        ...     import DocstyleDefinition\n        >>> from coalib.results.TextPosition import TextPosition\n        >>> Description = DocumentationComment.Description\n        >>> Parameter = DocumentationComment.Parameter\n        >>> python_default = DocstyleDefinition.load("python3", "default")\n        >>> parsed_doc = [Description(desc=\'\\nDescription\\n\'),\n        ...               Parameter(name=\'age\', desc=\' Age\\n\')]\n        >>> str(DocumentationComment.from_metadata(\n        ...         parsed_doc, python_default,\n        ...         python_default.markers[0], \'    \',\n        ...         TextPosition(1, 1)))\n        \'\\nDescription\\n:param age: Age\\n\'\n\n        :param doccomment:\n            The list of parsed documentation comment metadata.\n        :param docstyle_definition:\n            The ``DocstyleDefinition`` instance that defines what docstyle is\n            being used in a documentation comment.\n        :param marker:\n            The markers to be used in the documentation comment.\n        :param indent:\n            The indentation to be used in the documentation comment.\n        :param position:\n            The starting position of the documentation comment.\n        :return:\n            A ``DocumentationComment`` instance of the assembled documentation.\n        '
        assembled_doc = ''
        for section in doccomment:
            section_desc = section.desc.splitlines(keepends=True)
            if isinstance(section, cls.Parameter):
                assembled_doc += docstyle_definition.metadata.param_start + section.name + docstyle_definition.metadata.param_end
            elif isinstance(section, cls.ExceptionValue):
                assembled_doc += docstyle_definition.metadata.exception_start + section.name + docstyle_definition.metadata.exception_end
            elif isinstance(section, cls.ReturnValue):
                assembled_doc += docstyle_definition.metadata.return_sep
            assembled_doc += ''.join(section_desc)
        return DocumentationComment(assembled_doc, docstyle_definition, indent, marker, position)

    @lru_cache(maxsize=1)
    def assemble(self):
        if False:
            i = 10
            return i + 15
        '\n        Assembles parsed documentation to the original documentation.\n\n        This function assembles the whole documentation comment, with the\n        given markers and indentation.\n        '
        lines = self.documentation.splitlines(keepends=True)
        assembled = self.indent + self.marker[0]
        if len(lines) == 0:
            return self.marker[0] + self.marker[2]
        assembled += lines[0]
        assembled += ''.join(('\n' if line == '\n' and (not self.marker[1]) else self.indent + self.marker[1] + line for line in lines[1:]))
        assembled = assembled if self.marker[1] == self.marker[2] else assembled + (self.indent if lines[-1][-1] == '\n' else '') + self.marker[2]
        assembled = '\n' * self.top_padding + assembled + '\n' * self.bottom_padding
        return assembled

class MalformedComment:
    """
    The MalformedComment holds information about the errors generated by the
    DocumentationExtraction, DocumentationComment, DocstyleDefinition and
    DocBaseClass.

    When these classes are unable to parse certain docstrings, an instance
    of MalformedComment will be returned instead of DocumentationComment.
    """

    def __init__(self, message, line):
        if False:
            return 10
        '\n        Instantiate a MalformedComment, which contains the information about\n        the error: a message explaining the behaviour and a line no where the\n        error has occured.\n\n        :param message:\n            Contains the message about the error.\n        :param line:\n            Contains the current line number of the docstring where the error\n            has occured.\n        '
        self.message = message
        self.line = line