from coalib.bearlib.languages.documentation.DocstyleDefinition import DocstyleDefinition
from coalib.results.Diff import Diff
from coalib.results.TextRange import TextRange
from coalib.bearlib.languages.documentation.DocumentationExtraction import extract_documentation_with_markers

class DocBaseClass:
    """
    DocBaseClass holds important functions which will extract, parse
    and generates diffs for documentation. All bears that processes
    documentation should inherit from this.
    """

    @staticmethod
    def extract(content, language, docstyle):
        if False:
            i = 10
            return i + 15
        '\n        Extracts all documentation texts inside the given source-code-string\n        using the coala docstyle definition files.\n\n        The documentation texts are sorted by their order appearing in\n        ``content``.\n\n        For more information about how documentation comments are\n        identified and extracted, see DocstyleDefinition.doctypes enumeration.\n\n        :param content:            The source-code-string where to extract\n                                   documentation from. Needs to be a list\n                                   or tuple where each string item is a\n                                   single line(including ending whitespaces\n                                   like ``\\n``).\n        :param language:           The programming language used.\n        :param docstyle:           The documentation style/tool used\n                                   (e.g. doxygen).\n        :raises FileNotFoundError: Raised when the docstyle definition file\n                                   was not found.\n        :raises KeyError:          Raised when the given language is not\n                                   defined in given docstyle.\n        :raises ValueError:        Raised when a docstyle definition setting\n                                   has an invalid format.\n        :return:                   An iterator returning instances of\n                                   DocumentationComment or MalformedComment\n                                   found in the content.\n        '
        docstyle_definition = DocstyleDefinition.load(language, docstyle)
        return extract_documentation_with_markers(content, docstyle_definition)

    @staticmethod
    def generate_diff(file, doc_comment, new_comment):
        if False:
            return 10
        '\n        Generates diff between the original doc_comment and its fix\n        new_comment which are instances of DocumentationComment.\n\n        :param doc_comment:\n            Original instance of DocumentationComment.\n        :param new_comment:\n            Fixed instance of DocumentationComment.\n        :return:\n            Diff instance.\n        '
        diff = Diff(file)
        old_range = TextRange.from_values(doc_comment.range.start.line, 1, doc_comment.range.end.line, doc_comment.range.end.column)
        new_comment.assemble.cache_clear()
        diff.replace(old_range, new_comment.assemble())
        return diff

    def process_documentation(self, *args, **kwargs):
        if False:
            return 10
        '\n        Checks and handles the fixing part of documentation.\n\n        :return:\n            A tuple of processed documentation and warning_desc.\n        '
        raise NotImplementedError('This function has to be implemented for a documentation bear.')