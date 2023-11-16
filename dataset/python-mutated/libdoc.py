"""Module implementing the command line entry point for the Libdoc tool.

This module can be executed from the command line using the following
approaches::

    python -m robot.libdoc
    python path/to/robot/libdoc.py

This module also exposes the following public API:

- :func:`libdoc_cli` function for simple command line tools.
- :func:`libdoc` function as a high level programmatic API.
- :func:`~robot.libdocpkg.builder.LibraryDocumentation` as the API to generate
  :class:`~robot.libdocpkg.model.LibraryDoc` instances.

Libdoc itself is implemented in the :mod:`~robot.libdocpkg` package.
"""
import sys
from pathlib import Path
if __name__ == '__main__' and 'robot' not in sys.modules:
    import pythonpathsetter
from robot.utils import Application, seq2str
from robot.errors import DataError
from robot.libdocpkg import LibraryDocumentation, ConsoleViewer
USAGE = "Libdoc -- Robot Framework library documentation generator\n\nVersion:  <VERSION>\n\nUsage:  libdoc [options] library_or_resource output_file\n   or:  libdoc [options] library_or_resource list|show|version [names]\n\nLibdoc can generate documentation for Robot Framework libraries and resource\nfiles. It can generate HTML documentation for humans as well as machine\nreadable spec files in XML and JSON formats. Libdoc also has few special\ncommands to show library or resource information on the console.\n\nLibdoc supports all library and resource types and also earlier generated XML\nand JSON specs can be used as input. If a library needs arguments, they must be\ngiven as part of the library name and separated by two colons, for example,\nlike `LibraryName::arg1::arg2`.\n\nThe easiest way to run Libdoc is using the `libdoc` command created as part of\nthe normal installation. Alternatively it is possible to execute the\n`robot.libdoc` module directly like `python -m robot.libdoc`, where `python`\ncan be replaced with any supported Python interpreter. Yet another alternative\nis running the module as a script like `python path/to/robot/libdoc.py`.\n\nThe separate `libdoc` command and the support for JSON spec files are new in\nRobot Framework 4.0.\n\nOptions\n=======\n\n -f --format HTML|XML|JSON|LIBSPEC\n                          Specifies whether to generate an HTML output for\n                          humans or a machine readable spec file in XML or JSON\n                          format. The LIBSPEC format means XML spec with\n                          documentations converted to HTML. The default format\n                          is got from the output file extension.\n -s --specdocformat RAW|HTML\n                          Specifies the documentation format used with XML and\n                          JSON spec files. RAW means preserving the original\n                          documentation format and HTML means converting\n                          documentation to HTML. The default is RAW with XML\n                          spec files and HTML with JSON specs and when using\n                          the special LIBSPEC format. New in RF 4.0.\n -F --docformat ROBOT|HTML|TEXT|REST\n                          Specifies the source documentation format. Possible\n                          values are Robot Framework's documentation format,\n                          HTML, plain text, and reStructuredText. The default\n                          value can be specified in library source code and\n                          the initial default value is ROBOT.\n    --theme DARK|LIGHT|NONE\n                          Use dark or light HTML theme. If this option is not\n                          used, or the value is NONE, the theme is selected\n                          based on the browser color scheme. New in RF 6.0.\n -n --name name           Sets the name of the documented library or resource.\n -v --version version     Sets the version of the documented library or\n                          resource.\n    --quiet               Do not print the path of the generated output file\n                          to the console. New in RF 4.0.\n -P --pythonpath path *   Additional locations where to search for libraries\n                          and resources.\n -h -? --help             Print this help.\n\nCreating documentation\n======================\n\nWhen creating documentation in HTML, XML or JSON format, the output file must\nbe specified as the second argument after the library or resource name or path.\n\nOutput format is got automatically from the output file extension. In addition\nto `*.html`, `*.xml` and `*.json` extensions, it is possible to use the special\n`*.libspec` extensions which means XML spec with actual library and keyword\ndocumentation converted to HTML. The format can also be set explicitly using\nthe `--format` option.\n\nExamples:\n\n  libdoc src/MyLibrary.py doc/MyLibrary.html\n  libdoc doc/MyLibrary.json doc/MyLibrary.html\n  libdoc --name MyLibrary Remote::10.0.0.42:8270 MyLibrary.xml\n  libdoc MyLibrary MyLibrary.libspec\n\nViewing information on console\n==============================\n\nLibdoc has three special commands to show information on the console. These\ncommands are used instead of the name of the output file, and they can also\ntake additional arguments.\n\nlist:    List names of the keywords the library/resource contains. Can be\n         limited to show only certain keywords by passing optional patterns as\n         arguments. Keyword is listed if its name contains any given pattern.\nshow:    Show library/resource documentation. Can be limited to show only\n         certain keywords by passing names as arguments. Keyword is shown if\n         its name matches any given name. Special argument `intro` will show\n         the library introduction and importing sections.\nversion: Show library version\n\nOptional patterns given to `list` and `show` are case and space insensitive.\nBoth also accept `*` and `?` as wildcards.\n\nExamples:\n\n  libdoc Dialogs list\n  libdoc SeleniumLibrary list browser\n  libdoc Remote::10.0.0.42:8270 show\n  libdoc Dialogs show PauseExecution execute*\n  libdoc SeleniumLibrary show intro\n  libdoc SeleniumLibrary version\n\nAlternative execution\n=====================\n\nLibdoc works with all interpreters supported by Robot Framework.\n In the examples above Libdoc is executed as an\ninstalled module, but it can also be executed as a script like\n`python path/robot/libdoc.py`.\n\nFor more information about Libdoc and other built-in tools, see\nhttp://robotframework.org/robotframework/#built-in-tools.\n"

class LibDoc(Application):

    def __init__(self):
        if False:
            print('Hello World!')
        Application.__init__(self, USAGE, arg_limits=(2,), auto_version=False)

    def validate(self, options, arguments):
        if False:
            for i in range(10):
                print('nop')
        if ConsoleViewer.handles(arguments[1]):
            ConsoleViewer.validate_command(arguments[1], arguments[2:])
            return (options, arguments)
        if len(arguments) > 2:
            raise DataError('Only two arguments allowed when writing output.')
        return (options, arguments)

    def main(self, args, name='', version='', format=None, docformat=None, specdocformat=None, theme=None, pythonpath=None, quiet=False):
        if False:
            for i in range(10):
                print('nop')
        if pythonpath:
            sys.path = pythonpath + sys.path
        (lib_or_res, output) = args[:2]
        docformat = self._get_docformat(docformat)
        libdoc = LibraryDocumentation(lib_or_res, name, version, docformat)
        if ConsoleViewer.handles(output):
            ConsoleViewer(libdoc).view(output, *args[2:])
            return
        (format, specdocformat) = self._get_format_and_specdocformat(format, specdocformat, output)
        if format == 'HTML' or specdocformat == 'HTML' or (format in ('JSON', 'LIBSPEC') and specdocformat != 'RAW'):
            libdoc.convert_docs_to_html()
        libdoc.save(output, format, self._validate_theme(theme, format))
        if not quiet:
            self.console(Path(output).absolute())

    def _get_docformat(self, docformat):
        if False:
            while True:
                i = 10
        return self._validate('Doc format', docformat, 'ROBOT', 'TEXT', 'HTML', 'REST')

    def _get_format_and_specdocformat(self, format, specdocformat, output):
        if False:
            print('Hello World!')
        extension = Path(output).suffix[1:]
        format = self._validate('Format', format or extension, 'HTML', 'XML', 'JSON', 'LIBSPEC', allow_none=False)
        specdocformat = self._validate('Spec doc format', specdocformat, 'RAW', 'HTML')
        if format == 'HTML' and specdocformat:
            raise DataError('The --specdocformat option is not applicable with HTML outputs.')
        return (format, specdocformat)

    def _validate(self, kind, value, *valid, allow_none=True):
        if False:
            i = 10
            return i + 15
        if value:
            value = value.upper()
        elif allow_none:
            return None
        if value not in valid:
            raise DataError(f"{kind} must be {seq2str(valid, lastsep=' or ')}, got '{value}'.")
        return value

    def _validate_theme(self, theme, format):
        if False:
            print('Hello World!')
        theme = self._validate('Theme', theme, 'DARK', 'LIGHT', 'NONE')
        if not theme or theme == 'NONE':
            return None
        if format != 'HTML':
            raise DataError('The --theme option is only applicable with HTML outputs.')
        return theme

def libdoc_cli(arguments=None, exit=True):
    if False:
        print('Hello World!')
    "Executes Libdoc similarly as from the command line.\n\n    :param arguments: Command line options and arguments as a list of strings.\n        Starting from RF 4.0, defaults to ``sys.argv[1:]`` if not given.\n    :param exit: If ``True``, call ``sys.exit`` automatically. New in RF 4.0.\n\n    The :func:`libdoc` function may work better in programmatic usage.\n\n    Example::\n\n        from robot.libdoc import libdoc_cli\n\n        libdoc_cli(['--version', '1.0', 'MyLibrary.py', 'MyLibrary.html'])\n    "
    if arguments is None:
        arguments = sys.argv[1:]
    LibDoc().execute_cli(arguments, exit=exit)

def libdoc(library_or_resource, outfile, name='', version='', format=None, docformat=None, specdocformat=None, quiet=False):
    if False:
        i = 10
        return i + 15
    "Executes Libdoc.\n\n    :param library_or_resource: Name or path of the library or resource\n        file to be documented.\n    :param outfile: Path to the file where to write outputs.\n    :param name: Custom name to give to the documented library or resource.\n    :param version: Version to give to the documented library or resource.\n    :param format: Specifies whether to generate HTML, XML or JSON output.\n        If this options is not used, the format is got from the extension of\n        the output file. Possible values are ``'HTML'``, ``'XML'``, ``'JSON'``\n        and ``'LIBSPEC'``.\n    :param docformat: Documentation source format. Possible values are\n        ``'ROBOT'``, ``'reST'``, ``'HTML'`` and ``'TEXT'``. The default value\n        can be specified in library source code and the initial default\n        is ``'ROBOT'``.\n    :param specdocformat: Specifies whether the keyword documentation in spec\n        files is converted to HTML regardless of the original documentation\n        format. Possible values are ``'HTML'`` (convert to HTML) and ``'RAW'``\n        (use original format). The default depends on the output format.\n        New in Robot Framework 4.0.\n    :param quiet: When true, the path of the generated output file is not\n        printed the console. New in Robot Framework 4.0.\n\n    Arguments have same semantics as Libdoc command line options with same names.\n    Run ``libdoc --help`` or consult the Libdoc section in the Robot Framework\n    User Guide for more details.\n\n    Example::\n\n        from robot.libdoc import libdoc\n\n        libdoc('MyLibrary.py', 'MyLibrary.html', version='1.0')\n    "
    return LibDoc().execute(library_or_resource, outfile, name=name, version=version, format=format, docformat=docformat, specdocformat=specdocformat, quiet=quiet)
if __name__ == '__main__':
    libdoc_cli(sys.argv[1:])