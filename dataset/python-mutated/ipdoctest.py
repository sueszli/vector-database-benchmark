"""Nose Plugin that supports IPython doctests.

Limitations:

- When generating examples for use as doctests, make sure that you have
  pretty-printing OFF.  This can be done either by setting the
  ``PlainTextFormatter.pprint`` option in your configuration file to  False, or
  by interactively disabling it with  %Pprint.  This is required so that IPython
  output matches that of normal Python, which is used by doctest for internal
  execution.

- Do not rely on specific prompt numbers for results (such as using
  '_34==True', for example).  For IPython tests run via an external process the
  prompt numbers may be different, and IPython tests run as normal python code
  won't even have these special _NN variables set at all.
"""
import doctest
import logging
import re
from testpath import modified_env
log = logging.getLogger(__name__)

class DocTestFinder(doctest.DocTestFinder):

    def _get_test(self, obj, name, module, globs, source_lines):
        if False:
            print('Hello World!')
        test = super()._get_test(obj, name, module, globs, source_lines)
        if bool(getattr(obj, '__skip_doctest__', False)) and test is not None:
            for example in test.examples:
                example.options[doctest.SKIP] = True
        return test

class IPDoctestOutputChecker(doctest.OutputChecker):
    """Second-chance checker with support for random tests.

    If the default comparison doesn't pass, this checker looks in the expected
    output string for flags that tell us to ignore the output.
    """
    random_re = re.compile('#\\s*random\\s+')

    def check_output(self, want, got, optionflags):
        if False:
            for i in range(10):
                print('nop')
        "Check output, accepting special markers embedded in the output.\n\n        If the output didn't pass the default validation but the special string\n        '#random' is included, we accept it."
        ret = doctest.OutputChecker.check_output(self, want, got, optionflags)
        if not ret and self.random_re.search(want):
            return True
        return ret

class IPExample(doctest.Example):
    pass

class IPDocTestParser(doctest.DocTestParser):
    """
    A class used to parse strings containing doctest examples.

    Note: This is a version modified to properly recognize IPython input and
    convert any IPython examples into valid Python ones.
    """
    _PS1_PY = '>>>'
    _PS2_PY = '\\.\\.\\.'
    _PS1_IP = 'In\\ \\[\\d+\\]:'
    _PS2_IP = '\\ \\ \\ \\.\\.\\.+:'
    _RE_TPL = '\n        # Source consists of a PS1 line followed by zero or more PS2 lines.\n        (?P<source>\n            (?:^(?P<indent> [ ]*) (?P<ps1> %s) .*)    # PS1 line\n            (?:\\n           [ ]*  (?P<ps2> %s) .*)*)  # PS2 lines\n        \\n? # a newline\n        # Want consists of any non-blank lines that do not start with PS1.\n        (?P<want> (?:(?![ ]*$)    # Not a blank line\n                     (?![ ]*%s)   # Not a line starting with PS1\n                     (?![ ]*%s)   # Not a line starting with PS2\n                     .*$\\n?       # But any other line\n                  )*)\n                  '
    _EXAMPLE_RE_PY = re.compile(_RE_TPL % (_PS1_PY, _PS2_PY, _PS1_PY, _PS2_PY), re.MULTILINE | re.VERBOSE)
    _EXAMPLE_RE_IP = re.compile(_RE_TPL % (_PS1_IP, _PS2_IP, _PS1_IP, _PS2_IP), re.MULTILINE | re.VERBOSE)
    _RANDOM_TEST = re.compile('#\\s*all-random\\s+')

    def ip2py(self, source):
        if False:
            return 10
        'Convert input IPython source into valid Python.'
        block = _ip.input_transformer_manager.transform_cell(source)
        if len(block.splitlines()) == 1:
            return _ip.prefilter(block)
        else:
            return block

    def parse(self, string, name='<string>'):
        if False:
            print('Hello World!')
        '\n        Divide the given string into examples and intervening text,\n        and return them as a list of alternating Examples and strings.\n        Line numbers for the Examples are 0-based.  The optional\n        argument `name` is a name identifying this string, and is only\n        used for error messages.\n        '
        string = string.expandtabs()
        min_indent = self._min_indent(string)
        if min_indent > 0:
            string = '\n'.join([l[min_indent:] for l in string.split('\n')])
        output = []
        (charno, lineno) = (0, 0)
        if self._RANDOM_TEST.search(string):
            random_marker = '\n# random'
        else:
            random_marker = ''
        ip2py = False
        terms = list(self._EXAMPLE_RE_PY.finditer(string))
        if terms:
            Example = doctest.Example
        else:
            terms = list(self._EXAMPLE_RE_IP.finditer(string))
            Example = IPExample
            ip2py = True
        for m in terms:
            output.append(string[charno:m.start()])
            lineno += string.count('\n', charno, m.start())
            (source, options, want, exc_msg) = self._parse_example(m, name, lineno, ip2py)
            want += random_marker
            if not self._IS_BLANK_OR_COMMENT(source):
                output.append(Example(source, want, exc_msg, lineno=lineno, indent=min_indent + len(m.group('indent')), options=options))
            lineno += string.count('\n', m.start(), m.end())
            charno = m.end()
        output.append(string[charno:])
        return output

    def _parse_example(self, m, name, lineno, ip2py=False):
        if False:
            i = 10
            return i + 15
        "\n        Given a regular expression match from `_EXAMPLE_RE` (`m`),\n        return a pair `(source, want)`, where `source` is the matched\n        example's source code (with prompts and indentation stripped);\n        and `want` is the example's expected output (with indentation\n        stripped).\n\n        `name` is the string's name, and `lineno` is the line number\n        where the example starts; both are used for error messages.\n\n        Optional:\n        `ip2py`: if true, filter the input via IPython to convert the syntax\n        into valid python.\n        "
        indent = len(m.group('indent'))
        source_lines = m.group('source').split('\n')
        ps1 = m.group('ps1')
        ps2 = m.group('ps2')
        ps1_len = len(ps1)
        self._check_prompt_blank(source_lines, indent, name, lineno, ps1_len)
        if ps2:
            self._check_prefix(source_lines[1:], ' ' * indent + ps2, name, lineno)
        source = '\n'.join([sl[indent + ps1_len + 1:] for sl in source_lines])
        if ip2py:
            source = self.ip2py(source)
        want = m.group('want')
        want_lines = want.split('\n')
        if len(want_lines) > 1 and re.match(' *$', want_lines[-1]):
            del want_lines[-1]
        self._check_prefix(want_lines, ' ' * indent, name, lineno + len(source_lines))
        want_lines[0] = re.sub('Out\\[\\d+\\]: \\s*?\\n?', '', want_lines[0])
        want = '\n'.join([wl[indent:] for wl in want_lines])
        m = self._EXCEPTION_RE.match(want)
        if m:
            exc_msg = m.group('msg')
        else:
            exc_msg = None
        options = self._find_options(source, name, lineno)
        return (source, options, want, exc_msg)

    def _check_prompt_blank(self, lines, indent, name, lineno, ps1_len):
        if False:
            while True:
                i = 10
        '\n        Given the lines of a source string (including prompts and\n        leading indentation), check to make sure that every prompt is\n        followed by a space character.  If any line is not followed by\n        a space character, then raise ValueError.\n\n        Note: IPython-modified version which takes the input prompt length as a\n        parameter, so that prompts of variable length can be dealt with.\n        '
        space_idx = indent + ps1_len
        min_len = space_idx + 1
        for (i, line) in enumerate(lines):
            if len(line) >= min_len and line[space_idx] != ' ':
                raise ValueError('line %r of the docstring for %s lacks blank after %s: %r' % (lineno + i + 1, name, line[indent:space_idx], line))
SKIP = doctest.register_optionflag('SKIP')

class IPDocTestRunner(doctest.DocTestRunner, object):
    """Test runner that synchronizes the IPython namespace with test globals.
    """

    def run(self, test, compileflags=None, out=None, clear_globs=True):
        if False:
            print('Hello World!')
        with modified_env({'COLUMNS': '80', 'LINES': '24'}):
            return super(IPDocTestRunner, self).run(test, compileflags, out, clear_globs)