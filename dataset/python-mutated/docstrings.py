"""Docstring parsing module for Python Fire.

The following features of docstrings are not supported.
TODO(dbieber): Support these features.
- numpy docstrings may begin with the function signature.
- whitespace may be important for proper structuring of a docstring
- I've seen `argname` (with single backticks) as a style of documenting
  arguments. The `argname` appears on one line, and the description on the next.
- .. Sphinx directives such as .. note:: are not understood.
- After a section ends, future contents may be included in the section. E.g.
  :returns: This is what is returned.
  Example: An example goes here.
- @param is sometimes used.  E.g.
  @param argname (type) Description
  @return (type) Description
- The true signature of a function is not used by the docstring parser. It could
  be useful for determining whether something is a section header or an argument
  for example.
- This example confuses types as part of the docstrings.
  Parameters
  argname : argtype
  Arg description
- If there's no blank line after the summary, the description will be slurped
  up into the summary.
- "Examples" should be its own section type. aka "Usage".
- "Notes" should be a section type.
- Some people put parenthesis around their types in RST format, e.g.
  :param (type) paramname:
- :rtype: directive (return type)
- Also ":rtype str" with no closing ":" has come up.
- Return types are not supported.
- "# Returns" as a section title style
- ":raises ExceptionType: Description" ignores the ExceptionType currently.
- "Defaults to X" occurs sometimes.
- "True | False" indicates bool type.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import re
import textwrap

class DocstringInfo(collections.namedtuple('DocstringInfo', ('summary', 'description', 'args', 'returns', 'yields', 'raises'))):
    pass
DocstringInfo.__new__.__defaults__ = (None,) * len(DocstringInfo._fields)

class ArgInfo(collections.namedtuple('ArgInfo', ('name', 'type', 'description'))):
    pass
ArgInfo.__new__.__defaults__ = (None,) * len(ArgInfo._fields)

class KwargInfo(ArgInfo):
    pass
KwargInfo.__new__.__defaults__ = (None,) * len(KwargInfo._fields)

class Namespace(dict):
    """A dict with attribute (dot-notation) access enabled."""

    def __getattr__(self, key):
        if False:
            i = 10
            return i + 15
        if key not in self:
            self[key] = Namespace()
        return self[key]

    def __setattr__(self, key, value):
        if False:
            print('Hello World!')
        self[key] = value

    def __delattr__(self, key):
        if False:
            return 10
        if key in self:
            del self[key]

class Sections(enum.Enum):
    ARGS = 0
    RETURNS = 1
    YIELDS = 2
    RAISES = 3
    TYPE = 4

class Formats(enum.Enum):
    GOOGLE = 0
    NUMPY = 1
    RST = 2
SECTION_TITLES = {Sections.ARGS: ('argument', 'arg', 'parameter', 'param', 'key'), Sections.RETURNS: ('return',), Sections.YIELDS: ('yield',), Sections.RAISES: ('raise', 'except', 'exception', 'throw', 'error', 'warn'), Sections.TYPE: ('type',)}

def parse(docstring):
    if False:
        while True:
            i = 10
    'Returns DocstringInfo about the given docstring.\n\n  This parser aims to parse Google, numpy, and rst formatted docstrings. These\n  are the three most common docstring styles at the time of this writing.\n\n  This parser aims to be permissive, working even when the docstring deviates\n  from the strict recommendations of these styles.\n\n  This parser does not aim to fully extract all structured information from a\n  docstring, since there are simply too many ways to structure information in a\n  docstring. Sometimes content will remain as unstructured text and simply gets\n  included in the description.\n\n  The Google docstring style guide is available at:\n  https://github.com/google/styleguide/blob/gh-pages/pyguide.md\n\n  The numpy docstring style guide is available at:\n  https://numpydoc.readthedocs.io/en/latest/format.html\n\n  Information about the rST docstring format is available at:\n  https://www.python.org/dev/peps/pep-0287/\n  The full set of directives such as param and type for rST docstrings are at:\n  http://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html\n\n  Note: This function does not claim to handle all docstrings well. A list of\n  limitations is available at the top of the file. It does aim to run without\n  crashing in O(n) time on all strings on length n. If you find a string that\n  causes this to crash or run unacceptably slowly, please consider submitting\n  a pull request.\n\n  Args:\n    docstring: The docstring to parse.\n\n  Returns:\n    A DocstringInfo containing information about the docstring.\n  '
    if docstring is None:
        return DocstringInfo()
    lines = docstring.strip().split('\n')
    lines_len = len(lines)
    state = Namespace()
    state.section.title = None
    state.section.indentation = None
    state.section.line1_indentation = None
    state.section.format = None
    state.summary.permitted = True
    state.summary.lines = []
    state.description.lines = []
    state.args = []
    state.kwargs = []
    state.current_arg = None
    state.returns.lines = []
    state.yields.lines = []
    state.raises.lines = []
    for (index, line) in enumerate(lines):
        has_next = index + 1 < lines_len
        previous_line = lines[index - 1] if index > 0 else None
        next_line = lines[index + 1] if has_next else None
        line_info = _create_line_info(line, next_line, previous_line)
        _consume_line(line_info, state)
    summary = ' '.join(state.summary.lines) if state.summary.lines else None
    state.description.lines = _strip_blank_lines(state.description.lines)
    description = textwrap.dedent('\n'.join(state.description.lines))
    if not description:
        description = None
    returns = _join_lines(state.returns.lines)
    yields = _join_lines(state.yields.lines)
    raises = _join_lines(state.raises.lines)
    args = [ArgInfo(name=arg.name, type=_cast_to_known_type(_join_lines(arg.type.lines)), description=_join_lines(arg.description.lines)) for arg in state.args]
    args.extend([KwargInfo(name=arg.name, type=_cast_to_known_type(_join_lines(arg.type.lines)), description=_join_lines(arg.description.lines)) for arg in state.kwargs])
    return DocstringInfo(summary=summary, description=description, args=args or None, returns=returns, raises=raises, yields=yields)

def _strip_blank_lines(lines):
    if False:
        print('Hello World!')
    'Removes lines containing only blank characters before and after the text.\n\n  Args:\n    lines: A list of lines.\n  Returns:\n    A list of lines without trailing or leading blank lines.\n  '
    start = 0
    num_lines = len(lines)
    while lines and start < num_lines and _is_blank(lines[start]):
        start += 1
    lines = lines[start:]
    while lines and _is_blank(lines[-1]):
        lines.pop()
    return lines

def _is_blank(line):
    if False:
        i = 10
        return i + 15
    return not line or line.isspace()

def _join_lines(lines):
    if False:
        i = 10
        return i + 15
    "Joins lines with the appropriate connective whitespace.\n\n  This puts a single space between consecutive lines, unless there's a blank\n  line, in which case a full blank line is included.\n\n  Args:\n    lines: A list of lines to join.\n  Returns:\n    A string, the lines joined together.\n  "
    if not lines:
        return None
    started = False
    group_texts = []
    group_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line:
            started = True
            group_lines.append(stripped_line)
        elif started:
            group_text = ' '.join(group_lines)
            group_texts.append(group_text)
            group_lines = []
    if group_lines:
        group_text = ' '.join(group_lines)
        group_texts.append(group_text)
    return '\n\n'.join(group_texts)

def _get_or_create_arg_by_name(state, name, is_kwarg=False):
    if False:
        for i in range(10):
            print('nop')
    "Gets or creates a new Arg.\n\n  These Arg objects (Namespaces) are turned into the ArgInfo namedtuples\n  returned by parse. Each Arg object is used to collect the name, type, and\n  description of a single argument to the docstring's function.\n\n  Args:\n    state: The state of the parser.\n    name: The name of the arg to create.\n    is_kwarg: A boolean representing whether the argument is a keyword arg.\n  Returns:\n    The new Arg.\n  "
    for arg in state.args + state.kwargs:
        if arg.name == name:
            return arg
    arg = Namespace()
    arg.name = name
    arg.type.lines = []
    arg.description.lines = []
    if is_kwarg:
        state.kwargs.append(arg)
    else:
        state.args.append(arg)
    return arg

def _is_arg_name(name):
    if False:
        print('Hello World!')
    "Returns whether name is a valid arg name.\n\n  This is used to prevent multiple words (plaintext) from being misinterpreted\n  as an argument name. Any line that doesn't match the pattern for a valid\n  argument is treated as not being an argument.\n\n  Args:\n    name: The name of the potential arg.\n  Returns:\n    True if name looks like an arg name, False otherwise.\n  "
    name = name.strip()
    arg_pattern = '^[a-zA-Z_]\\w*$'
    re.match(arg_pattern, name)
    return re.match(arg_pattern, name) is not None

def _as_arg_name_and_type(text):
    if False:
        i = 10
        return i + 15
    'Returns text as a name and type, if text looks like an arg name and type.\n\n  Example:\n    _as_arg_name_and_type("foo (int)") == "foo", "int"\n\n  Args:\n    text: The text, which may or may not be an arg name and type.\n  Returns:\n    The arg name and type, if text looks like an arg name and type.\n    None otherwise.\n  '
    tokens = text.split()
    if len(tokens) < 2:
        return None
    if _is_arg_name(tokens[0]):
        type_token = ' '.join(tokens[1:])
        type_token = type_token.lstrip('{([').rstrip('])}')
        return (tokens[0], type_token)
    else:
        return None

def _as_arg_names(names_str):
    if False:
        return 10
    'Converts names_str to a list of arg names.\n\n  Example:\n    _as_arg_names("a, b, c") == ["a", "b", "c"]\n\n  Args:\n    names_str: A string with multiple space or comma separated arg names.\n  Returns:\n    A list of arg names, or None if names_str doesn\'t look like a list of arg\n    names.\n  '
    names = re.split(',| ', names_str)
    names = [name.strip() for name in names if name.strip()]
    for name in names:
        if not _is_arg_name(name):
            return None
    if not names:
        return None
    return names

def _cast_to_known_type(name):
    if False:
        print('Hello World!')
    'Canonicalizes a string representing a type if possible.\n\n  # TODO(dbieber): Support additional canonicalization, such as string/str, and\n  # boolean/bool.\n\n  Example:\n    _cast_to_known_type("str.") == "str"\n\n  Args:\n    name: A string representing a type, or None.\n  Returns:\n    A canonicalized version of the type string.\n  '
    if name is None:
        return None
    return name.rstrip('.')

def _consume_google_args_line(line_info, state):
    if False:
        return 10
    'Consume a single line from a Google args section.'
    split_line = line_info.remaining.split(':', 1)
    if len(split_line) > 1:
        (first, second) = split_line
        if _is_arg_name(first.strip()):
            arg = _get_or_create_arg_by_name(state, first.strip())
            arg.description.lines.append(second.strip())
            state.current_arg = arg
        else:
            arg_name_and_type = _as_arg_name_and_type(first)
            if arg_name_and_type:
                (arg_name, type_str) = arg_name_and_type
                arg = _get_or_create_arg_by_name(state, arg_name)
                arg.type.lines.append(type_str)
                arg.description.lines.append(second.strip())
                state.current_arg = arg
            elif state.current_arg:
                state.current_arg.description.lines.append(split_line[0])
    elif state.current_arg:
        state.current_arg.description.lines.append(split_line[0])

def _consume_line(line_info, state):
    if False:
        i = 10
        return i + 15
    'Consumes one line of text, updating the state accordingly.\n\n  When _consume_line is called, part of the line may already have been processed\n  for header information.\n\n  Args:\n    line_info: Information about the current and next line of the docstring.\n    state: The state of the docstring parser.\n  '
    _update_section_state(line_info, state)
    if state.section.title is None:
        if state.summary.permitted:
            if line_info.remaining:
                state.summary.lines.append(line_info.remaining)
            elif state.summary.lines:
                state.summary.permitted = False
        else:
            state.description.lines.append(line_info.remaining_raw)
    else:
        state.summary.permitted = False
    if state.section.new and state.section.format == Formats.RST:
        directive = _get_directive(line_info)
        directive_tokens = directive.split()
        if state.section.title == Sections.ARGS:
            name = directive_tokens[-1]
            arg = _get_or_create_arg_by_name(state, name, is_kwarg=directive_tokens[0] == 'key')
            if len(directive_tokens) == 3:
                arg.type.lines.append(directive_tokens[1])
            state.current_arg = arg
        elif state.section.title == Sections.TYPE:
            name = directive_tokens[-1]
            arg = _get_or_create_arg_by_name(state, name)
            state.current_arg = arg
    if state.section.format == Formats.NUMPY and _line_is_hyphens(line_info.remaining):
        return
    if state.section.title == Sections.ARGS:
        if state.section.format == Formats.GOOGLE:
            _consume_google_args_line(line_info, state)
        elif state.section.format == Formats.RST:
            state.current_arg.description.lines.append(line_info.remaining.strip())
        elif state.section.format == Formats.NUMPY:
            line_stripped = line_info.remaining.strip()
            if _is_arg_name(line_stripped):
                arg = _get_or_create_arg_by_name(state, line_stripped)
                state.current_arg = arg
            elif _line_is_numpy_parameter_type(line_info):
                (possible_args, type_data) = line_stripped.split(':', 1)
                arg_names = _as_arg_names(possible_args)
                if arg_names:
                    for arg_name in arg_names:
                        arg = _get_or_create_arg_by_name(state, arg_name)
                        arg.type.lines.append(type_data)
                        state.current_arg = arg
                elif state.current_arg:
                    state.current_arg.description.lines.append(line_info.remaining.strip())
                else:
                    pass
            elif state.current_arg:
                state.current_arg.description.lines.append(line_info.remaining.strip())
            else:
                pass
    elif state.section.title == Sections.RETURNS:
        state.returns.lines.append(line_info.remaining.strip())
    elif state.section.title == Sections.YIELDS:
        state.yields.lines.append(line_info.remaining.strip())
    elif state.section.title == Sections.RAISES:
        state.raises.lines.append(line_info.remaining.strip())
    elif state.section.title == Sections.TYPE:
        if state.section.format == Formats.RST:
            assert state.current_arg is not None
            state.current_arg.type.lines.append(line_info.remaining.strip())
        else:
            pass

def _create_line_info(line, next_line, previous_line):
    if False:
        print('Hello World!')
    'Returns information about the current line and surrounding lines.'
    line_info = Namespace()
    line_info.line = line
    line_info.stripped = line.strip()
    line_info.remaining_raw = line_info.line
    line_info.remaining = line_info.stripped
    line_info.indentation = len(line) - len(line.lstrip())
    line_info.next.line = next_line
    next_line_exists = next_line is not None
    line_info.next.stripped = next_line.strip() if next_line_exists else None
    line_info.next.indentation = len(next_line) - len(next_line.lstrip()) if next_line_exists else None
    line_info.previous.line = previous_line
    previous_line_exists = previous_line is not None
    line_info.previous.indentation = len(previous_line) - len(previous_line.lstrip()) if previous_line_exists else None
    return line_info

def _update_section_state(line_info, state):
    if False:
        for i in range(10):
            print('nop')
    'Uses line_info to determine the current section of the docstring.\n\n  Updates state and line_info.remaining.\n\n  Args:\n    line_info: Information about the current line.\n    state: The state of the parser.\n  '
    section_updated = False
    google_section_permitted = _google_section_permitted(line_info, state)
    google_section = google_section_permitted and _google_section(line_info)
    if google_section:
        state.section.format = Formats.GOOGLE
        state.section.title = google_section
        line_info.remaining = _get_after_google_header(line_info)
        line_info.remaining_raw = line_info.remaining
        section_updated = True
    rst_section = _rst_section(line_info)
    if rst_section:
        state.section.format = Formats.RST
        state.section.title = rst_section
        line_info.remaining = _get_after_directive(line_info)
        line_info.remaining_raw = line_info.remaining
        section_updated = True
    numpy_section = _numpy_section(line_info)
    if numpy_section:
        state.section.format = Formats.NUMPY
        state.section.title = numpy_section
        line_info.remaining = ''
        line_info.remaining_raw = line_info.remaining
        section_updated = True
    if section_updated:
        state.section.new = True
        state.section.indentation = line_info.indentation
        state.section.line1_indentation = line_info.next.indentation
    else:
        state.section.new = False

def _google_section_permitted(line_info, state):
    if False:
        while True:
            i = 10
    'Returns whether a new google section is permitted to start here.\n\n  Q: Why might a new Google section not be allowed?\n  A: If we\'re in the middle of a Google "Args" section, then lines that start\n  "param:" will usually be a new arg, rather than a new section.\n  We use whitespace to determine when the Args section has actually ended.\n\n  A Google section ends when either:\n  - A new google section begins at either\n    - indentation less than indentation of line 1 of the previous section\n    - or <= indentation of the previous section\n  - Or the docstring terminates.\n\n  Args:\n    line_info: Information about the current line.\n    state: The state of the parser.\n  Returns:\n    True or False, indicating whether a new Google section is permitted at the\n    current line.\n  '
    if state.section.indentation is None:
        return True
    return line_info.indentation <= state.section.indentation or line_info.indentation < state.section.line1_indentation

def _matches_section_title(title, section_title):
    if False:
        return 10
    "Returns whether title is a match for a specific section_title.\n\n  Example:\n    _matches_section_title('Yields', 'yield') == True\n\n  Args:\n    title: The title to check for matching.\n    section_title: A specific known section title to check against.\n  "
    title = title.lower()
    section_title = section_title.lower()
    return section_title in (title, title[:-1])

def _matches_section(title, section):
    if False:
        for i in range(10):
            print('nop')
    "Returns whether title is a match any known title for a specific section.\n\n  Example:\n    _matches_section_title('Yields', Sections.YIELDS) == True\n    _matches_section_title('param', Sections.Args) == True\n\n  Args:\n    title: The title to check for matching.\n    section: A specific section to check all possible titles for.\n  Returns:\n    True or False, indicating whether title is a match for the specified\n    section.\n  "
    for section_title in SECTION_TITLES[section]:
        if _matches_section_title(title, section_title):
            return True
    return False

def _section_from_possible_title(possible_title):
    if False:
        for i in range(10):
            print('nop')
    'Returns a section matched by the possible title, or None if none match.\n\n  Args:\n    possible_title: A string that may be the title of a new section.\n  Returns:\n    A Section type if one matches, or None if no section type matches.\n  '
    for section in SECTION_TITLES:
        if _matches_section(possible_title, section):
            return section
    return None

def _google_section(line_info):
    if False:
        i = 10
        return i + 15
    'Checks whether the current line is the start of a new Google-style section.\n\n  This docstring is a Google-style docstring. Google-style sections look like\n  this:\n\n    Section Name:\n      section body goes here\n\n  Args:\n    line_info: Information about the current line.\n  Returns:\n    A Section type if one matches, or None if no section type matches.\n  '
    colon_index = line_info.remaining.find(':')
    possible_title = line_info.remaining[:colon_index]
    return _section_from_possible_title(possible_title)

def _get_after_google_header(line_info):
    if False:
        return 10
    'Gets the remainder of the line, after a Google header.'
    colon_index = line_info.remaining.find(':')
    return line_info.remaining[colon_index + 1:]

def _get_directive(line_info):
    if False:
        i = 10
        return i + 15
    'Gets a directive from the start of the line.\n\n  If the line is ":param str foo: Description of foo", then\n  _get_directive(line_info) returns "param str foo".\n\n  Args:\n    line_info: Information about the current line.\n  Returns:\n    The contents of a directive, or None if the line doesn\'t start with a\n    directive.\n  '
    if line_info.stripped.startswith(':'):
        return line_info.stripped.split(':', 2)[1]
    else:
        return None

def _get_after_directive(line_info):
    if False:
        for i in range(10):
            print('nop')
    'Gets the remainder of the line, after a directive.'
    sections = line_info.stripped.split(':', 2)
    if len(sections) > 2:
        return sections[-1]
    else:
        return ''

def _rst_section(line_info):
    if False:
        for i in range(10):
            print('nop')
    'Checks whether the current line is the start of a new RST-style section.\n\n  RST uses directives to specify information. An RST directive, which we refer\n  to as a section here, are surrounded with colons. For example, :param name:.\n\n  Args:\n    line_info: Information about the current line.\n  Returns:\n    A Section type if one matches, or None if no section type matches.\n  '
    directive = _get_directive(line_info)
    if directive:
        possible_title = directive.split()[0]
        return _section_from_possible_title(possible_title)
    else:
        return None

def _line_is_hyphens(line):
    if False:
        print('Hello World!')
    'Returns whether the line is entirely hyphens (and not blank).'
    return line and (not line.strip('-'))

def _numpy_section(line_info):
    if False:
        for i in range(10):
            print('nop')
    'Checks whether the current line is the start of a new numpy-style section.\n\n  Numpy style sections are followed by a full line of hyphens, for example:\n\n    Section Name\n    ------------\n    Section body goes here.\n\n  Args:\n    line_info: Information about the current line.\n  Returns:\n    A Section type if one matches, or None if no section type matches.\n  '
    next_line_is_hyphens = _line_is_hyphens(line_info.next.stripped)
    if next_line_is_hyphens:
        possible_title = line_info.remaining
        return _section_from_possible_title(possible_title)
    else:
        return None

def _line_is_numpy_parameter_type(line_info):
    if False:
        for i in range(10):
            print('nop')
    'Returns whether the line contains a numpy style parameter type definition.\n\n  We look for a line of the form:\n  x : type\n\n  And we have to exclude false positives on argument descriptions containing a\n  colon by checking the indentation of the line above.\n\n  Args:\n    line_info: Information about the current line.\n  Returns:\n    True if the line is a numpy parameter type definition, False otherwise.\n  '
    line_stripped = line_info.remaining.strip()
    if ':' in line_stripped:
        previous_indent = line_info.previous.indentation
        current_indent = line_info.indentation
        if ':' in line_info.previous.line and current_indent > previous_indent:
            return False
        else:
            return True
    return False