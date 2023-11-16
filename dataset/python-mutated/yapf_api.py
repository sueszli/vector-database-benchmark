"""Entry points for YAPF.

The main APIs that YAPF exposes to drive the reformatting.

  FormatFile(): reformat a file.
  FormatCode(): reformat a string of code.

These APIs have some common arguments:

  style_config: (string) Either a style name or a path to a file that contains
    formatting style settings. If None is specified, use the default style
    as set in style.DEFAULT_STYLE_FACTORY
  lines: (list of tuples of integers) A list of tuples of lines, [start, end],
    that we want to format. The lines are 1-based indexed. It can be used by
    third-party code (e.g., IDEs) when reformatting a snippet of code rather
    than a whole file.
  print_diff: (bool) Instead of returning the reformatted source, return a
    diff that turns the formatted source into reformatter source.
"""
import codecs
import difflib
import re
from yapf.pyparser import pyparser
from yapf.pytree import blank_line_calculator
from yapf.pytree import comment_splicer
from yapf.pytree import continuation_splicer
from yapf.pytree import pytree_unwrapper
from yapf.pytree import pytree_utils
from yapf.pytree import split_penalty
from yapf.pytree import subtype_assigner
from yapf.yapflib import errors
from yapf.yapflib import file_resources
from yapf.yapflib import identify_container
from yapf.yapflib import reformatter
from yapf.yapflib import style

def FormatFile(filename, style_config=None, lines=None, print_diff=False, in_place=False, logger=None):
    if False:
        return 10
    'Format a single Python file and return the formatted code.\n\n  Arguments:\n    filename: (unicode) The file to reformat.\n    style_config: (string) Either a style name or a path to a file that contains\n      formatting style settings. If None is specified, use the default style\n      as set in style.DEFAULT_STYLE_FACTORY\n    lines: (list of tuples of integers) A list of tuples of lines, [start, end],\n      that we want to format. The lines are 1-based indexed. It can be used by\n      third-party code (e.g., IDEs) when reformatting a snippet of code rather\n      than a whole file.\n    print_diff: (bool) Instead of returning the reformatted source, return a\n      diff that turns the formatted source into reformatter source.\n    in_place: (bool) If True, write the reformatted code back to the file.\n    logger: (io streamer) A stream to output logging.\n\n  Returns:\n    Tuple of (reformatted_code, encoding, changed). reformatted_code is None if\n    the file is successfully written to (having used in_place). reformatted_code\n    is a diff if print_diff is True.\n\n  Raises:\n    IOError: raised if there was an error reading the file.\n    ValueError: raised if in_place and print_diff are both specified.\n  '
    if in_place and print_diff:
        raise ValueError('Cannot pass both in_place and print_diff.')
    (original_source, newline, encoding) = ReadFile(filename, logger)
    (reformatted_source, changed) = FormatCode(original_source, style_config=style_config, filename=filename, lines=lines, print_diff=print_diff)
    if newline != '\n':
        reformatted_source = reformatted_source.replace('\n', newline)
    if in_place:
        if changed:
            file_resources.WriteReformattedCode(filename, reformatted_source, encoding, in_place)
        return (None, encoding, changed)
    return (reformatted_source, encoding, changed)

def FormatTree(tree, style_config=None, lines=None):
    if False:
        while True:
            i = 10
    'Format a parsed lib2to3 pytree.\n\n  This provides an alternative entry point to YAPF.\n\n  Arguments:\n    tree: (pytree.Node) The root of the pytree to format.\n    style_config: (string) Either a style name or a path to a file that contains\n      formatting style settings. If None is specified, use the default style\n      as set in style.DEFAULT_STYLE_FACTORY\n    lines: (list of tuples of integers) A list of tuples of lines, [start, end],\n      that we want to format. The lines are 1-based indexed. It can be used by\n      third-party code (e.g., IDEs) when reformatting a snippet of code rather\n      than a whole file.\n\n  Returns:\n    The source formatted according to the given formatting style.\n  '
    style.SetGlobalStyle(style.CreateStyleFromConfig(style_config))
    comment_splicer.SpliceComments(tree)
    continuation_splicer.SpliceContinuations(tree)
    subtype_assigner.AssignSubtypes(tree)
    identify_container.IdentifyContainers(tree)
    split_penalty.ComputeSplitPenalties(tree)
    blank_line_calculator.CalculateBlankLines(tree)
    llines = pytree_unwrapper.UnwrapPyTree(tree)
    for lline in llines:
        lline.CalculateFormattingInformation()
    lines = _LineRangesToSet(lines)
    _MarkLinesToFormat(llines, lines)
    return reformatter.Reformat(_SplitSemicolons(llines), lines)

def FormatAST(ast, style_config=None, lines=None):
    if False:
        i = 10
        return i + 15
    'Format a parsed lib2to3 pytree.\n\n  This provides an alternative entry point to YAPF.\n\n  Arguments:\n    unformatted_source: (unicode) The code to format.\n    style_config: (string) Either a style name or a path to a file that contains\n      formatting style settings. If None is specified, use the default style\n      as set in style.DEFAULT_STYLE_FACTORY\n    lines: (list of tuples of integers) A list of tuples of lines, [start, end],\n      that we want to format. The lines are 1-based indexed. It can be used by\n      third-party code (e.g., IDEs) when reformatting a snippet of code rather\n      than a whole file.\n\n  Returns:\n    The source formatted according to the given formatting style.\n  '
    style.SetGlobalStyle(style.CreateStyleFromConfig(style_config))
    llines = pyparser.ParseCode(ast)
    for lline in llines:
        lline.CalculateFormattingInformation()
    lines = _LineRangesToSet(lines)
    _MarkLinesToFormat(llines, lines)
    return reformatter.Reformat(_SplitSemicolons(llines), lines)

def FormatCode(unformatted_source, filename='<unknown>', style_config=None, lines=None, print_diff=False):
    if False:
        while True:
            i = 10
    'Format a string of Python code.\n\n  This provides an alternative entry point to YAPF.\n\n  Arguments:\n    unformatted_source: (unicode) The code to format.\n    filename: (unicode) The name of the file being reformatted.\n    style_config: (string) Either a style name or a path to a file that contains\n      formatting style settings. If None is specified, use the default style\n      as set in style.DEFAULT_STYLE_FACTORY\n    lines: (list of tuples of integers) A list of tuples of lines, [start, end],\n      that we want to format. The lines are 1-based indexed. It can be used by\n      third-party code (e.g., IDEs) when reformatting a snippet of code rather\n      than a whole file.\n    print_diff: (bool) Instead of returning the reformatted source, return a\n      diff that turns the formatted source into reformatter source.\n\n  Returns:\n    Tuple of (reformatted_source, changed). reformatted_source conforms to the\n    desired formatting style. changed is True if the source changed.\n  '
    try:
        tree = pytree_utils.ParseCodeToTree(unformatted_source)
    except Exception as e:
        e.filename = filename
        raise errors.YapfError(errors.FormatErrorMsg(e))
    reformatted_source = FormatTree(tree, style_config=style_config, lines=lines)
    if unformatted_source == reformatted_source:
        return ('' if print_diff else reformatted_source, False)
    if print_diff:
        code_diff = _GetUnifiedDiff(unformatted_source, reformatted_source, filename=filename)
        return (code_diff, code_diff.strip() != '')
    return (reformatted_source, True)

def ReadFile(filename, logger=None):
    if False:
        i = 10
        return i + 15
    'Read the contents of the file.\n\n  An optional logger can be specified to emit messages to your favorite logging\n  stream. If specified, then no exception is raised. This is external so that it\n  can be used by third-party applications.\n\n  Arguments:\n    filename: (unicode) The name of the file.\n    logger: (function) A function or lambda that takes a string and emits it.\n\n  Returns:\n    The contents of filename.\n\n  Raises:\n    IOError: raised if there was an error reading the file.\n  '
    try:
        encoding = file_resources.FileEncoding(filename)
        with codecs.open(filename, mode='r', encoding=encoding) as fd:
            lines = fd.readlines()
        line_ending = file_resources.LineEnding(lines)
        source = '\n'.join((line.rstrip('\r\n') for line in lines)) + '\n'
        return (source, line_ending, encoding)
    except IOError as e:
        if logger:
            logger(e)
        e.args = (e.args[0], (filename, e.args[1][1], e.args[1][2], e.args[1][3]))
        raise
    except UnicodeDecodeError as e:
        if logger:
            logger('Could not parse %s! Consider excluding this file with --exclude.', filename)
            logger(e)
        e.args = (e.args[0], (filename, e.args[1][1], e.args[1][2], e.args[1][3]))
        raise

def _SplitSemicolons(lines):
    if False:
        i = 10
        return i + 15
    res = []
    for line in lines:
        res.extend(line.Split())
    return res
DISABLE_PATTERN = '^#.*\\b(?:yapf:\\s*disable|fmt: ?off)\\b'
ENABLE_PATTERN = '^#.*\\b(?:yapf:\\s*enable|fmt: ?on)\\b'

def _LineRangesToSet(line_ranges):
    if False:
        while True:
            i = 10
    'Return a set of lines in the range.'
    if line_ranges is None:
        return None
    line_set = set()
    for (low, high) in sorted(line_ranges):
        line_set.update(range(low, high + 1))
    return line_set

def _MarkLinesToFormat(llines, lines):
    if False:
        i = 10
        return i + 15
    "Skip sections of code that we shouldn't reformat."
    if lines:
        for uwline in llines:
            uwline.disable = not lines.intersection(range(uwline.lineno, uwline.last.lineno + 1))
    index = 0
    while index < len(llines):
        uwline = llines[index]
        if uwline.is_comment:
            if _DisableYAPF(uwline.first.value.strip()):
                index += 1
                while index < len(llines):
                    uwline = llines[index]
                    line = uwline.first.value.strip()
                    if uwline.is_comment and _EnableYAPF(line):
                        if not _DisableYAPF(line):
                            break
                    uwline.disable = True
                    index += 1
        elif re.search(DISABLE_PATTERN, uwline.last.value.strip(), re.IGNORECASE):
            uwline.disable = True
        index += 1

def _DisableYAPF(line):
    if False:
        i = 10
        return i + 15
    return re.search(DISABLE_PATTERN, line.split('\n')[0].strip(), re.IGNORECASE) or re.search(DISABLE_PATTERN, line.split('\n')[-1].strip(), re.IGNORECASE)

def _EnableYAPF(line):
    if False:
        for i in range(10):
            print('nop')
    return re.search(ENABLE_PATTERN, line.split('\n')[0].strip(), re.IGNORECASE) or re.search(ENABLE_PATTERN, line.split('\n')[-1].strip(), re.IGNORECASE)

def _GetUnifiedDiff(before, after, filename='code'):
    if False:
        for i in range(10):
            print('nop')
    "Get a unified diff of the changes.\n\n  Arguments:\n    before: (unicode) The original source code.\n    after: (unicode) The reformatted source code.\n    filename: (unicode) The code's filename.\n\n  Returns:\n    The unified diff text.\n  "
    before = before.splitlines()
    after = after.splitlines()
    return '\n'.join(difflib.unified_diff(before, after, filename, filename, '(original)', '(reformatted)', lineterm='')) + '\n'