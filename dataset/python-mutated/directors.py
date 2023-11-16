"""Code and data structures for managing source directives."""
import bisect
import collections
import logging
import sys
from pytype import config
from pytype.directors import annotations
if sys.version_info[:2] >= (3, 9):
    from pytype.directors import parser
else:
    from pytype.directors import parser_libcst as parser
log = logging.getLogger(__name__)
SkipFileError = parser.SkipFileError
parse_src = parser.parse_src
_ALL_ERRORS = '*'
_ALLOWED_FEATURES = frozenset((x.flag for x in config.FEATURE_FLAGS))
_FUNCTION_CALL_ERRORS = frozenset(('attribute-error', 'duplicate-keyword', 'invalid-annotation', 'missing-parameter', 'not-instantiable', 'wrong-arg-count', 'wrong-arg-types', 'wrong-keyword-args', 'unsupported-operands'))
_ALL_ADJUSTABLE_ERRORS = _FUNCTION_CALL_ERRORS.union(('annotation-type-mismatch', 'bad-return-type', 'bad-yield-annotation', 'container-type-mismatch', 'not-supported-yet', 'signature-mismatch'))

class _DirectiveError(Exception):
    pass

class _LineSet:
    """A set of line numbers.

  The data structure is optimized to represent the union of a sparse set
  of integers and ranges of non-negative integers.  This supports the two styles
  of directives: those after a statement apply only to that line and those on
  their own line apply until countered by the opposing directive.
  """

    def __init__(self):
        if False:
            while True:
                i = 10
        self._lines = {}
        self._transitions = []

    @property
    def lines(self):
        if False:
            i = 10
            return i + 15
        return self._lines

    def set_line(self, line, membership):
        if False:
            print('Hello World!')
        'Set whether a given line is a member of the set.'
        self._lines[line] = membership

    def start_range(self, line, membership):
        if False:
            i = 10
            return i + 15
        'Start a range of lines that are either included/excluded from the set.\n\n    Args:\n      line: A line number.\n      membership: If True, lines >= line are included in the set (starting\n        a range), otherwise they are excluded (ending a range).\n\n    Raises:\n      ValueError: if line is less than that of a previous call to start_range().\n    '
        last = self._transitions[-1] if self._transitions else -1
        if line < last:
            raise ValueError('Line number less than previous start_range() call.')
        previous = len(self._transitions) % 2 == 1
        if membership == previous:
            return
        elif line == last:
            self._transitions.pop()
        else:
            self._transitions.append(line)

    def __contains__(self, line):
        if False:
            for i in range(10):
                print('nop')
        'Return if a line is a member of the set.'
        specific = self._lines.get(line)
        if specific is not None:
            return specific
        pos = bisect.bisect(self._transitions, line)
        return pos % 2 == 1

    def get_disable_after(self, line):
        if False:
            while True:
                i = 10
        'Get an unclosed disable, if any, that starts after line.'
        if len(self._transitions) % 2 == 1 and self._transitions[-1] >= line:
            return self._transitions[-1]
        return None

class _BlockRanges:
    """A collection of possibly nested start..end ranges from AST nodes."""

    def __init__(self, start_to_end_mapping):
        if False:
            while True:
                i = 10
        self._starts = sorted(start_to_end_mapping)
        self._start_to_end = start_to_end_mapping
        self._end_to_start = {v: k for (k, v) in start_to_end_mapping.items()}

    def has_start(self, line):
        if False:
            print('Hello World!')
        return line in self._start_to_end

    def has_end(self, line):
        if False:
            i = 10
            return i + 15
        return line in self._end_to_start

    def find_outermost(self, line):
        if False:
            while True:
                i = 10
        'Find the outermost interval containing line.'
        i = bisect.bisect_left(self._starts, line)
        num_intervals = len(self._starts)
        if i or line == self._starts[0]:
            if i < num_intervals and self._starts[i] == line:
                start = self._starts[i]
            else:
                while 1 < i <= num_intervals and self._start_to_end[self._starts[i - 1]] < line:
                    i -= 1
                start = self._starts[i - 1]
            end = self._start_to_end[start]
            if line in range(start, end):
                return (start, end)
        return (None, None)

    def adjust_end(self, old_end, new_end):
        if False:
            for i in range(10):
                print('nop')
        start = self._end_to_start[old_end]
        self._start_to_end[start] = new_end
        del self._end_to_start[old_end]
        self._end_to_start[new_end] = start

class Director:
    """Holds all of the directive information for a source file."""

    def __init__(self, src_tree, errorlog, filename, disable):
        if False:
            return 10
        'Create a Director for a source file.\n\n    Args:\n      src_tree: The source text as an ast.\n      errorlog: An ErrorLog object.  Directive errors will be logged to the\n        errorlog.\n      filename: The name of the source file.\n      disable: List of error messages to always ignore.\n    '
        self._filename = filename
        self._errorlog = errorlog
        self._variable_annotations = annotations.VariableAnnotations()
        self._param_annotations = None
        self._ignore = _LineSet()
        self._disables = collections.defaultdict(_LineSet)
        self._decorators = collections.defaultdict(list)
        self._decorated_functions = {}
        for error_name in disable:
            self._disables[error_name].start_range(0, True)
        self.return_lines = set()
        self.block_returns = None
        self._function_ranges = _BlockRanges({})
        self._parse_src_tree(src_tree)

    @property
    def type_comments(self):
        if False:
            print('Hello World!')
        return self._variable_annotations.type_comments

    @property
    def annotations(self):
        if False:
            for i in range(10):
                print('nop')
        return self._variable_annotations.annotations

    @property
    def param_annotations(self):
        if False:
            i = 10
            return i + 15
        ret = {}
        for a in self._param_annotations:
            for i in range(a.start_line, a.end_line):
                ret[i] = a.annotations
        return ret

    @property
    def ignore(self):
        if False:
            while True:
                i = 10
        return self._ignore

    @property
    def decorators(self):
        if False:
            i = 10
            return i + 15
        return self._decorators

    @property
    def decorated_functions(self):
        if False:
            i = 10
            return i + 15
        return self._decorated_functions

    def _parse_src_tree(self, src_tree):
        if False:
            print('Hello World!')
        'Parse a source file, extracting directives from comments.'
        visitor = parser.visit_src_tree(src_tree)
        if not visitor:
            return
        self.block_returns = visitor.block_returns
        self.return_lines = visitor.block_returns.all_returns()
        self._function_ranges = _BlockRanges(visitor.function_ranges)
        self._param_annotations = visitor.param_annotations
        self.matches = visitor.matches
        self.features = set()
        for (line_range, group) in visitor.structured_comment_groups.items():
            for comment in group:
                if comment.tool == 'type':
                    self._process_type(comment.line, comment.data, comment.open_ended, line_range)
                else:
                    assert comment.tool == 'pytype'
                    try:
                        self._process_pytype(comment.line, comment.data, comment.open_ended, line_range)
                    except _DirectiveError as e:
                        self._errorlog.invalid_directive(self._filename, comment.line, str(e))
                if not isinstance(line_range, parser.Call) and self._function_ranges.has_end(line_range.end_line):
                    end = line_range.start_line
                    self._function_ranges.adjust_end(line_range.end_line, end)
        for annot in visitor.variable_annotations:
            self._variable_annotations.add_annotation(annot.start_line, annot.name, annot.annotation)
        for (lineno, decorators) in visitor.decorators.items():
            for (decorator_lineno, decorator_name) in decorators:
                self._decorators[lineno].append(decorator_name)
                self._decorated_functions[decorator_lineno] = lineno
        if visitor.defs_start is not None:
            disables = list(self._disables.items())
            disables.append(('Type checking', self._ignore))
            for (name, lineset) in disables:
                lineno = lineset.get_disable_after(visitor.defs_start)
                if lineno is not None:
                    self._errorlog.late_directive(self._filename, lineno, name)

    def _process_type(self, line: int, data: str, open_ended: bool, line_range: parser.LineRange):
        if False:
            i = 10
            return i + 15
        'Process a type: comment.'
        is_ignore = parser.IGNORE_RE.match(data)
        if not is_ignore and line != line_range.end_line:
            self._errorlog.ignored_type_comment(self._filename, line, data)
            return
        final_line = line_range.start_line
        if is_ignore:
            if open_ended:
                self._ignore.start_range(line, True)
            else:
                self._ignore.set_line(line, True)
                self._ignore.set_line(final_line, True)
        else:
            if final_line in self._variable_annotations.type_comments:
                self._errorlog.invalid_directive(self._filename, line, 'Multiple type comments on the same line.')
            self._variable_annotations.add_type_comment(final_line, data)

    def _process_pytype(self, line: int, data: str, open_ended: bool, line_range: parser.LineRange):
        if False:
            print('Hello World!')
        'Process a pytype: comment.'
        if not data:
            raise _DirectiveError('Invalid directive syntax.')
        for option in data.split():
            try:
                (command, values) = option.split('=', 1)
                values = values.split(',')
            except ValueError as e:
                raise _DirectiveError('Invalid directive syntax.') from e
            if command == 'disable':
                disable = True
            elif command == 'enable':
                disable = False
            elif command == 'features':
                features = set(values)
                invalid = features - _ALLOWED_FEATURES
                if invalid:
                    raise _DirectiveError(f"Unknown pytype features: {','.join(invalid)}")
                self.features |= features
                continue
            else:
                raise _DirectiveError(f"Unknown pytype directive: '{command}'")
            if not values:
                raise _DirectiveError('Disable/enable must specify one or more error names.')

            def keep(error_name):
                if False:
                    print('Hello World!')
                if isinstance(line_range, parser.Call):
                    return error_name in _FUNCTION_CALL_ERRORS
                else:
                    return True
            for error_name in values:
                if error_name == _ALL_ERRORS or self._errorlog.is_valid_error_name(error_name):
                    if not keep(error_name):
                        continue
                    lines = self._disables[error_name]
                    if open_ended:
                        lines.start_range(line, disable)
                    else:
                        final_line = self._adjust_line_number_for_pytype_directive(line, error_name, line_range)
                        if final_line != line:
                            lines.set_line(line, disable)
                        lines.set_line(final_line, disable)
                else:
                    self._errorlog.invalid_directive(self._filename, line, f"Invalid error name: '{error_name}'")

    def _adjust_line_number_for_pytype_directive(self, line: int, error_class: str, line_range: parser.LineRange):
        if False:
            i = 10
            return i + 15
        'Adjusts the line number for a pytype directive.'
        if error_class not in _ALL_ADJUSTABLE_ERRORS:
            return line
        return line_range.start_line

    def filter_error(self, error):
        if False:
            print('Hello World!')
        'Return whether the error should be logged.\n\n    This method is suitable for use as an error filter.\n\n    Args:\n      error: An error._Error object.\n\n    Returns:\n      True iff the error should be included in the log.\n    '
        if error.filename != self._filename or error.lineno is None:
            return True
        if error.name == 'bad-return-type' and error.opcode_name == 'RETURN_VALUE' and (error.lineno not in self.return_lines):
            (_, end) = self._function_ranges.find_outermost(error.lineno)
            if end:
                error.set_lineno(end)
        line = error.lineno or sys.maxsize
        return line not in self._ignore and line not in self._disables[_ALL_ERRORS] and (line not in self._disables[error.name])