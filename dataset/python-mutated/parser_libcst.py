"""LibCST-based parser."""
import collections
import dataclasses
import logging
import re
import libcst
from pytype import utils
log = logging.getLogger(__name__)
IGNORE_RE = re.compile('^ignore(\\[.+\\])?$')
_DIRECTIVE_RE = re.compile('#\\s*(pytype|type)\\s*:\\s?([^#]*)')

class SkipFileError(Exception):
    """Exception thrown if we encounter "pytype: skip-file" in the source code."""

@dataclasses.dataclass(frozen=True)
class LineRange:
    start_line: int
    end_line: int

    def __contains__(self, line):
        if False:
            while True:
                i = 10
        return self.start_line <= line <= self.end_line

@dataclasses.dataclass(frozen=True)
class Call(LineRange):
    """Tag to identify function calls."""

@dataclasses.dataclass(frozen=True)
class _StructuredComment:
    """A structured comment.

  Attributes:
    line: The line number.
    tool: The tool label, e.g., "type" for "# type: int".
    data: The data, e.g., "int" for "# type: int".
    open_ended: True if the comment appears on a line by itself (i.e., it is
     open-ended rather than attached to a line of code).
  """
    line: int
    tool: str
    data: str
    open_ended: bool

@dataclasses.dataclass(frozen=True)
class _VariableAnnotation(LineRange):
    name: str
    annotation: str

class _BlockReturns:
    """Tracks return statements in with/try blocks."""

    def __init__(self):
        if False:
            print('Hello World!')
        self._block_ranges = []
        self._returns = []
        self._block_returns = {}

    def add_return(self, pos):
        if False:
            print('Hello World!')
        self._returns.append(pos.start.line)

    def all_returns(self):
        if False:
            i = 10
            return i + 15
        return set(self._returns)

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter(self._block_returns.items())

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'\n      Blocks: {self._block_ranges}\n      Returns: {self._returns}\n      {self._block_returns}\n    '

class _Matches:
    """Tracks branches of match statements."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.matches = []

class _ParseVisitor(libcst.CSTVisitor):
    """Visitor for parsing a source tree.

  Attributes:
    structured_comment_groups: Ordered map from a line range to the "type:" and
      "pytype:" comments within the range. Line ranges come in several flavors:
      * Instances of the base LineRange class represent single logical
        statements. These ranges are ascending and non-overlapping and record
        all structured comments found.
      * Instances of the Call subclass represent function calls. These ranges
        are ascending by start_line but may overlap and only record "pytype:"
        comments.
    variable_annotations: Sequence of PEP 526-style variable annotations with
      line numbers.
    decorators: Sequence of lines at which decorated functions are defined.
    defs_start: The line number at which the first class or function definition
      appears, if any.
  """
    METADATA_DEPENDENCIES = (libcst.metadata.PositionProvider, libcst.metadata.ParentNodeProvider)

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.structured_comment_groups = collections.OrderedDict()
        self.variable_annotations = []
        self.param_annotations = []
        self.decorators = collections.defaultdict(list)
        self.defs_start = None
        self.function_ranges = {}
        self.block_returns = _BlockReturns()
        self.matches = _Matches()

    def _get_containing_groups(self, start_line, end_line=None):
        if False:
            print('Hello World!')
        'Get _StructuredComment groups that fully contain the given line range.'
        end_line = end_line or start_line
        for (line_range, group) in reversed(self.structured_comment_groups.items()):
            if line_range.start_line <= start_line and end_line <= line_range.end_line:
                yield (line_range, group)
            elif not isinstance(line_range, Call) and line_range.end_line < start_line:
                return

    def _has_containing_group(self, start_line, end_line=None):
        if False:
            while True:
                i = 10
        for (line_range, _) in self._get_containing_groups(start_line, end_line):
            if not isinstance(line_range, Call):
                return True
        return False

    def _add_structured_comment_group(self, start_line, end_line, cls=LineRange):
        if False:
            while True:
                i = 10
        'Adds an empty _StructuredComment group with the given line range.'
        if cls is LineRange and self._has_containing_group(start_line, end_line):
            return
        keys_to_absorb = []
        keys_to_move = []
        for line_range in reversed(self.structured_comment_groups):
            if cls is LineRange and start_line <= line_range.start_line and (line_range.end_line <= end_line):
                if type(line_range) is LineRange:
                    keys_to_absorb.append(line_range)
                else:
                    keys_to_move.append(line_range)
            elif line_range.start_line > start_line:
                keys_to_move.append(line_range)
            else:
                break
        self.structured_comment_groups[cls(start_line, end_line)] = new_group = []
        for k in reversed(keys_to_absorb):
            new_group.extend(self.structured_comment_groups[k])
            del self.structured_comment_groups[k]
        for k in reversed(keys_to_move):
            self.structured_comment_groups.move_to_end(k)

    def _process_comment(self, line, comment, open_ended):
        if False:
            return 10
        'Process a single comment.'
        matches = list(_DIRECTIVE_RE.finditer(comment))
        if not matches:
            return
        is_nested = matches[0].start(0) > 0
        for m in matches:
            (tool, data) = m.groups()
            assert data is not None
            data = data.strip()
            if tool == 'pytype' and data == 'skip-file':
                raise SkipFileError()
            if tool == 'type' and open_ended and is_nested:
                continue
            structured_comment = _StructuredComment(line, tool, data, open_ended)
            for (line_range, group) in self._get_containing_groups(line):
                if not isinstance(line_range, Call):
                    group.append(structured_comment)
                    break
                elif not open_ended and (tool == 'pytype' or (tool == 'type' and IGNORE_RE.match(data))):
                    group.append(structured_comment)
            else:
                raise AssertionError(f'Could not find a line range for comment {structured_comment} on line {line}')

    def _get_position(self, node):
        if False:
            i = 10
            return i + 15
        return self.get_metadata(libcst.metadata.PositionProvider, node)

    def _visit_comment_owner(self, node, cls=LineRange):
        if False:
            print('Hello World!')
        pos = self._get_position(node)
        self._add_structured_comment_group(pos.start.line, pos.end.line, cls)

    def visit_Decorator(self, node):
        if False:
            return 10
        self._visit_comment_owner(node)

    def visit_SimpleStatementLine(self, node):
        if False:
            while True:
                i = 10
        self._visit_comment_owner(node)

    def visit_SimpleStatementSuite(self, node):
        if False:
            return 10
        self._visit_comment_owner(node)

    def visit_IndentedBlock(self, node):
        if False:
            return 10
        parent = self.get_metadata(libcst.metadata.ParentNodeProvider, node)
        if not isinstance(parent, libcst.FunctionDef):
            start = self._get_position(parent).start
            end = self._get_position(node.header).start
            self._add_structured_comment_group(start.line, end.line)

    def visit_ParenthesizedWhitespace(self, node):
        if False:
            for i in range(10):
                print('nop')
        self._visit_comment_owner(node)

    def visit_Call(self, node):
        if False:
            for i in range(10):
                print('nop')
        self._visit_comment_owner(node, cls=Call)

    def visit_Comparison(self, node):
        if False:
            i = 10
            return i + 15
        self._visit_comment_owner(node, cls=Call)

    def visit_Subscript(self, node):
        if False:
            i = 10
            return i + 15
        self._visit_comment_owner(node, cls=Call)

    def visit_TrailingWhitespace(self, node):
        if False:
            for i in range(10):
                print('nop')
        if node.comment:
            line = self._get_position(node).start.line
            self._process_comment(line, node.comment.value, open_ended=False)

    def visit_EmptyLine(self, node):
        if False:
            print('Hello World!')
        if node.comment:
            line = self._get_position(node).start.line
            self._add_structured_comment_group(line, line)
            self._process_comment(line, node.comment.value, open_ended=True)

    def visit_AnnAssign(self, node):
        if False:
            print('Hello World!')
        if not node.value:
            return
        pos = self._get_position(node)
        annotation = re.sub('\\s*(#.*)?\\n\\s*', '', libcst.Module([node.annotation.annotation]).code)
        if isinstance(node.target, libcst.Name):
            name = node.target.value
        else:
            name = None
        self.variable_annotations.append(_VariableAnnotation(pos.start.line, pos.end.line, name, annotation))

    def visit_Return(self, node):
        if False:
            return 10
        self.block_returns.add_return(self._get_position(node))

    def _visit_decorators(self, node):
        if False:
            print('Hello World!')
        funcdef_pos = self._get_position(node.name).start.line
        for decorator in node.decorators:
            dec = decorator.decorator
            dec_base = dec.func if isinstance(dec, libcst.Call) else dec
            dec_name = libcst.Module([dec_base]).code
            dec_pos = self._get_position(decorator).start.line
            self.decorators[funcdef_pos].append((dec_pos, dec_name))

    def _visit_def(self, node):
        if False:
            for i in range(10):
                print('nop')
        line = self._get_position(node).start.line
        if not self.defs_start or line < self.defs_start:
            self.defs_start = line

    def visit_ClassDef(self, node):
        if False:
            for i in range(10):
                print('nop')
        self._visit_decorators(node)
        self._visit_def(node)

    def visit_FunctionDef(self, node):
        if False:
            print('Hello World!')
        pos = self._get_position(node)
        self._add_structured_comment_group(pos.start.line, self._get_position(node.whitespace_before_colon).end.line)
        self._visit_decorators(node)
        self._visit_def(node)
        self.function_ranges[pos.start.line] = pos.end.line

def parse_src(src, python_version):
    if False:
        return 10
    'Parses a string of source code into a LibCST tree.'
    assert python_version < (3, 9)
    version_str = utils.format_version(python_version)
    config = libcst.PartialParserConfig(python_version=version_str)
    src_tree = libcst.parse_module(src, config)
    return libcst.metadata.MetadataWrapper(src_tree, unsafe_skip_copy=True)

def visit_src_tree(src_tree):
    if False:
        for i in range(10):
            print('nop')
    visitor = _ParseVisitor()
    try:
        src_tree.visit(visitor)
    except RecursionError:
        log.warning('File parsing failed. Comment directives and some variable annotations will be ignored.')
        return None
    return visitor