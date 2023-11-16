"""Container for origin source code information before AutoGraph compilation."""
import collections
import difflib
import inspect
import os
import tokenize
import gast
import six
from nvidia.dali._autograph.pyct import anno
from nvidia.dali._autograph.pyct import ast_util
from nvidia.dali._autograph.pyct import parser
from nvidia.dali._autograph.pyct import pretty_printer

class LineLocation(collections.namedtuple('LineLocation', ('filename', 'lineno'))):
    """Similar to Location, but without column information.

  Attributes:
    filename: Text
    lineno: int, 1-based
  """
    pass

class Location(collections.namedtuple('Location', ('filename', 'lineno', 'col_offset'))):
    """Encodes code location information.

  Attributes:
    filename: Text
    lineno: int, 1-based
    col_offset: int
    line_loc: LineLocation
  """

    @property
    def line_loc(self):
        if False:
            print('Hello World!')
        return LineLocation(self.filename, self.lineno)

class OriginInfo(collections.namedtuple('OriginInfo', ('loc', 'function_name', 'source_code_line', 'comment'))):
    """Container for information about the source code before conversion.

  Attributes:
    loc: Location
    function_name: Optional[Text]
    source_code_line: Text
    comment: Optional[Text]
  """

    def as_frame(self):
        if False:
            while True:
                i = 10
        'Returns a 4-tuple consistent with the return of traceback.extract_tb.'
        return (self.loc.filename, self.loc.lineno, self.function_name, self.source_code_line)

    def __repr__(self):
        if False:
            print('Hello World!')
        if self.loc.filename:
            return '{}:{}:{}'.format(os.path.split(self.loc.filename)[1], self.loc.lineno, self.loc.col_offset)
        return '<no file>:{}:{}'.format(self.loc.lineno, self.loc.col_offset)

def create_source_map(nodes, code, filepath):
    if False:
        while True:
            i = 10
    'Creates a source map between an annotated AST and the code it compiles to.\n\n  Note: this function assumes nodes nodes, code and filepath correspond to the\n  same code.\n\n  Args:\n    nodes: Iterable[ast.AST, ...], one or more AST modes.\n    code: Text, the source code in which nodes are found.\n    filepath: Text\n\n  Returns:\n    Dict[LineLocation, OriginInfo], mapping locations in code to locations\n    indicated by origin annotations in node.\n  '
    reparsed_nodes = parser.parse(code, preamble_len=0, single_node=False)
    for node in reparsed_nodes:
        resolve(node, code, filepath, node.lineno, node.col_offset)
    source_map = {}
    try:
        for (before, after) in ast_util.parallel_walk(nodes, reparsed_nodes):
            origin_info = anno.getanno(before, anno.Basic.ORIGIN, default=None)
            final_info = anno.getanno(after, anno.Basic.ORIGIN, default=None)
            if origin_info is None or final_info is None:
                continue
            line_loc = LineLocation(final_info.loc.filename, final_info.loc.lineno)
            existing_origin = source_map.get(line_loc)
            if existing_origin is not None:
                if existing_origin.loc.line_loc == origin_info.loc.line_loc:
                    if existing_origin.loc.lineno >= origin_info.loc.lineno:
                        continue
                if existing_origin.loc.col_offset <= origin_info.loc.col_offset:
                    continue
            source_map[line_loc] = origin_info
    except ValueError as err:
        new_msg = 'Inconsistent ASTs detected. This is a bug. Cause: \n'
        new_msg += str(err)
        new_msg += 'Diff:\n'
        for (n, rn) in zip(nodes, reparsed_nodes):
            nodes_str = pretty_printer.fmt(n, color=False, noanno=True)
            reparsed_nodes_str = pretty_printer.fmt(rn, color=False, noanno=True)
            diff = difflib.context_diff(nodes_str.split('\n'), reparsed_nodes_str.split('\n'), fromfile='Original nodes', tofile='Reparsed nodes', n=7)
            diff = '\n'.join(diff)
            new_msg += diff + '\n'
        raise ValueError(new_msg)
    return source_map

class _Function(object):

    def __init__(self, name):
        if False:
            print('Hello World!')
        self.name = name

class OriginResolver(gast.NodeVisitor):
    """Annotates an AST with additional source information like file name."""

    def __init__(self, root_node, source_lines, comments_map, context_lineno, context_col_offset, filepath):
        if False:
            for i in range(10):
                print('nop')
        self._source_lines = source_lines
        self._comments_map = comments_map
        if hasattr(root_node, 'decorator_list') and root_node.decorator_list and hasattr(root_node.decorator_list[0], 'lineno'):
            self._lineno_offset = context_lineno - root_node.decorator_list[0].lineno
        else:
            self._lineno_offset = context_lineno - root_node.lineno
        self._col_offset = context_col_offset - root_node.col_offset
        self._filepath = filepath
        self._function_stack = []

    def _absolute_lineno(self, lineno):
        if False:
            i = 10
            return i + 15
        return lineno + self._lineno_offset

    def _absolute_col_offset(self, col_offset):
        if False:
            print('Hello World!')
        if col_offset is None:
            return 0
        return col_offset + self._col_offset

    def _attach_origin_info(self, node):
        if False:
            print('Hello World!')
        lineno = getattr(node, 'lineno', None)
        col_offset = getattr(node, 'col_offset', None)
        if lineno is None:
            return
        if self._function_stack:
            function_name = self._function_stack[-1].name
        else:
            function_name = None
        source_code_line = self._source_lines[lineno - 1]
        comment = self._comments_map.get(lineno)
        loc = Location(self._filepath, self._absolute_lineno(lineno), self._absolute_col_offset(col_offset))
        origin = OriginInfo(loc, function_name, source_code_line, comment)
        anno.setanno(node, 'lineno', lineno)
        anno.setanno(node, anno.Basic.ORIGIN, origin)

    def visit(self, node):
        if False:
            print('Hello World!')
        entered_function = False
        if isinstance(node, gast.FunctionDef):
            entered_function = True
            self._function_stack.append(_Function(node.name))
        self._attach_origin_info(node)
        self.generic_visit(node)
        if entered_function:
            self._function_stack.pop()

def resolve(node, source, context_filepath, context_lineno, context_col_offset):
    if False:
        while True:
            i = 10
    'Adds origin information to an AST, based on the source it was loaded from.\n\n  This allows us to map the original source code line numbers to generated\n  source code.\n\n  Note: the AST may be a part of a larger context (e.g. a function is part of\n  a module that may contain other things). However, this function does not\n  assume the source argument contains the entire context, nor that it contains\n  only code corresponding to node itself. However, it assumes that node was\n  parsed from the given source code.\n  For this reason, two extra arguments are required, and they indicate the\n  location of the node in the original context.\n\n  Args:\n    node: gast.AST, the AST to annotate.\n    source: Text, the source code representing node.\n    context_filepath: Text\n    context_lineno: int\n    context_col_offset: int\n  '
    code_reader = six.StringIO(source)
    comments_map = {}
    try:
        for token in tokenize.generate_tokens(code_reader.readline):
            (tok_type, tok_string, loc, _, _) = token
            (srow, _) = loc
            if tok_type == tokenize.COMMENT:
                comments_map[srow] = tok_string.strip()[1:].strip()
    except tokenize.TokenError:
        if isinstance(node, gast.Lambda):
            pass
        else:
            raise
    source_lines = source.split('\n')
    visitor = OriginResolver(node, source_lines, comments_map, context_lineno, context_col_offset, context_filepath)
    visitor.visit(node)

def resolve_entity(node, source, entity):
    if False:
        while True:
            i = 10
    'Like resolve, but extracts the context information from an entity.'
    (lines, lineno) = inspect.getsourcelines(entity)
    filepath = inspect.getsourcefile(entity)
    definition_line = lines[0]
    col_offset = len(definition_line) - len(definition_line.lstrip())
    resolve(node, source, filepath, lineno, col_offset)

def copy_origin(from_node, to_node):
    if False:
        print('Hello World!')
    'Copies the origin info from a node to another, recursively.'
    origin = anno.Basic.ORIGIN.of(from_node, default=None)
    if origin is None:
        return
    if not isinstance(to_node, (list, tuple)):
        to_node = (to_node,)
    for node in to_node:
        for n in gast.walk(node):
            anno.setanno(n, anno.Basic.ORIGIN, origin)