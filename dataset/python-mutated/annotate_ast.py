"""Library to take a Python AST and add Pytype type information to it."""
from pytype import io
from pytype.pytd import pytd_utils
from pytype.tools.traces import traces

def annotate_source(source, ast_module, pytype_options):
    if False:
        for i in range(10):
            print('nop')
    'Infer types for `source`, and return an AST of it with types added.\n\n  Args:\n    source: Text, the source code to type-infer and parse to an AST.\n    ast_module: An ast-module like object used to parse the source to an AST\n      and traverse the created ast.Module object.\n    pytype_options: pytype.config.Options, the options to pass onto Pytype.\n\n  Returns:\n    The created Module object from what `ast_factory` returned.\n  '
    source_code = infer_types(source, pytype_options)
    module = ast_module.parse(source, pytype_options.input)
    visitor = AnnotateAstVisitor(source_code, ast_module)
    visitor.visit(module)
    return module

def infer_types(source, options):
    if False:
        i = 10
        return i + 15
    'Infer types for the provided source.\n\n  Args:\n    source: Text, the source code to analyze.\n    options: pytype.config.Options, the options to pass onto Pytype.\n\n  Returns:\n    source.Code object with information gathered by Pytype.\n  '
    with io.wrap_pytype_exceptions(PytypeError, filename=options.input):
        return traces.trace(source, options)

class AnnotateAstVisitor(traces.MatchAstVisitor):
    """Traverses an AST and sets type information on its nodes.

  This is modeled after ast.NodeVisitor, but doesn't inherit from it because
  it is ast-module agnostic so that different AST implementations can be used.
  """

    def visit_Name(self, node):
        if False:
            while True:
                i = 10
        self._maybe_annotate(node)

    def visit_Attribute(self, node):
        if False:
            while True:
                i = 10
        self._maybe_annotate(node)

    def visit_FunctionDef(self, node):
        if False:
            print('Hello World!')
        self._maybe_annotate(node)

    def _maybe_annotate(self, node):
        if False:
            i = 10
            return i + 15
        'Annotates a node.'
        try:
            ops = self.match(node)
        except NotImplementedError:
            return
        (unused_loc, entry) = next(iter(ops), (None, None))
        self._maybe_set_type(node, entry)

    def _maybe_set_type(self, node, trace):
        if False:
            while True:
                i = 10
        'Sets type information on the node, if there is any to set.'
        if not trace:
            return
        node.resolved_type = trace.types[-1]
        node.resolved_annotation = _annotation_str_from_type_def(trace.types[-1])

class PytypeError(Exception):
    """Wrap exceptions raised by Pytype."""

def _annotation_str_from_type_def(type_def):
    if False:
        for i in range(10):
            print('nop')
    return pytd_utils.Print(type_def)