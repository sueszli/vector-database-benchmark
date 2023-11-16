"""pytree-related utilities.

This module collects various utilities related to the parse trees produced by
the lib2to3 library.

  NodeName(): produces a string name for pytree nodes.
  ParseCodeToTree(): convenience wrapper around lib2to3 interfaces to parse
                     a given string with code to a pytree.
  InsertNodeBefore(): insert a node before another in a pytree.
  InsertNodeAfter(): insert a node after another in a pytree.
  {Get,Set}NodeAnnotation(): manage custom annotations on pytree nodes.
"""
import ast
import os
from yapf_third_party._ylib2to3 import pygram
from yapf_third_party._ylib2to3 import pytree
from yapf_third_party._ylib2to3.pgen2 import driver
from yapf_third_party._ylib2to3.pgen2 import parse
from yapf_third_party._ylib2to3.pgen2 import token
NONSEMANTIC_TOKENS = frozenset(['DEDENT', 'INDENT', 'NEWLINE', 'ENDMARKER'])

class Annotation(object):
    """Annotation names associated with pytrees."""
    CHILD_INDENT = 'child_indent'
    NEWLINES = 'newlines'
    MUST_SPLIT = 'must_split'
    SPLIT_PENALTY = 'split_penalty'
    SUBTYPE = 'subtype'

def NodeName(node):
    if False:
        for i in range(10):
            print('nop')
    'Produce a string name for a given node.\n\n  For a Leaf this is the token name, and for a Node this is the type.\n\n  Arguments:\n    node: a tree node\n\n  Returns:\n    Name as a string.\n  '
    if node.type < 256:
        return token.tok_name[node.type]
    else:
        return pygram.python_grammar.number2symbol[node.type]

def FirstLeafNode(node):
    if False:
        i = 10
        return i + 15
    if isinstance(node, pytree.Leaf):
        return node
    return FirstLeafNode(node.children[0])

def LastLeafNode(node):
    if False:
        i = 10
        return i + 15
    if isinstance(node, pytree.Leaf):
        return node
    return LastLeafNode(node.children[-1])
_PYTHON_GRAMMAR = pygram.python_grammar_no_print_statement.copy()
del _PYTHON_GRAMMAR.keywords['exec']

def ParseCodeToTree(code):
    if False:
        for i in range(10):
            print('nop')
    'Parse the given code to a lib2to3 pytree.\n\n  Arguments:\n    code: a string with the code to parse.\n\n  Raises:\n    SyntaxError if the code is invalid syntax.\n    parse.ParseError if some other parsing failure.\n\n  Returns:\n    The root node of the parsed tree.\n  '
    if not code.endswith(os.linesep):
        code += os.linesep
    try:
        parser_driver = driver.Driver(_PYTHON_GRAMMAR, convert=pytree.convert)
        tree = parser_driver.parse_string(code, debug=False)
    except parse.ParseError:
        ast.parse(code)
        raise
    return _WrapEndMarker(tree)

def _WrapEndMarker(tree):
    if False:
        while True:
            i = 10
    'Wrap a single ENDMARKER token in a "file_input" node.\n\n  Arguments:\n    tree: (pytree.Node) The root node of the parsed tree.\n\n  Returns:\n    The root node of the parsed tree. If the tree is a single ENDMARKER node,\n    then that node is wrapped in a "file_input" node. That will ensure we don\'t\n    skip comments attached to that node.\n  '
    if isinstance(tree, pytree.Leaf) and tree.type == token.ENDMARKER:
        return pytree.Node(pygram.python_symbols.file_input, [tree])
    return tree

def InsertNodesBefore(new_nodes, target):
    if False:
        i = 10
        return i + 15
    'Insert new_nodes before the given target location in the tree.\n\n  Arguments:\n    new_nodes: a sequence of new nodes to insert (the nodes should not be in the\n      tree).\n    target: the target node before which the new node node will be inserted.\n\n  Raises:\n    RuntimeError: if the tree is corrupted, or the insertion would corrupt it.\n  '
    for node in new_nodes:
        _InsertNodeAt(node, target, after=False)

def InsertNodesAfter(new_nodes, target):
    if False:
        print('Hello World!')
    'Insert new_nodes after the given target location in the tree.\n\n  Arguments:\n    new_nodes: a sequence of new nodes to insert (the nodes should not be in the\n      tree).\n    target: the target node after which the new node node will be inserted.\n\n  Raises:\n    RuntimeError: if the tree is corrupted, or the insertion would corrupt it.\n  '
    for node in reversed(new_nodes):
        _InsertNodeAt(node, target, after=True)

def _InsertNodeAt(new_node, target, after=False):
    if False:
        return 10
    "Underlying implementation for node insertion.\n\n  Arguments:\n    new_node: a new node to insert (this node should not be in the tree).\n    target: the target node.\n    after: if True, new_node is inserted after target. Otherwise, it's inserted\n      before target.\n\n  Returns:\n    nothing\n\n  Raises:\n    RuntimeError: if the tree is corrupted, or the insertion would corrupt it.\n  "
    if new_node.parent is not None:
        raise RuntimeError('inserting node which already has a parent', (new_node, new_node.parent))
    parent_of_target = target.parent
    if parent_of_target is None:
        raise RuntimeError('expected target node to have a parent', (target,))
    for (i, child) in enumerate(parent_of_target.children):
        if child is target:
            insertion_index = i + 1 if after else i
            parent_of_target.insert_child(insertion_index, new_node)
            return
    raise RuntimeError('unable to find insertion point for target node', (target,))
_NODE_ANNOTATION_PREFIX = '_yapf_annotation_'

def CopyYapfAnnotations(src, dst):
    if False:
        i = 10
        return i + 15
    'Copy all YAPF annotations from the source node to the destination node.\n\n  Arguments:\n    src: the source node.\n    dst: the destination node.\n  '
    for annotation in dir(src):
        if annotation.startswith(_NODE_ANNOTATION_PREFIX):
            setattr(dst, annotation, getattr(src, annotation, None))

def GetNodeAnnotation(node, annotation, default=None):
    if False:
        while True:
            i = 10
    "Get annotation value from a node.\n\n  Arguments:\n    node: the node.\n    annotation: annotation name - a string.\n    default: the default value to return if there's no annotation.\n\n  Returns:\n    Value of the annotation in the given node. If the node doesn't have this\n    particular annotation name yet, returns default.\n  "
    return getattr(node, _NODE_ANNOTATION_PREFIX + annotation, default)

def SetNodeAnnotation(node, annotation, value):
    if False:
        i = 10
        return i + 15
    'Set annotation value on a node.\n\n  Arguments:\n    node: the node.\n    annotation: annotation name - a string.\n    value: annotation value to set.\n  '
    setattr(node, _NODE_ANNOTATION_PREFIX + annotation, value)

def AppendNodeAnnotation(node, annotation, value):
    if False:
        for i in range(10):
            print('nop')
    'Appends an annotation value to a list of annotations on the node.\n\n  Arguments:\n    node: the node.\n    annotation: annotation name - a string.\n    value: annotation value to set.\n  '
    attr = GetNodeAnnotation(node, annotation, set())
    attr.add(value)
    SetNodeAnnotation(node, annotation, attr)

def RemoveSubtypeAnnotation(node, value):
    if False:
        i = 10
        return i + 15
    'Removes an annotation value from the subtype annotations on the node.\n\n  Arguments:\n    node: the node.\n    value: annotation value to remove.\n  '
    attr = GetNodeAnnotation(node, Annotation.SUBTYPE)
    if attr and value in attr:
        attr.remove(value)
        SetNodeAnnotation(node, Annotation.SUBTYPE, attr)

def GetOpeningBracket(node):
    if False:
        while True:
            i = 10
    "Get opening bracket value from a node.\n\n  Arguments:\n    node: the node.\n\n  Returns:\n    The opening bracket node or None if it couldn't find one.\n  "
    return getattr(node, _NODE_ANNOTATION_PREFIX + 'container_bracket', None)

def SetOpeningBracket(node, bracket):
    if False:
        return 10
    'Set opening bracket value for a node.\n\n  Arguments:\n    node: the node.\n    bracket: opening bracket to set.\n  '
    setattr(node, _NODE_ANNOTATION_PREFIX + 'container_bracket', bracket)

def DumpNodeToString(node):
    if False:
        while True:
            i = 10
    'Dump a string representation of the given node. For debugging.\n\n  Arguments:\n    node: the node.\n\n  Returns:\n    The string representation.\n  '
    if isinstance(node, pytree.Leaf):
        fmt = '{name}({value}) [lineno={lineno}, column={column}, prefix={prefix}, penalty={penalty}]'
        return fmt.format(name=NodeName(node), value=_PytreeNodeRepr(node), lineno=node.lineno, column=node.column, prefix=repr(node.prefix), penalty=GetNodeAnnotation(node, Annotation.SPLIT_PENALTY, None))
    else:
        fmt = '{node} [{len} children] [child_indent="{indent}"]'
        return fmt.format(node=NodeName(node), len=len(node.children), indent=GetNodeAnnotation(node, Annotation.CHILD_INDENT))

def _PytreeNodeRepr(node):
    if False:
        return 10
    'Like pytree.Node.__repr__, but names instead of numbers for tokens.'
    if isinstance(node, pytree.Node):
        return '%s(%s, %r)' % (node.__class__.__name__, NodeName(node), [_PytreeNodeRepr(c) for c in node.children])
    if isinstance(node, pytree.Leaf):
        return '%s(%s, %r)' % (node.__class__.__name__, NodeName(node), node.value)

def IsCommentStatement(node):
    if False:
        while True:
            i = 10
    return NodeName(node) == 'simple_stmt' and node.children[0].type == token.COMMENT