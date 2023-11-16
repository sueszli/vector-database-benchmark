"""Calculate the number of blank lines between top-level entities.

Calculates how many blank lines we need between classes, functions, and other
entities at the same level.

  CalculateBlankLines(): the main function exported by this module.

Annotations:
  newlines: The number of newlines required before the node.
"""
from yapf_third_party._ylib2to3.pgen2 import token as grammar_token
from yapf.pytree import pytree_utils
from yapf.pytree import pytree_visitor
from yapf.yapflib import style
_NO_BLANK_LINES = 1
_ONE_BLANK_LINE = 2
_TWO_BLANK_LINES = 3
_PYTHON_STATEMENTS = frozenset({'small_stmt', 'expr_stmt', 'print_stmt', 'del_stmt', 'pass_stmt', 'break_stmt', 'continue_stmt', 'return_stmt', 'raise_stmt', 'yield_stmt', 'import_stmt', 'global_stmt', 'exec_stmt', 'assert_stmt', 'if_stmt', 'while_stmt', 'for_stmt', 'try_stmt', 'with_stmt', 'nonlocal_stmt', 'async_stmt', 'simple_stmt'})

def CalculateBlankLines(tree):
    if False:
        print('Hello World!')
    'Run the blank line calculator visitor over the tree.\n\n  This modifies the tree in place.\n\n  Arguments:\n    tree: the top-level pytree node to annotate with subtypes.\n  '
    blank_line_calculator = _BlankLineCalculator()
    blank_line_calculator.Visit(tree)

class _BlankLineCalculator(pytree_visitor.PyTreeVisitor):
    """_BlankLineCalculator - see file-level docstring for a description."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.class_level = 0
        self.function_level = 0
        self.last_comment_lineno = 0
        self.last_was_decorator = False
        self.last_was_class_or_function = False

    def Visit_simple_stmt(self, node):
        if False:
            i = 10
            return i + 15
        self.DefaultNodeVisit(node)
        if node.children[0].type == grammar_token.COMMENT:
            self.last_comment_lineno = node.children[0].lineno

    def Visit_decorator(self, node):
        if False:
            while True:
                i = 10
        if self.last_comment_lineno and self.last_comment_lineno == node.children[0].lineno - 1:
            _SetNumNewlines(node.children[0], _NO_BLANK_LINES)
        else:
            _SetNumNewlines(node.children[0], self._GetNumNewlines(node))
        for child in node.children:
            self.Visit(child)
        self.last_was_decorator = True

    def Visit_classdef(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.last_was_class_or_function = False
        index = self._SetBlankLinesBetweenCommentAndClassFunc(node)
        self.last_was_decorator = False
        self.class_level += 1
        for child in node.children[index:]:
            self.Visit(child)
        self.class_level -= 1
        self.last_was_class_or_function = True

    def Visit_funcdef(self, node):
        if False:
            return 10
        self.last_was_class_or_function = False
        index = self._SetBlankLinesBetweenCommentAndClassFunc(node)
        if _AsyncFunction(node):
            index = self._SetBlankLinesBetweenCommentAndClassFunc(node.prev_sibling.parent)
            _SetNumNewlines(node.children[0], None)
        else:
            index = self._SetBlankLinesBetweenCommentAndClassFunc(node)
        self.last_was_decorator = False
        self.function_level += 1
        for child in node.children[index:]:
            self.Visit(child)
        self.function_level -= 1
        self.last_was_class_or_function = True

    def DefaultNodeVisit(self, node):
        if False:
            print('Hello World!')
        'Override the default visitor for Node.\n\n    This will set the blank lines required if the last entity was a class or\n    function.\n\n    Arguments:\n      node: (pytree.Node) The node to visit.\n    '
        if self.last_was_class_or_function:
            if pytree_utils.NodeName(node) in _PYTHON_STATEMENTS:
                leaf = pytree_utils.FirstLeafNode(node)
                _SetNumNewlines(leaf, self._GetNumNewlines(leaf))
        self.last_was_class_or_function = False
        super(_BlankLineCalculator, self).DefaultNodeVisit(node)

    def _SetBlankLinesBetweenCommentAndClassFunc(self, node):
        if False:
            while True:
                i = 10
        'Set the number of blanks between a comment and class or func definition.\n\n    Class and function definitions have leading comments as children of the\n    classdef and functdef nodes.\n\n    Arguments:\n      node: (pytree.Node) The classdef or funcdef node.\n\n    Returns:\n      The index of the first child past the comment nodes.\n    '
        index = 0
        while pytree_utils.IsCommentStatement(node.children[index]):
            self.Visit(node.children[index].children[0])
            if not self.last_was_decorator:
                _SetNumNewlines(node.children[index].children[0], _ONE_BLANK_LINE)
            index += 1
        if index and node.children[index].lineno - 1 == node.children[index - 1].children[0].lineno:
            _SetNumNewlines(node.children[index], _NO_BLANK_LINES)
        else:
            if self.last_comment_lineno + 1 == node.children[index].lineno:
                num_newlines = _NO_BLANK_LINES
            else:
                num_newlines = self._GetNumNewlines(node)
            _SetNumNewlines(node.children[index], num_newlines)
        return index

    def _GetNumNewlines(self, node):
        if False:
            return 10
        if self.last_was_decorator:
            return _NO_BLANK_LINES
        elif self._IsTopLevel(node):
            return 1 + style.Get('BLANK_LINES_AROUND_TOP_LEVEL_DEFINITION')
        return _ONE_BLANK_LINE

    def _IsTopLevel(self, node):
        if False:
            i = 10
            return i + 15
        return not (self.class_level or self.function_level) and _StartsInZerothColumn(node)

def _SetNumNewlines(node, num_newlines):
    if False:
        for i in range(10):
            print('nop')
    pytree_utils.SetNodeAnnotation(node, pytree_utils.Annotation.NEWLINES, num_newlines)

def _StartsInZerothColumn(node):
    if False:
        while True:
            i = 10
    return pytree_utils.FirstLeafNode(node).column == 0 or (_AsyncFunction(node) and node.prev_sibling.column == 0)

def _AsyncFunction(node):
    if False:
        while True:
            i = 10
    return node.prev_sibling and node.prev_sibling.type == grammar_token.ASYNC