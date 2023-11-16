"""Comment splicer for lib2to3 trees.

The lib2to3 syntax tree produced by the parser holds comments and whitespace in
prefix attributes of nodes, rather than nodes themselves. This module provides
functionality to splice comments out of prefixes and into nodes of their own,
making them easier to process.

  SpliceComments(): the main function exported by this module.
"""
from yapf_third_party._ylib2to3 import pygram
from yapf_third_party._ylib2to3 import pytree
from yapf_third_party._ylib2to3.pgen2 import token
from yapf.pytree import pytree_utils

def SpliceComments(tree):
    if False:
        i = 10
        return i + 15
    'Given a pytree, splice comments into nodes of their own right.\n\n  Extract comments from the prefixes where they are housed after parsing.\n  The prefixes that previously housed the comments become empty.\n\n  Args:\n    tree: a pytree.Node - the tree to work on. The tree is modified by this\n        function.\n  '
    prev_leaf = [None]
    _AnnotateIndents(tree)

    def _VisitNodeRec(node):
        if False:
            return 10
        'Recursively visit each node to splice comments into the AST.'
        for child in node.children[:]:
            if isinstance(child, pytree.Node):
                _VisitNodeRec(child)
            else:
                if child.prefix.lstrip().startswith('#'):
                    comment_prefix = child.prefix
                    comment_lineno = child.lineno - comment_prefix.count('\n')
                    comment_column = child.column
                    child_prefix = child.prefix.lstrip('\n')
                    prefix_indent = child_prefix[:child_prefix.find('#')]
                    if '\n' in prefix_indent:
                        prefix_indent = prefix_indent[prefix_indent.rfind('\n') + 1:]
                    child.prefix = ''
                    if child.type == token.NEWLINE:
                        comment_column -= len(comment_prefix.lstrip())
                        pytree_utils.InsertNodesAfter(_CreateCommentsFromPrefix(comment_prefix, comment_lineno, comment_column, standalone=False), prev_leaf[0])
                    elif child.type == token.DEDENT:
                        comment_groups = []
                        comment_column = None
                        for cmt in comment_prefix.split('\n'):
                            col = cmt.find('#')
                            if col < 0:
                                if comment_column is None:
                                    comment_lineno += 1
                                    continue
                            elif comment_column is None or col < comment_column:
                                comment_column = col
                                comment_indent = cmt[:comment_column]
                                comment_groups.append((comment_column, comment_indent, []))
                            comment_groups[-1][-1].append(cmt)
                        for (comment_column, comment_indent, comment_group) in comment_groups:
                            ancestor_at_indent = _FindAncestorAtIndent(child, comment_indent)
                            if ancestor_at_indent.type == token.DEDENT:
                                InsertNodes = pytree_utils.InsertNodesBefore
                            else:
                                InsertNodes = pytree_utils.InsertNodesAfter
                            InsertNodes(_CreateCommentsFromPrefix('\n'.join(comment_group) + '\n', comment_lineno, comment_column, standalone=True), ancestor_at_indent)
                            comment_lineno += len(comment_group)
                    else:
                        stmt_parent = _FindStmtParent(child)
                        for leaf_in_parent in stmt_parent.leaves():
                            if leaf_in_parent.type == token.NEWLINE:
                                continue
                            elif id(leaf_in_parent) == id(child):
                                node_with_line_parent = _FindNodeWithStandaloneLineParent(child)
                                if pytree_utils.NodeName(node_with_line_parent.parent) in {'funcdef', 'classdef'}:
                                    comment_end = comment_lineno + comment_prefix.rstrip('\n').count('\n')
                                    if comment_end < node_with_line_parent.lineno - 1:
                                        node_with_line_parent = node_with_line_parent.parent
                                pytree_utils.InsertNodesBefore(_CreateCommentsFromPrefix(comment_prefix, comment_lineno, 0, standalone=True), node_with_line_parent)
                                break
                            else:
                                if comment_lineno == prev_leaf[0].lineno:
                                    comment_lines = comment_prefix.splitlines()
                                    value = comment_lines[0].lstrip()
                                    if value.rstrip('\n'):
                                        comment_column = prev_leaf[0].column
                                        comment_column += len(prev_leaf[0].value)
                                        comment_column += len(comment_lines[0]) - len(comment_lines[0].lstrip())
                                        comment_leaf = pytree.Leaf(type=token.COMMENT, value=value.rstrip('\n'), context=('', (comment_lineno, comment_column)))
                                        pytree_utils.InsertNodesAfter([comment_leaf], prev_leaf[0])
                                        comment_prefix = '\n'.join(comment_lines[1:])
                                        comment_lineno += 1
                                rindex = 0 if '\n' not in comment_prefix.rstrip() else comment_prefix.rstrip().rindex('\n') + 1
                                comment_column = len(comment_prefix[rindex:]) - len(comment_prefix[rindex:].lstrip())
                                comments = _CreateCommentsFromPrefix(comment_prefix, comment_lineno, comment_column, standalone=False)
                                pytree_utils.InsertNodesBefore(comments, child)
                                break
                prev_leaf[0] = child
    _VisitNodeRec(tree)

def _CreateCommentsFromPrefix(comment_prefix, comment_lineno, comment_column, standalone=False):
    if False:
        print('Hello World!')
    "Create pytree nodes to represent the given comment prefix.\n\n  Args:\n    comment_prefix: (unicode) the text of the comment from the node's prefix.\n    comment_lineno: (int) the line number for the start of the comment.\n    comment_column: (int) the column for the start of the comment.\n    standalone: (bool) determines if the comment is standalone or not.\n\n  Returns:\n    The simple_stmt nodes if this is a standalone comment, otherwise a list of\n    new COMMENT leafs. The prefix may consist of multiple comment blocks,\n    separated by blank lines. Each block gets its own leaf.\n  "
    comments = []
    lines = comment_prefix.split('\n')
    index = 0
    while index < len(lines):
        comment_block = []
        while index < len(lines) and lines[index].lstrip().startswith('#'):
            comment_block.append(lines[index].strip())
            index += 1
        if comment_block:
            new_lineno = comment_lineno + index - 1
            comment_block[0] = comment_block[0].strip()
            comment_block[-1] = comment_block[-1].strip()
            comment_leaf = pytree.Leaf(type=token.COMMENT, value='\n'.join(comment_block), context=('', (new_lineno, comment_column)))
            comment_node = comment_leaf if not standalone else pytree.Node(pygram.python_symbols.simple_stmt, [comment_leaf])
            comments.append(comment_node)
        while index < len(lines) and (not lines[index].lstrip()):
            index += 1
    return comments
_STANDALONE_LINE_NODES = frozenset(['suite', 'if_stmt', 'while_stmt', 'for_stmt', 'try_stmt', 'with_stmt', 'funcdef', 'classdef', 'decorated', 'file_input'])

def _FindNodeWithStandaloneLineParent(node):
    if False:
        while True:
            i = 10
    "Find a node whose parent is a 'standalone line' node.\n\n  See the comment above _STANDALONE_LINE_NODES for more details.\n\n  Arguments:\n    node: node to start from\n\n  Returns:\n    Suitable node that's either the node itself or one of its ancestors.\n  "
    if pytree_utils.NodeName(node.parent) in _STANDALONE_LINE_NODES:
        return node
    else:
        return _FindNodeWithStandaloneLineParent(node.parent)
_STATEMENT_NODES = frozenset(['simple_stmt']) | _STANDALONE_LINE_NODES

def _FindStmtParent(node):
    if False:
        print('Hello World!')
    'Find the nearest parent of node that is a statement node.\n\n  Arguments:\n    node: node to start from\n\n  Returns:\n    Nearest parent (or node itself, if suitable).\n  '
    if pytree_utils.NodeName(node) in _STATEMENT_NODES:
        return node
    else:
        return _FindStmtParent(node.parent)

def _FindAncestorAtIndent(node, indent):
    if False:
        print('Hello World!')
    "Find an ancestor of node with the given indentation.\n\n  Arguments:\n    node: node to start from. This must not be the tree root.\n    indent: indentation string for the ancestor we're looking for.\n        See _AnnotateIndents for more details.\n\n  Returns:\n    An ancestor node with suitable indentation. If no suitable ancestor is\n    found, the closest ancestor to the tree root is returned.\n  "
    if node.parent.parent is None:
        return node
    parent_indent = pytree_utils.GetNodeAnnotation(node.parent, pytree_utils.Annotation.CHILD_INDENT)
    if parent_indent is not None and indent.startswith(parent_indent):
        return node
    else:
        return _FindAncestorAtIndent(node.parent, indent)

def _AnnotateIndents(tree):
    if False:
        return 10
    'Annotate the tree with child_indent annotations.\n\n  A child_indent annotation on a node specifies the indentation (as a string,\n  like "  ") of its children. It is inferred from the INDENT child of a node.\n\n  Arguments:\n    tree: root of a pytree. The pytree is modified to add annotations to nodes.\n\n  Raises:\n    RuntimeError: if the tree is malformed.\n  '
    if tree.parent is None:
        pytree_utils.SetNodeAnnotation(tree, pytree_utils.Annotation.CHILD_INDENT, '')
    for child in tree.children:
        if child.type == token.INDENT:
            child_indent = pytree_utils.GetNodeAnnotation(tree, pytree_utils.Annotation.CHILD_INDENT)
            if child_indent is not None and child_indent != child.value:
                raise RuntimeError('inconsistent indentation for child', (tree, child))
            pytree_utils.SetNodeAnnotation(tree, pytree_utils.Annotation.CHILD_INDENT, child.value)
        _AnnotateIndents(child)