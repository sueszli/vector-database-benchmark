import re
import textwrap
from ast import literal_eval
from inspect import cleandoc
from weakref import WeakKeyDictionary
from parso.python import tree
from parso.cache import parser_cache
from parso import split_lines
_EXECUTE_NODES = {'funcdef', 'classdef', 'import_from', 'import_name', 'test', 'or_test', 'and_test', 'not_test', 'comparison', 'expr', 'xor_expr', 'and_expr', 'shift_expr', 'arith_expr', 'atom_expr', 'term', 'factor', 'power', 'atom'}
_FLOW_KEYWORDS = ('try', 'except', 'finally', 'else', 'if', 'elif', 'with', 'for', 'while')

def get_executable_nodes(node, last_added=False):
    if False:
        i = 10
        return i + 15
    '\n    For static analysis.\n    '
    result = []
    typ = node.type
    if typ == 'name':
        next_leaf = node.get_next_leaf()
        if last_added is False and node.parent.type != 'param' and (next_leaf != '='):
            result.append(node)
    elif typ == 'expr_stmt':
        result.append(node)
        for child in node.children:
            result += get_executable_nodes(child, last_added=True)
    elif typ == 'decorator':
        if node.children[-2] == ')':
            node = node.children[-3]
            if node != '(':
                result += get_executable_nodes(node)
    else:
        try:
            children = node.children
        except AttributeError:
            pass
        else:
            if node.type in _EXECUTE_NODES and (not last_added):
                result.append(node)
            for child in children:
                result += get_executable_nodes(child, last_added)
    return result

def get_sync_comp_fors(comp_for):
    if False:
        return 10
    yield comp_for
    last = comp_for.children[-1]
    while True:
        if last.type == 'comp_for':
            yield last.children[1]
        elif last.type == 'sync_comp_for':
            yield last
        elif not last.type == 'comp_if':
            break
        last = last.children[-1]

def for_stmt_defines_one_name(for_stmt):
    if False:
        return 10
    '\n    Returns True if only one name is returned: ``for x in y``.\n    Returns False if the for loop is more complicated: ``for x, z in y``.\n\n    :returns: bool\n    '
    return for_stmt.children[1].type == 'name'

def get_flow_branch_keyword(flow_node, node):
    if False:
        for i in range(10):
            print('nop')
    start_pos = node.start_pos
    if not flow_node.start_pos < start_pos <= flow_node.end_pos:
        raise ValueError('The node is not part of the flow.')
    keyword = None
    for (i, child) in enumerate(flow_node.children):
        if start_pos < child.start_pos:
            return keyword
        first_leaf = child.get_first_leaf()
        if first_leaf in _FLOW_KEYWORDS:
            keyword = first_leaf
    return None

def clean_scope_docstring(scope_node):
    if False:
        i = 10
        return i + 15
    ' Returns a cleaned version of the docstring token. '
    node = scope_node.get_doc_node()
    if node is not None:
        return cleandoc(safe_literal_eval(node.value))
    return ''

def find_statement_documentation(tree_node):
    if False:
        for i in range(10):
            print('nop')
    if tree_node.type == 'expr_stmt':
        tree_node = tree_node.parent
        maybe_string = tree_node.get_next_sibling()
        if maybe_string is not None:
            if maybe_string.type == 'simple_stmt':
                maybe_string = maybe_string.children[0]
                if maybe_string.type == 'string':
                    return cleandoc(safe_literal_eval(maybe_string.value))
    return ''

def safe_literal_eval(value):
    if False:
        for i in range(10):
            print('nop')
    first_two = value[:2].lower()
    if first_two[0] == 'f' or first_two in ('fr', 'rf'):
        return ''
    return literal_eval(value)

def get_signature(funcdef, width=72, call_string=None, omit_first_param=False, omit_return_annotation=False):
    if False:
        print('Hello World!')
    '\n    Generate a string signature of a function.\n\n    :param width: Fold lines if a line is longer than this value.\n    :type width: int\n    :arg func_name: Override function name when given.\n    :type func_name: str\n\n    :rtype: str\n    '
    if call_string is None:
        if funcdef.type == 'lambdef':
            call_string = '<lambda>'
        else:
            call_string = funcdef.name.value
    params = funcdef.get_params()
    if omit_first_param:
        params = params[1:]
    p = '(' + ''.join((param.get_code() for param in params)).strip() + ')'
    p = re.sub('\\s+', ' ', p)
    if funcdef.annotation and (not omit_return_annotation):
        rtype = ' ->' + funcdef.annotation.get_code()
    else:
        rtype = ''
    code = call_string + p + rtype
    return '\n'.join(textwrap.wrap(code, width))

def move(node, line_offset):
    if False:
        i = 10
        return i + 15
    '\n    Move the `Node` start_pos.\n    '
    try:
        children = node.children
    except AttributeError:
        node.line += line_offset
    else:
        for c in children:
            move(c, line_offset)

def get_following_comment_same_line(node):
    if False:
        while True:
            i = 10
    '\n    returns (as string) any comment that appears on the same line,\n    after the node, including the #\n    '
    try:
        if node.type == 'for_stmt':
            whitespace = node.children[5].get_first_leaf().prefix
        elif node.type == 'with_stmt':
            whitespace = node.children[3].get_first_leaf().prefix
        elif node.type == 'funcdef':
            whitespace = node.children[4].get_first_leaf().get_next_leaf().prefix
        else:
            whitespace = node.get_last_leaf().get_next_leaf().prefix
    except AttributeError:
        return None
    except ValueError:
        return None
    if '#' not in whitespace:
        return None
    comment = whitespace[whitespace.index('#'):]
    if '\r' in comment:
        comment = comment[:comment.index('\r')]
    if '\n' in comment:
        comment = comment[:comment.index('\n')]
    return comment

def is_scope(node):
    if False:
        i = 10
        return i + 15
    t = node.type
    if t == 'comp_for':
        return node.children[1].type != 'sync_comp_for'
    return t in ('file_input', 'classdef', 'funcdef', 'lambdef', 'sync_comp_for')

def _get_parent_scope_cache(func):
    if False:
        i = 10
        return i + 15
    cache = WeakKeyDictionary()

    def wrapper(parso_cache_node, node, include_flows=False):
        if False:
            return 10
        if parso_cache_node is None:
            return func(node, include_flows)
        try:
            for_module = cache[parso_cache_node]
        except KeyError:
            for_module = cache[parso_cache_node] = {}
        try:
            return for_module[node]
        except KeyError:
            result = for_module[node] = func(node, include_flows)
            return result
    return wrapper

def get_parent_scope(node, include_flows=False):
    if False:
        i = 10
        return i + 15
    '\n    Returns the underlying scope.\n    '
    scope = node.parent
    if scope is None:
        return None
    while True:
        if is_scope(scope):
            if scope.type in ('classdef', 'funcdef', 'lambdef'):
                index = scope.children.index(':')
                if scope.children[index].start_pos >= node.start_pos:
                    if node.parent.type == 'param' and node.parent.name == node:
                        pass
                    elif node.parent.type == 'tfpdef' and node.parent.children[0] == node:
                        pass
                    else:
                        scope = scope.parent
                        continue
            return scope
        elif include_flows and isinstance(scope, tree.Flow):
            if not (scope.type == 'if_stmt' and any((n.start_pos <= node.start_pos < n.end_pos for n in scope.get_test_nodes()))):
                return scope
        scope = scope.parent
get_cached_parent_scope = _get_parent_scope_cache(get_parent_scope)

def get_cached_code_lines(grammar, path):
    if False:
        i = 10
        return i + 15
    '\n    Basically access the cached code lines in parso. This is not the nicest way\n    to do this, but we avoid splitting all the lines again.\n    '
    return get_parso_cache_node(grammar, path).lines

def get_parso_cache_node(grammar, path):
    if False:
        for i in range(10):
            print('nop')
    "\n    This is of course not public. But as long as I control parso, this\n    shouldn't be a problem. ~ Dave\n\n    The reason for this is mostly caching. This is obviously also a sign of a\n    broken caching architecture.\n    "
    return parser_cache[grammar._hashed][path]

def cut_value_at_position(leaf, position):
    if False:
        for i in range(10):
            print('nop')
    '\n    Cuts of the value of the leaf at position\n    '
    lines = split_lines(leaf.value, keepends=True)[:position[0] - leaf.line + 1]
    column = position[1]
    if leaf.line == position[0]:
        column -= leaf.column
    if not lines:
        return ''
    lines[-1] = lines[-1][:column]
    return ''.join(lines)

def expr_is_dotted(node):
    if False:
        i = 10
        return i + 15
    '\n    Checks if a path looks like `name` or `name.foo.bar` and not `name()`.\n    '
    if node.type == 'atom':
        if len(node.children) == 3 and node.children[0] == '(':
            return expr_is_dotted(node.children[1])
        return False
    if node.type == 'atom_expr':
        children = node.children
        if children[0] == 'await':
            return False
        if not expr_is_dotted(children[0]):
            return False
        return all((c.children[0] == '.' for c in children[1:]))
    return node.type == 'name'

def _function_is_x_method(*method_names):
    if False:
        print('Hello World!')

    def wrapper(function_node):
        if False:
            print('Hello World!')
        "\n        This is a heuristic. It will not hold ALL the times, but it will be\n        correct pretty much for anyone that doesn't try to beat it.\n        staticmethod/classmethod are builtins and unless overwritten, this will\n        be correct.\n        "
        for decorator in function_node.get_decorators():
            dotted_name = decorator.children[1]
            if dotted_name.get_code() in method_names:
                return True
        return False
    return wrapper
function_is_staticmethod = _function_is_x_method('staticmethod')
function_is_classmethod = _function_is_x_method('classmethod')
function_is_property = _function_is_x_method('property', 'cached_property')