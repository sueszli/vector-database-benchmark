import copy
from collections import defaultdict
from paddle.base import unique_name
from paddle.jit.dy2static.utils import FOR_ITER_INDEX_PREFIX, FOR_ITER_ITERATOR_PREFIX, FOR_ITER_TARGET_PREFIX, FOR_ITER_TUPLE_INDEX_PREFIX, FOR_ITER_TUPLE_PREFIX, FOR_ITER_VAR_LEN_PREFIX, FOR_ITER_VAR_NAME_PREFIX, FOR_ITER_ZIP_TO_LIST_PREFIX, FunctionNameLivenessAnalysis, GetterSetterHelper, ast_to_source_code, create_funcDef_node, create_get_args_node, create_name_str, create_nonlocal_stmt_nodes, create_set_args_node
from paddle.utils import gast
from .base_transformer import BaseTransformer
from .utils import FALSE_FUNC_PREFIX, TRUE_FUNC_PREFIX
__all__ = []
GET_ARGS_FUNC_PREFIX = 'get_args'
SET_ARGS_FUNC_PREFIX = 'set_args'
ARGS_NAME = '__args'

class IfElseTransformer(BaseTransformer):
    """
    Transform if/else statement of Dygraph into Static Graph.
    """

    def __init__(self, root):
        if False:
            i = 10
            return i + 15
        self.root = root
        FunctionNameLivenessAnalysis(self.root)

    def transform(self):
        if False:
            while True:
                i = 10
        '\n        Main function to transform AST.\n        '
        self.visit(self.root)

    def visit_If(self, node):
        if False:
            while True:
                i = 10
        self.generic_visit(node)
        (true_func_node, false_func_node, get_args_node, set_args_node, return_name_ids, push_pop_ids) = transform_if_else(node, self.root)
        new_node = create_convert_ifelse_node(return_name_ids, push_pop_ids, node.test, true_func_node, false_func_node, get_args_node, set_args_node)
        return [get_args_node, set_args_node, true_func_node, false_func_node] + [new_node]

    def visit_Call(self, node):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(node.func, gast.Attribute):
            attribute = node.func
            if attribute.attr == 'numpy':
                node = attribute.value
        self.generic_visit(node)
        return node

    def visit_IfExp(self, node):
        if False:
            while True:
                i = 10
        '\n        Transformation with `true_fn(x) if Tensor > 0 else false_fn(x)`\n        '
        self.generic_visit(node)
        new_node = create_convert_ifelse_node(None, None, node.test, node.body, node.orelse, None, None, True)
        if isinstance(new_node, gast.Expr):
            new_node = new_node.value
        return new_node

class NameVisitor(gast.NodeVisitor):

    def __init__(self, after_node=None, end_node=None):
        if False:
            i = 10
            return i + 15
        self.after_node = after_node
        self.end_node = end_node
        self.name_ids = defaultdict(list)
        self.ancestor_nodes = []
        self._in_range = after_node is None
        self._candidate_ctxs = (gast.Store, gast.Load, gast.Param)
        self._def_func_names = set()

    def visit(self, node):
        if False:
            i = 10
            return i + 15
        'Visit a node.'
        if self.after_node is not None and node == self.after_node:
            self._in_range = True
            return
        if node == self.end_node:
            self._in_range = False
            return
        self.ancestor_nodes.append(node)
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        ret = visitor(node)
        self.ancestor_nodes.pop()
        return ret

    def visit_If(self, node):
        if False:
            for i in range(10):
                print('nop')
        '\n        For nested `if/else`, the created vars are not always visible for parent node.\n        In addition, the vars created in `if.body` are not visible for `if.orelse`.\n\n        Case 1:\n            x = 1\n            if m > 1:\n                res = new_tensor\n            res = res + 1   # Error, `res` is not visible here.\n\n        Case 2:\n            if x_tensor > 0:\n                res = new_tensor\n            else:\n                res = res + 1   # Error, `res` is not visible here.\n\n        In above two cases, we should consider to manage the scope of vars to parsing\n        the arguments and returned vars correctly.\n        '
        if not self._in_range or not self.end_node:
            self.generic_visit(node)
            return
        else:
            before_if_name_ids = copy.deepcopy(self.name_ids)
            body_name_ids = self._visit_child(node.body)
            if not self._in_range:
                self._update_name_ids(before_if_name_ids)
            else:
                else_name_ids = self._visit_child(node.orelse)
                if not self._in_range:
                    self._update_name_ids(before_if_name_ids)
                else:
                    new_name_ids = self._find_new_name_ids(body_name_ids, else_name_ids)
                    for new_name_id in new_name_ids:
                        before_if_name_ids[new_name_id].append(gast.Store())
                    self.name_ids = before_if_name_ids

    def visit_Attribute(self, node):
        if False:
            while True:
                i = 10
        if not self._in_range or not self._is_call_func_name_node(node):
            self.generic_visit(node)

    def visit_Name(self, node):
        if False:
            i = 10
            return i + 15
        if not self._in_range:
            self.generic_visit(node)
            return
        blacklist = {'True', 'False', 'None'}
        if node.id in blacklist:
            return
        if node.id in self._def_func_names:
            return
        if not self._is_call_func_name_node(node):
            if isinstance(node.ctx, self._candidate_ctxs):
                self.name_ids[node.id].append(node.ctx)

    def visit_Assign(self, node):
        if False:
            print('Hello World!')
        if not self._in_range:
            self.generic_visit(node)
            return
        node._fields = ('value', 'targets')
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if False:
            for i in range(10):
                print('nop')
        if GET_ARGS_FUNC_PREFIX in node.name or SET_ARGS_FUNC_PREFIX in node.name:
            return
        if not self._in_range:
            self.generic_visit(node)
            return
        self._def_func_names.add(node.name)
        if not self.end_node:
            self.generic_visit(node)
        else:
            before_name_ids = copy.deepcopy(self.name_ids)
            self.name_ids = defaultdict(list)
            self.generic_visit(node)
            if not self._in_range:
                self._update_name_ids(before_name_ids)
            else:
                self.name_ids = before_name_ids

    def _visit_child(self, node):
        if False:
            print('Hello World!')
        self.name_ids = defaultdict(list)
        if isinstance(node, list):
            for item in node:
                if isinstance(item, gast.AST):
                    self.visit(item)
        elif isinstance(node, gast.AST):
            self.visit(node)
        return copy.deepcopy(self.name_ids)

    def _find_new_name_ids(self, body_name_ids, else_name_ids):
        if False:
            i = 10
            return i + 15

        def is_required_ctx(ctxs, required_ctx):
            if False:
                return 10
            for ctx in ctxs:
                if isinstance(ctx, required_ctx):
                    return True
            return False
        candidate_name_ids = set(body_name_ids.keys()) & set(else_name_ids.keys())
        store_ctx = gast.Store
        new_name_ids = set()
        for name_id in candidate_name_ids:
            if is_required_ctx(body_name_ids[name_id], store_ctx) and is_required_ctx(else_name_ids[name_id], store_ctx):
                new_name_ids.add(name_id)
        return new_name_ids

    def _is_call_func_name_node(self, node):
        if False:
            return 10
        white_func_names = {'append', 'extend'}
        if len(self.ancestor_nodes) > 1:
            assert self.ancestor_nodes[-1] == node
            parent_node = self.ancestor_nodes[-2]
            if isinstance(parent_node, gast.Call) and parent_node.func == node:
                should_skip = isinstance(node, gast.Attribute) and node.attr in white_func_names
                if not should_skip:
                    return True
        return False

    def _update_name_ids(self, new_name_ids):
        if False:
            while True:
                i = 10
        for (name_id, ctxs) in new_name_ids.items():
            self.name_ids[name_id] = ctxs + self.name_ids[name_id]

def _valid_nonlocal_names(return_name_ids, nonlocal_names):
    if False:
        return 10
    '\n    All var in return_name_ids should be in nonlocal_names.\n    Moreover, we will always put return_name_ids in front of nonlocal_names.\n\n    For Example:\n\n        return_name_ids: [x, y]\n        nonlocal_names : [a, y, b, x]\n\n    Return:\n        nonlocal_names : [x, y, a, b]\n    '
    assert isinstance(return_name_ids, list)
    for name in return_name_ids:
        if name not in nonlocal_names:
            raise ValueError(f"Required returned var '{name}' must be in 'nonlocal' statement '', but not found.")
        nonlocal_names.remove(name)
    return return_name_ids + nonlocal_names

def transform_if_else(node, root):
    if False:
        i = 10
        return i + 15
    '\n    Transform ast.If into control flow statement of Paddle static graph.\n    '
    return_name_ids = sorted(node.pd_scope.modified_vars())
    push_pop_ids = sorted(node.pd_scope.variadic_length_vars())
    nonlocal_names = list(return_name_ids)
    nonlocal_names.sort()
    nonlocal_names = _valid_nonlocal_names(return_name_ids, nonlocal_names)
    filter_names = [ARGS_NAME, FOR_ITER_INDEX_PREFIX, FOR_ITER_TUPLE_PREFIX, FOR_ITER_TARGET_PREFIX, FOR_ITER_ITERATOR_PREFIX, FOR_ITER_TUPLE_INDEX_PREFIX, FOR_ITER_VAR_LEN_PREFIX, FOR_ITER_VAR_NAME_PREFIX, FOR_ITER_ZIP_TO_LIST_PREFIX]

    def remove_if(x):
        if False:
            return 10
        for name in filter_names:
            if x.startswith(name):
                return False
        return True
    nonlocal_names = list(filter(remove_if, nonlocal_names))
    return_name_ids = nonlocal_names
    nonlocal_stmt_node = create_nonlocal_stmt_nodes(nonlocal_names)
    empty_arg_node = gast.arguments(args=[], posonlyargs=[], vararg=None, kwonlyargs=[], kw_defaults=None, kwarg=None, defaults=[])
    true_func_node = create_funcDef_node(nonlocal_stmt_node + node.body, name=unique_name.generate(TRUE_FUNC_PREFIX), input_args=empty_arg_node, return_name_ids=[])
    false_func_node = create_funcDef_node(nonlocal_stmt_node + node.orelse, name=unique_name.generate(FALSE_FUNC_PREFIX), input_args=empty_arg_node, return_name_ids=[])
    helper = GetterSetterHelper(None, None, nonlocal_names, push_pop_ids)
    get_args_node = create_get_args_node(helper.union())
    set_args_node = create_set_args_node(helper.union())
    return (true_func_node, false_func_node, get_args_node, set_args_node, return_name_ids, push_pop_ids)

def create_convert_ifelse_node(return_name_ids, push_pop_ids, pred, true_func, false_func, get_args_func, set_args_func, is_if_expr=False):
    if False:
        return 10
    '\n    Create `paddle.jit.dy2static.convert_ifelse(\n            pred, true_fn, false_fn, get_args, set_args, return_name_ids)`\n    to replace original `python if/else` statement.\n    '
    if is_if_expr:
        true_func_source = f'lambda : {ast_to_source_code(true_func)}'
        false_func_source = f'lambda : {ast_to_source_code(false_func)}'
    else:
        true_func_source = true_func.name
        false_func_source = false_func.name
    convert_ifelse_layer = gast.parse('_jst.IfElse({pred}, {true_fn}, {false_fn}, {get_args}, {set_args}, {return_name_ids}, push_pop_names={push_pop_ids})'.format(pred=ast_to_source_code(pred), true_fn=true_func_source, false_fn=false_func_source, get_args=get_args_func.name if not is_if_expr else 'lambda: None', set_args=set_args_func.name if not is_if_expr else 'lambda args: None', return_name_ids=create_name_str(return_name_ids), push_pop_ids=create_name_str(push_pop_ids))).body[0]
    return convert_ifelse_layer