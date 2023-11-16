import copy
from collections import defaultdict
from paddle.base import unique_name
from paddle.utils import gast
from .base_transformer import BaseTransformer, ForLoopTuplePreTransformer, ForNodeVisitor
from .ifelse_transformer import ARGS_NAME
from .static_analysis import NodeVarType, StaticAnalysisVisitor
from .utils import FOR_BODY_PREFIX, FOR_CONDITION_PREFIX, WHILE_BODY_PREFIX, WHILE_CONDITION_PREFIX, FunctionNameLivenessAnalysis, GetterSetterHelper, ast_to_source_code, create_get_args_node, create_name_str, create_nonlocal_stmt_nodes, create_set_args_node, get_attribute_full_name
__all__ = []

def create_while_nodes(condition_name, body_name, loop_var_names, push_pop_names, getter_name, setter_name):
    if False:
        return 10
    "\n    Returns a list of gast.Node which represents the calling of Paddle\n    controlflow while_loop.\n\n    Usually, the list just contain 1 statement such as:\n\n    [a, b, c] = paddle.jit.dy2static.convert_while_loop(\n            condition_name, body_name, [a, b, c])\n\n    where a, b, c are in loop_var_names.\n\n    However, if loop_var_names contains property such as foo.x, we cannot\n    assign the property as output of convert_while_loop because Python\n    property is a kind of read-only attribute. To handle the case, we replace\n    the attributes which are output of convert_while_loop with generated\n    variables, then if we know the attribute is not read-only at runtime, we\n    assign the attribute. The created statements are like:\n\n    [a, b, __attribute_variable_1] = paddle.jit.dy2static.convert_while_loop(\n            condition_name, body_name, [a, b, foo.x])\n    if not isinstance(getattr(type(foo), x, None), property): foo.x = __attribute_variable_1\n\n    The number of above statements is not only 1, that's why the return type is\n    a list of gast.Node.\n    "
    loop_var_names = list(loop_var_names)
    assign_loop_var_names = []
    for name in loop_var_names:
        assign_loop_var_names.append(name)
    while_func_name = '_jst.While'
    while_node_str = '{}({}, {}, {}, {}, return_name_ids={}, push_pop_names={})'.format(while_func_name, condition_name, body_name, getter_name, setter_name, create_name_str(loop_var_names), create_name_str(push_pop_names))
    while_node = gast.parse(while_node_str).body[0]
    ret = [while_node]
    return ret

class NameVisitor(gast.NodeVisitor):
    """
    Analysis name liveness for loop transformer
    """

    def __init__(self, root_node):
        if False:
            i = 10
            return i + 15
        self.current_seen_vars = set()
        self.current_loop = []
        self.nodes_with_scope = []
        self.blacklist_names = {'False', 'True', 'None'}
        self.before_loop_body_vars = defaultdict(set)
        self.in_loop_vars = defaultdict(list)
        self.write_in_loop = defaultdict(set)
        self.condition_vars = defaultdict(set)
        self.in_condition = False
        self.type_vars = set()
        self.static_analysis_visitor = StaticAnalysisVisitor(root_node)
        self.node_to_wrapper_map = self.static_analysis_visitor.get_node_to_wrapper_map()
        self.visit(root_node)

    def get_loop_var_names(self, node):
        if False:
            print('Hello World!')
        assert isinstance(node, (gast.While, gast.For)), 'Input node is not gast loop node'
        loop_var_names = set()
        create_var_names = set()
        read_context = {type(gast.Load()), type(gast.AugLoad())}
        in_loop_vars_list = self.in_loop_vars[node]
        var_name_to_ctxs = defaultdict(list)
        for var_node in in_loop_vars_list:
            var_name_to_ctxs[self._var_node_to_name(var_node)].append(var_node.ctx)
        in_loop_vars = set(in_loop_vars_list)
        in_loop_vars = self._remove_unnecessary_vars(in_loop_vars, node)
        in_loop_name_strs = self._var_nodes_to_names(in_loop_vars)
        before_loop_body_vars = self.before_loop_body_vars[node]
        before_loop_body_vars = self._remove_unnecessary_vars(before_loop_body_vars, node)
        before_loop_name_strs = self._var_nodes_to_names(before_loop_body_vars)
        after_loop_vars = self.current_seen_vars - before_loop_body_vars - in_loop_vars
        after_loop_vars = self._remove_unnecessary_vars(after_loop_vars, node)
        after_loop_name_strs = self._var_nodes_to_names(after_loop_vars, read_context)
        condition_vars = self.condition_vars[node]
        condition_names = self._var_nodes_to_names(condition_vars)
        write_vars = self.write_in_loop[node]
        write_names = self._var_nodes_to_names(write_vars)
        name_to_type = {}
        for var in in_loop_vars:
            wrapper = self.node_to_wrapper_map[var]
            name_to_type[self._var_node_to_name(var)] = wrapper.node_var_type
        for name in in_loop_name_strs:
            if name in before_loop_name_strs:
                if name not in condition_names and name not in write_names:
                    continue
                loop_var_names.add(name)
            elif name in after_loop_name_strs:
                loop_var_names.add(name)
                create_var_names.add(name)
            else:
                is_created = False
                for ctx in var_name_to_ctxs[name]:
                    if isinstance(ctx, gast.Store):
                        is_created = True
                if isinstance(var_name_to_ctxs[name][0], gast.Load) and is_created:
                    loop_var_names.add(name)
                    create_var_names.add(name)
        return (loop_var_names, create_var_names)

    def visit_Name(self, node):
        if False:
            for i in range(10):
                print('nop')
        if self._is_call_func_name_node(node):
            self.generic_visit(node)
            return
        if node.id in self.blacklist_names:
            self.generic_visit(node)
            return
        self.current_seen_vars.add(node)
        write_context = {type(gast.Store()), type(gast.AugStore()), type(gast.Del())}
        for loop_node in self.current_loop:
            self.in_loop_vars[loop_node].append(node)
            if type(node.ctx) in write_context:
                self.write_in_loop[loop_node].add(node)
        if self.in_condition:
            self.condition_vars[loop_node].add(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if False:
            while True:
                i = 10
        self.nodes_with_scope.append(node)
        self.blacklist_names.add(node.name)
        before_func_seen_vars = copy.copy(self.current_seen_vars)
        self.generic_visit(node)
        self.nodes_with_scope.pop()
        if self.nodes_with_scope:
            self.current_seen_vars = before_func_seen_vars

    def visit(self, node):
        if False:
            for i in range(10):
                print('nop')
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        ret = visitor(node)
        return ret

    def visit_Attribute(self, node):
        if False:
            print('Hello World!')
        if self._is_call_func_name_node(node):
            return
        attr_full_name = get_attribute_full_name(node)
        '\n        def class_func(self):\n            def while_loop_body(self.x, y) # `self.x` is illegal.\n        '
        if attr_full_name.startswith('self.'):
            return
        self.current_seen_vars.add(node)
        for loop_node in self.current_loop:
            self.in_loop_vars[loop_node].append(node)

    def visit_For(self, node):
        if False:
            while True:
                i = 10
        self.current_loop.append(node)
        self.in_condition = True
        self.visit(node.target)
        self.visit(node.iter)
        self.in_condition = False
        self.before_loop_body_vars[node] = copy.copy(self.current_seen_vars)
        self.generic_visit(node)
        self.current_loop.pop()

    def visit_While(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.current_loop.append(node)
        self.in_condition = True
        self.visit(node.test)
        self.in_condition = False
        self.before_loop_body_vars[node] = copy.copy(self.current_seen_vars)
        self.generic_visit(node)
        self.current_loop.pop()

    def visit_Call(self, node):
        if False:
            while True:
                i = 10
        if isinstance(node.func, gast.Name) and node.func.id == 'isinstance':
            type_node = node.args[1]
            if isinstance(type_node, gast.Tuple):
                for element in type_node.elts:
                    self.type_vars.add(ast_to_source_code(element).strip())
            else:
                self.type_vars.add(ast_to_source_code(type_node).strip())
        self.generic_visit(node)

    def _var_nodes_to_names(self, node_set, ctx_filter_set=None):
        if False:
            return 10
        ret = set()
        for node in node_set:
            if ctx_filter_set is None or type(node.ctx) in ctx_filter_set:
                ret.add(self._var_node_to_name(node))
        return ret

    def _var_node_to_name(self, node):
        if False:
            while True:
                i = 10
        if isinstance(node, gast.Name):
            return node.id
        elif isinstance(node, gast.Attribute):
            return get_attribute_full_name(node)

    def _node_var_type_is_basic(self, node_var_type):
        if False:
            print('Hello World!')
        basic_types = {NodeVarType.BOOLEAN, NodeVarType.INT, NodeVarType.FLOAT, NodeVarType.STRING}
        for t in node_var_type:
            if t in basic_types:
                return True
        return False

    def _is_call_func_name_node(self, node):
        if False:
            print('Hello World!')
        parent_node = self._get_parent_node(node)
        if isinstance(parent_node, gast.Call) and parent_node.func == node:
            return True
        return False

    def _is_global_or_nonlocal(self, node):
        if False:
            i = 10
            return i + 15
        return False

    def _is_ancestor_node(self, ancestor_node, node):
        if False:
            i = 10
            return i + 15
        parent_node = self._get_parent_node(node)
        while parent_node is not None:
            if parent_node == ancestor_node:
                return True
            parent_node = self._get_parent_node(parent_node)
        return False

    def _get_parent_node(self, node):
        if False:
            while True:
                i = 10
        wrapper_node = self.node_to_wrapper_map.get(node)
        if wrapper_node:
            if wrapper_node.parent:
                parent_node = wrapper_node.parent.node
                return parent_node
        return None

    def _remove_unnecessary_vars(self, loop_vars, loop_node):
        if False:
            while True:
                i = 10
        '\n        Remove unnecessary vars from before_loop_vars, after_loop_vars or in_loop_vars about loop_node.\n            1. Remove target vars of gast.For from before_loop_vars or after_loop_vars.\n            2. Remove vars only in gast.comprehension.\n            3. Remove vars that are type names, for example: "isinstance(x, var_type_name)"\n        :param loop_vars: before_loop_vars, after_loop_vars or in_loop_vars of loop_node.\n        :param loop_node: Current loop node.\n        '

        def filter_name_nodes_from(root_node, target_var_names):
            if False:
                i = 10
                return i + 15
            '\n            Filter children with gast.Name type from node.(inclusivly)\n            '
            name_nodes = set()
            if isinstance(root_node, gast.Name):
                if node.id in target_var_names:
                    name_nodes.add(root_node)
            for child_node in gast.walk(root_node):
                if isinstance(child_node, gast.Name):
                    if child_node.id in target_var_names:
                        name_nodes.add(child_node)
            return name_nodes
        vars_of_list_generator = set()
        target_vars_of_for_node = set()
        for name_node in loop_vars:
            if not isinstance(name_node, gast.Name):
                continue
            parent_node = self._get_parent_node(name_node)
            if isinstance(parent_node, gast.Tuple):
                parent_node = self._get_parent_node(parent_node)
            if isinstance(parent_node, gast.comprehension):
                target_node = parent_node.target
                if isinstance(target_node, gast.Tuple):
                    target_vars = target_node.elts
                else:
                    target_vars = [target_node]
                vars_of_list_generator = vars_of_list_generator | set(target_vars)
                target_var_names = {var.id for var in target_vars}
                comp_node = self._get_parent_node(parent_node)
                elt_nodes = []
                if isinstance(comp_node, gast.ListComp):
                    elt_nodes.append(comp_node.elt)
                elif isinstance(comp_node, gast.DictComp):
                    elt_nodes.extend([comp_node.key, comp_node.value])
                for node in elt_nodes:
                    vars_of_list_generator |= filter_name_nodes_from(node, target_var_names)
            elif isinstance(parent_node, gast.For):
                if parent_node is loop_node:
                    continue
                if self._is_ancestor_node(parent_node, loop_node):
                    continue
                target_node = parent_node.target
                if isinstance(target_node, gast.Tuple):
                    target_vars = target_node.elts
                else:
                    target_vars = [target_node]
                target_vars_of_for_node = target_vars_of_for_node | set(target_vars)
        target_vars_name_strs = {var.id for var in target_vars_of_for_node}
        for var in loop_vars:
            if not isinstance(var, gast.Name):
                continue
            if var.id in target_vars_name_strs and var not in self.condition_vars[loop_node]:
                target_vars_of_for_node.add(var)
        removed_vars = target_vars_of_for_node | vars_of_list_generator
        for var in loop_vars:
            if ast_to_source_code(var).strip() in self.type_vars:
                removed_vars.add(var)
        return loop_vars - removed_vars

class LoopTransformer(BaseTransformer):
    """
    This class transforms python while/for statement into Static Graph Ast
    """

    def __init__(self, root):
        if False:
            i = 10
            return i + 15
        self.root = root
        FunctionNameLivenessAnalysis(self.root)

    def transform(self):
        if False:
            i = 10
            return i + 15
        ForLoopTuplePreTransformer(self.root).transform()
        self.visit(self.root)

    def visit_While(self, node):
        if False:
            while True:
                i = 10
        self.generic_visit(node)
        new_stmts = self.get_while_stmt_nodes(node)
        return new_stmts

    def visit_For(self, node):
        if False:
            while True:
                i = 10
        self.generic_visit(node)
        new_stmts = self.get_for_stmt_nodes(node)
        return new_stmts

    def replace_stmt_list(self, body_list):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(body_list, list):
            return
        i = 0
        while i < len(body_list):
            if isinstance(body_list[i], gast.While):
                new_stmts = self.get_while_stmt_nodes(body_list[i])
                body_list[i:i + 1] = new_stmts
                i += len(new_stmts)
            elif isinstance(body_list[i], gast.For):
                new_stmts = self.get_for_stmt_nodes(body_list[i])
                body_list[i:i + 1] = new_stmts
                i += len(new_stmts)
            else:
                i += 1

    def get_for_stmt_nodes(self, node):
        if False:
            i = 10
            return i + 15
        current_for_node_parser = ForNodeVisitor(node)
        stmts_tuple = current_for_node_parser.parse()
        if stmts_tuple is None:
            return [node]
        (init_stmts, cond_stmt, body_stmts) = stmts_tuple
        (loop_var_names, create_var_names) = (node.pd_scope.modified_vars(), node.pd_scope.created_vars())
        push_pop_names = list(node.pd_scope.variadic_length_vars())
        if current_for_node_parser.is_for_iter():
            iter_var_name = current_for_node_parser.iter_var_name
            iter_idx_name = current_for_node_parser.iter_idx_name
            loop_var_names.add(iter_idx_name)
            if current_for_node_parser.enum_idx_name is not None:
                loop_var_names.add(current_for_node_parser.enum_idx_name)
        new_stmts = []
        nonlocal_names = list(loop_var_names | create_var_names)
        nonlocal_names.sort()
        if ARGS_NAME in nonlocal_names:
            nonlocal_names.remove(ARGS_NAME)
        nonlocal_stmt_node = create_nonlocal_stmt_nodes(nonlocal_names)
        new_stmts.extend(init_stmts)
        condition_func_node = gast.FunctionDef(name=unique_name.generate(FOR_CONDITION_PREFIX), args=gast.arguments(args=[], posonlyargs=[], vararg=None, kwonlyargs=[], kw_defaults=None, kwarg=None, defaults=[]), body=nonlocal_stmt_node + [gast.Return(value=cond_stmt)], decorator_list=[], returns=None, type_comment=None)
        new_stmts.append(condition_func_node)
        body_func_node = gast.FunctionDef(name=unique_name.generate(FOR_BODY_PREFIX), args=gast.arguments(args=[], posonlyargs=[], vararg=None, kwonlyargs=[], kw_defaults=None, kwarg=None, defaults=[]), body=nonlocal_stmt_node + body_stmts, decorator_list=[], returns=None, type_comment=None)
        new_stmts.append(body_func_node)
        helper = GetterSetterHelper(None, None, nonlocal_names, push_pop_names)
        get_args_node = create_get_args_node(helper.union())
        set_args_node = create_set_args_node(helper.union())
        while_loop_nodes = create_while_nodes(condition_func_node.name, body_func_node.name, nonlocal_names, push_pop_names, get_args_node.name, set_args_node.name)
        new_stmts.extend([get_args_node, set_args_node])
        new_stmts.extend(while_loop_nodes)
        return new_stmts

    def get_while_stmt_nodes(self, node):
        if False:
            return 10
        (loop_var_names, create_var_names) = (node.pd_scope.modified_vars(), node.pd_scope.created_vars())
        push_pop_names = list(node.pd_scope.variadic_length_vars())
        new_stmts = []
        nonlocal_names = list(loop_var_names | create_var_names)
        nonlocal_names.sort()
        if ARGS_NAME in nonlocal_names:
            nonlocal_names.remove(ARGS_NAME)
        nonlocal_stmt_node = create_nonlocal_stmt_nodes(nonlocal_names)
        condition_func_node = gast.FunctionDef(name=unique_name.generate(WHILE_CONDITION_PREFIX), args=gast.arguments(args=[], posonlyargs=[], vararg=None, kwonlyargs=[], kw_defaults=None, kwarg=None, defaults=[]), body=nonlocal_stmt_node + [gast.Return(value=node.test)], decorator_list=[], returns=None, type_comment=None)
        new_stmts.append(condition_func_node)
        new_body = node.body
        body_func_node = gast.FunctionDef(name=unique_name.generate(WHILE_BODY_PREFIX), args=gast.arguments(args=[], posonlyargs=[], vararg=None, kwonlyargs=[], kw_defaults=None, kwarg=None, defaults=[]), body=nonlocal_stmt_node + new_body, decorator_list=[], returns=None, type_comment=None)
        new_stmts.append(body_func_node)
        helper = GetterSetterHelper(None, None, nonlocal_names, push_pop_names)
        get_args_node = create_get_args_node(helper.union())
        set_args_node = create_set_args_node(helper.union())
        while_loop_nodes = create_while_nodes(condition_func_node.name, body_func_node.name, nonlocal_names, push_pop_names, get_args_node.name, set_args_node.name)
        new_stmts.extend([get_args_node, set_args_node])
        new_stmts.extend(while_loop_nodes)
        return new_stmts