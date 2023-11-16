from paddle.base import unique_name
from paddle.utils import gast
from .base_transformer import BaseTransformer
from .break_continue_transformer import ForToWhileTransformer
from .utils import ORIGI_INFO, Dygraph2StaticException, ast_to_source_code, index_in_list
__all__ = []
RETURN_PREFIX = '__return'
RETURN_VALUE_PREFIX = '__return_value'
RETURN_VALUE_INIT_NAME = '__return_value_init'

def get_return_size(return_node):
    if False:
        return 10
    assert isinstance(return_node, gast.Return), 'Input is not gast.Return node'
    return_length = 0
    if return_node.value is not None:
        if isinstance(return_node.value, gast.Tuple):
            return_length = len(return_node.value.elts)
        else:
            return_length = 1
    return return_length

class ReplaceReturnNoneTransformer(BaseTransformer):
    """
    Replace 'return None' to  'return' because 'None' cannot be a valid input
    in control flow. In ReturnTransformer single 'Return' will be appended no
    value placeholder
    """

    def __init__(self, root_node):
        if False:
            return 10
        self.root = root_node

    def transform(self):
        if False:
            return 10
        self.visit(self.root)

    def visit_Return(self, node):
        if False:
            while True:
                i = 10
        if isinstance(node.value, gast.Name) and node.value.id == 'None':
            node.value = None
            return node
        if isinstance(node.value, gast.Constant) and node.value.value is None:
            node.value = None
            return node
        return node

class ReturnAnalysisVisitor(gast.NodeVisitor):
    """
    Visits gast Tree and analyze the information about 'return'.
    """

    def __init__(self, root_node):
        if False:
            i = 10
            return i + 15
        self.root = root_node
        assert isinstance(self.root, gast.FunctionDef), 'Input is not gast.FunctionDef node'
        self.count_return = 0
        self.max_return_length = 0
        self.visit(self.root)

    def visit_FunctionDef(self, node):
        if False:
            return 10
        "\n        don't analysis closure, just analyze current func def level.\n        "
        if node == self.root:
            self.generic_visit(node)

    def visit_Return(self, node):
        if False:
            i = 10
            return i + 15
        self.count_return += 1
        return_length = get_return_size(node)
        self.max_return_length = max(self.max_return_length, return_length)
        self.generic_visit(node)

    def get_func_return_count(self):
        if False:
            return 10
        return self.count_return

    def get_func_max_return_length(self):
        if False:
            i = 10
            return i + 15
        return self.max_return_length

class ReturnTransformer(BaseTransformer):
    """
    Transforms return statements into equivalent python statements containing
    only one return statement at last. The basics idea is using a return value
    variable to store the early return statements and boolean states with
    if-else to skip the statements after the return.

    Go through all the function definition and call SingleReturnTransformer for each function.
    SingleReturnTransformer don't care the nested function def.
    """

    def __init__(self, root):
        if False:
            return 10
        self.root = root
        pre_transformer = ReplaceReturnNoneTransformer(self.root)
        pre_transformer.transform()

    def transform(self):
        if False:
            while True:
                i = 10
        self.visit(self.root)

    def visit_FunctionDef(self, node):
        if False:
            print('Hello World!')
        node = self.generic_visit(node)
        node = SingleReturnTransformer(node).transform()
        return node

class SingleReturnTransformer(BaseTransformer):
    """
    This function only apply to single function. don't care the nested function_def
    """

    def __init__(self, root):
        if False:
            print('Hello World!')
        self.root = root
        assert isinstance(self.root, gast.FunctionDef), 'Input is not gast.FunctionDef node'
        self.ancestor_nodes = []
        self.return_value_name = None
        self.return_name = []
        self.pre_analysis = None

    def assert_parent_is_not_while(self, parent_node_of_return):
        if False:
            while True:
                i = 10
        if isinstance(parent_node_of_return, (gast.While, gast.For)):
            raise Dygraph2StaticException('Found return statement in While or For body and loop is meaningless, please check you code and remove return in while/for.')

    def generic_visit(self, node):
        if False:
            return 10
        for (field, value) in gast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, gast.AST):
                        self.visit(item)
            elif isinstance(value, gast.AST):
                self.visit(value)

    def visit(self, node):
        if False:
            return 10
        '\n        Self-defined visit for appending ancestor\n        '
        self.ancestor_nodes.append(node)
        ret = super().visit(node)
        self.ancestor_nodes.pop()
        return ret

    def visit_FunctionDef(self, node):
        if False:
            return 10
        "\n        don't analysis closure, just analyze current func def level.\n        "
        if node == self.root:
            self.generic_visit(node)
        return node

    def append_assign_to_return_node(self, value, parent_node_of_return, return_name, assign_nodes):
        if False:
            return 10
        self.assert_parent_is_not_while(parent_node_of_return)
        assert value in [True, False], 'value must be True or False.'
        if isinstance(parent_node_of_return, gast.If):
            node_str = '{} = _jst.create_bool_as_type({}, {})'.format(return_name, ast_to_source_code(parent_node_of_return.test).strip(), value)
            assign_node = gast.parse(node_str).body[0]
            assign_nodes.append(assign_node)

    def transform(self):
        if False:
            for i in range(10):
                print('nop')
        node = self.root
        self.pre_analysis = ReturnAnalysisVisitor(node)
        max_return_length = self.pre_analysis.get_func_max_return_length()
        while self.pre_analysis.get_func_return_count() > 0:
            self.visit(node)
            self.pre_analysis = ReturnAnalysisVisitor(node)
        if max_return_length == 0:
            return node
        value_name = self.return_value_name
        if value_name is not None:
            node.body.append(gast.Return(value=gast.Name(id=value_name, ctx=gast.Load(), annotation=None, type_comment=None)))
            assign_return_value_node = gast.Assign(targets=[gast.Name(id=value_name, ctx=gast.Store(), annotation=None, type_comment=None)], value=gast.Constant(kind=None, value=None))
            node.body.insert(0, assign_return_value_node)
        return node

    def visit_Return(self, node):
        if False:
            i = 10
            return i + 15
        return_name = unique_name.generate(RETURN_PREFIX)
        self.return_name.append(return_name)
        max_return_length = self.pre_analysis.get_func_max_return_length()
        parent_node_of_return = self.ancestor_nodes[-2]
        for ancestor_index in reversed(range(len(self.ancestor_nodes) - 1)):
            ancestor = self.ancestor_nodes[ancestor_index]
            cur_node = self.ancestor_nodes[ancestor_index + 1]

            def _deal_branches(branch_name):
                if False:
                    while True:
                        i = 10
                if hasattr(ancestor, branch_name):
                    branch_node = getattr(ancestor, branch_name)
                    if index_in_list(branch_node, cur_node) != -1:
                        if cur_node == node:
                            self._replace_return_in_stmt_list(branch_node, cur_node, return_name, max_return_length, parent_node_of_return)
                        self._replace_after_node_to_if_in_stmt_list(branch_node, cur_node, return_name, parent_node_of_return)
            _deal_branches('body')
            _deal_branches('orelse')
            if isinstance(ancestor, gast.While):
                cond_var_node = gast.UnaryOp(op=gast.Not(), operand=gast.Name(id=return_name, ctx=gast.Load(), annotation=None, type_comment=None))
                ancestor.test = gast.BoolOp(op=gast.And(), values=[ancestor.test, cond_var_node])
                continue
            if isinstance(ancestor, gast.For):
                cond_var_node = gast.UnaryOp(op=gast.Not(), operand=gast.Name(id=return_name, ctx=gast.Load(), annotation=None, type_comment=None))
                parent_node = self.ancestor_nodes[ancestor_index - 1]
                for_to_while = ForToWhileTransformer(parent_node, ancestor, cond_var_node)
                new_stmts = for_to_while.transform()
                while_node = new_stmts[-1]
                self.ancestor_nodes[ancestor_index] = while_node
            if ancestor == self.root:
                break

    def _replace_return_in_stmt_list(self, stmt_list, return_node, return_name, max_return_length, parent_node_of_return):
        if False:
            i = 10
            return i + 15
        assert max_return_length >= 0, 'Input illegal max_return_length'
        i = index_in_list(stmt_list, return_node)
        if i == -1:
            return False
        assign_nodes = []
        self.append_assign_to_return_node(True, parent_node_of_return, return_name, assign_nodes)
        return_length = get_return_size(return_node)
        if return_node.value is not None:
            if self.return_value_name is None:
                self.return_value_name = unique_name.generate(RETURN_VALUE_PREFIX)
            assign_nodes.append(gast.Assign(targets=[gast.Name(id=self.return_value_name, ctx=gast.Store(), annotation=None, type_comment=None)], value=return_node.value))
            return_origin_info = getattr(return_node, ORIGI_INFO, None)
            setattr(assign_nodes[-1], ORIGI_INFO, return_origin_info)
        stmt_list[i:] = assign_nodes
        return True

    def _replace_after_node_to_if_in_stmt_list(self, stmt_list, node, return_name, parent_node_of_return):
        if False:
            for i in range(10):
                print('nop')
        i = index_in_list(stmt_list, node)
        if i < 0 or i >= len(stmt_list):
            return False
        if i == len(stmt_list) - 1:
            return True
        if_stmt = gast.If(test=gast.UnaryOp(op=gast.Not(), operand=gast.Name(id=return_name, ctx=gast.Store(), annotation=None, type_comment=None)), body=stmt_list[i + 1:], orelse=[])
        stmt_list[i + 1:] = [if_stmt]
        assign_nodes = []
        self.append_assign_to_return_node(False, parent_node_of_return, return_name, assign_nodes)
        stmt_list[i:i] = assign_nodes
        return True