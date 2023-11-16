from paddle.utils import gast
from .utils_helper import NodeVarType, index_in_list, is_dygraph_api, is_numpy_api, is_paddle_api
__all__ = []

class AstNodeWrapper:
    """
    Wrapper for python gast.node. We need a node wrapper because gast.node
    doesn't store all required information when we are transforming AST.
    We should collect additional information which the actual transformation
    needs.
    """

    def __init__(self, node):
        if False:
            print('Hello World!')
        self.node = node
        self.parent = None
        self.children = []
        self.node_var_type = {NodeVarType.UNKNOWN}

class AstVarScope:
    """
    AstVarScope is a class holding the map from current scope variable to its
    type.
    """
    SCOPE_TYPE_SCRIPT = 0
    SCOPE_TYPE_FUNCTION = 1
    SCOPE_TYPE_CLASS = 2

    def __init__(self, scope_name='', scope_type=SCOPE_TYPE_SCRIPT, parent_scope=None):
        if False:
            while True:
                i = 10
        self.sub_scopes = []
        self.name_to_id = {}
        self.id_to_type = {}
        self.cur_id = 0
        self.scope_name = scope_name
        self.scope_type = scope_type
        self.parent_scope = parent_scope
        if parent_scope is not None:
            parent_scope.sub_scopes.append(self)

    def add_var_type(self, var_name, node_var_type):
        if False:
            i = 10
            return i + 15
        var_type = self.get_var_type(var_name)
        if var_type == {NodeVarType.UNKNOWN}:
            self.set_var_type(var_name, node_var_type)
        elif isinstance(node_var_type, set):
            var_type.update(node_var_type)
        else:
            var_type.add(node_var_type)

    def set_var_type(self, var_name, node_var_type):
        if False:
            i = 10
            return i + 15
        if var_name in self.name_to_id:
            num_id = self.name_to_id[var_name]
        else:
            num_id = self.cur_id
            self.cur_id += 1
            self.name_to_id[var_name] = num_id
        self.id_to_type[num_id] = node_var_type if isinstance(node_var_type, set) else {node_var_type}

    def get_var_type(self, var_name):
        if False:
            return 10
        if var_name in self.name_to_id:
            num_id = self.name_to_id[var_name]
            return self.id_to_type[num_id]
        if self.parent_scope is None:
            return {NodeVarType.UNKNOWN}
        return self.parent_scope.get_var_type(var_name)

class AstVarEnv:
    """
    A class maintains scopes and mapping from name strings to type.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.cur_scope = AstVarScope()

    def enter_scope(self, scope_name, scope_type):
        if False:
            return 10
        self.cur_scope = AstVarScope(scope_name, scope_type, parent_scope=self.cur_scope)
        return self.cur_scope

    def exit_scope(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.cur_scope.parent_scope is not None, "Call exit_scope in AstVarEnv when current scope doesn't have parent scope."
        self.cur_scope = self.cur_scope.parent_scope
        return self.cur_scope

    def get_parent_scope(self):
        if False:
            while True:
                i = 10
        assert self.cur_scope.parent_scope is not None, "Call parent_scope in AstVarEnv when current scope doesn't have parent scope."
        return self.cur_scope.parent_scope

    def add_var_type(self, var_name, node_var_type):
        if False:
            return 10
        self.cur_scope.add_var_type(var_name, node_var_type)

    def set_var_type(self, var_name, node_var_type):
        if False:
            return 10
        self.cur_scope.set_var_type(var_name, node_var_type)

    def get_var_type(self, var_name):
        if False:
            return 10
        return self.cur_scope.get_var_type(var_name)

    def get_scope_var_type(self):
        if False:
            return 10
        '\n        Returns a dict mapping from variable name to type. Used for debug and\n        test.\n        '
        cur_scope_dict = {}
        for name in self.cur_scope.name_to_id:
            node_var_type = self.cur_scope.get_var_type(name)
            cur_scope_dict[name] = node_var_type
        return cur_scope_dict

class StaticAnalysisVisitor:
    """
    A class that does static analysis
    """

    def __init__(self, ast_root=None):
        if False:
            while True:
                i = 10
        if ast_root is not None:
            self.run(ast_root)

    def run(self, ast_root):
        if False:
            while True:
                i = 10
        self.node_wrapper_root = None
        self.ancestor_wrappers = []
        self.node_to_wrapper_map = {}
        self.var_env = AstVarEnv()
        self.dfs_visit(ast_root)

    def dfs_visit(self, node):
        if False:
            return 10
        if node not in self.node_to_wrapper_map:
            cur_wrapper = AstNodeWrapper(node)
            self.node_to_wrapper_map[node] = cur_wrapper
        else:
            cur_wrapper = self.node_to_wrapper_map[node]
        if self.node_wrapper_root is None:
            self.node_wrapper_root = cur_wrapper
        if len(self.ancestor_wrappers) != 0:
            last_wrapper = self.ancestor_wrappers[-1]
            last_wrapper.children.append(cur_wrapper)
            cur_wrapper.parent = last_wrapper
        self.ancestor_wrappers.append(cur_wrapper)
        for child in gast.iter_child_nodes(node):
            if isinstance(child, (gast.FunctionDef, gast.AsyncFunctionDef)):
                self.var_env.enter_scope(child.name, AstVarScope.SCOPE_TYPE_FUNCTION)
                func_type = self.dfs_visit(child)
                self.var_env.exit_scope()
            else:
                self.dfs_visit(child)
        self.ancestor_wrappers.pop()
        cur_wrapper.node_var_type = self._get_node_var_type(cur_wrapper)
        return cur_wrapper.node_var_type

    def get_node_wrapper_root(self):
        if False:
            i = 10
            return i + 15
        return self.node_wrapper_root

    def get_node_to_wrapper_map(self):
        if False:
            i = 10
            return i + 15
        return self.node_to_wrapper_map

    def get_var_env(self):
        if False:
            i = 10
            return i + 15
        return self.var_env

    def is_tensor_node(self, node):
        if False:
            i = 10
            return i + 15
        tensor_types = {NodeVarType.TENSOR, NodeVarType.PADDLE_RETURN_TYPES}
        node_wrapper = self.node_to_wrapper_map.get(node, None)
        if node_wrapper is None:
            return False
        if node_wrapper.node_var_type & tensor_types:
            return True

    def _get_constant_node_type(self, node):
        if False:
            while True:
                i = 10
        assert isinstance(node, gast.Constant), 'Type of input node should be gast.Constant, but received %s' % type(node)
        if node.value is None:
            return {NodeVarType.NONE}
        if isinstance(node.value, bool):
            return {NodeVarType.BOOLEAN}
        if isinstance(node.value, int):
            return {NodeVarType.INT}
        if isinstance(node.value, float):
            return {NodeVarType.FLOAT}
        if isinstance(node.value, str):
            return {NodeVarType.STRING}
        return {NodeVarType.UNKNOWN}

    def _get_node_var_type(self, cur_wrapper):
        if False:
            for i in range(10):
                print('nop')
        node = cur_wrapper.node
        if isinstance(node, gast.Constant):
            return self._get_constant_node_type(node)
        if isinstance(node, gast.BoolOp):
            return {NodeVarType.BOOLEAN}
        if isinstance(node, gast.Compare):
            return {NodeVarType.BOOLEAN}
        if isinstance(node, gast.Dict):
            return {NodeVarType.DICT}
        if isinstance(node, gast.Set):
            return {NodeVarType.SET}
        if isinstance(node, gast.UnaryOp):
            return self.node_to_wrapper_map[node.operand].node_var_type
        if isinstance(node, gast.BinOp):
            left_type = self.node_to_wrapper_map[node.left].node_var_type
            right_type = self.node_to_wrapper_map[node.right].node_var_type
            result_type = set()
            for l in left_type:
                for r in right_type:
                    result_type.add(NodeVarType.binary_op_output_type(l, r))
            return result_type
        if isinstance(node, gast.Assign):
            ret_type = self.node_to_wrapper_map[node.value].node_var_type
            for target in node.targets:
                if isinstance(target, gast.Name):
                    self.node_to_wrapper_map[target].node_var_type = ret_type
                    self.var_env.set_var_type(target.id, ret_type)
                elif isinstance(target, gast.Tuple):
                    for sub_target in target.elts:
                        if isinstance(sub_target, gast.Name):
                            self.node_to_wrapper_map[sub_target].node_var_type = ret_type
                            self.var_env.set_var_type(sub_target.id, ret_type)
            return ret_type
        if isinstance(node, gast.AnnAssign):
            ret_type = {NodeVarType.type_from_annotation(node.annotation)}
            if node.value:
                node_value_type = self.node_to_wrapper_map[node.value].node_var_type
                if not node_value_type & {NodeVarType.UNKNOWN, NodeVarType.STATEMENT}:
                    ret_type = node_value_type
            if isinstance(node.target, gast.Name):
                self.node_to_wrapper_map[node.target].node_var_type = ret_type
                self.var_env.set_var_type(node.target.id, ret_type)
            return ret_type
        if isinstance(node, gast.Name):
            if node.id == 'None':
                return {NodeVarType.NONE}
            if node.id in {'True', 'False'}:
                return {NodeVarType.BOOLEAN}
            parent_node_wrapper = cur_wrapper.parent
            if parent_node_wrapper and isinstance(parent_node_wrapper.node, gast.arguments):
                return self._get_func_argument_type(parent_node_wrapper, node)
            return self.var_env.get_var_type(node.id)
        if isinstance(node, gast.Return):
            if node.value is None:
                return {NodeVarType.NONE}
            return_type = self.node_to_wrapper_map[node.value].node_var_type
            assert self.var_env.cur_scope.scope_type == AstVarScope.SCOPE_TYPE_FUNCTION, 'Return at non-function scope'
            func_name = self.var_env.cur_scope.scope_name
            parent_scope = self.var_env.get_parent_scope()
            parent_scope.add_var_type(func_name, return_type)
            return return_type
        if isinstance(node, gast.Call):
            if is_dygraph_api(node):
                if isinstance(node.func, gast.Attribute):
                    if node.func.attr == 'to_variable':
                        return {NodeVarType.TENSOR}
            if is_paddle_api(node):
                return {NodeVarType.PADDLE_RETURN_TYPES}
            if is_numpy_api(node):
                return {NodeVarType.NUMPY_NDARRAY}
            if isinstance(node.func, gast.Name):
                return self.var_env.get_var_type(node.func.id)
        if isinstance(node, gast.Subscript):
            if self.is_tensor_node(node.value):
                return {NodeVarType.TENSOR}
        return {NodeVarType.STATEMENT}

    def _get_func_argument_type(self, parent_node_wrapper, node):
        if False:
            print('Hello World!')
        "\n        Returns type information by parsing annotation or default values.\n\n        For example:\n            1. parse by default values.\n                foo(x, y=1, z='s') -> x: UNKNOWN, y: INT, z: STR\n\n            2. parse by Py3 type annotation.\n                foo(x: Tensor, y: int, z: str) -> x: Tensor, y: INT, z: STR\n\n            3. parse by type annotation and default values.\n                foo(x: Tensor, y: int, z: str = 'abc') -> x: Tensor, y: INT, z: STR\n\n        NOTE: Currently, we only support Tensor, int, bool, float, str et.al.\n              Other complicate types will be supported later.\n        "
        assert isinstance(node, gast.Name)
        parent_node = parent_node_wrapper.node
        var_type = {NodeVarType.UNKNOWN}
        if node.annotation is not None:
            var_type = {NodeVarType.type_from_annotation(node.annotation)}
            self.var_env.set_var_type(node.id, var_type)
        if parent_node.defaults:
            index = index_in_list(parent_node.args, node)
            args_len = len(parent_node.args)
            if index != -1 and args_len - index <= len(parent_node.defaults):
                defaults_node = parent_node.defaults[index - args_len]
                if isinstance(defaults_node, gast.Constant):
                    var_type = self._get_constant_node_type(defaults_node)
                    self.var_env.set_var_type(node.id, var_type)
        return var_type