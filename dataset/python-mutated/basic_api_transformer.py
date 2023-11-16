import astor
from paddle.utils import gast
from . import utils
from .base_transformer import BaseTransformer
__all__ = []

class BasicApiTransformer(BaseTransformer):
    """
    Class to transform basic API from dygraph to static graph.
    """

    def __init__(self, root):
        if False:
            while True:
                i = 10
        self.root = root
        self.class_node_dict = {}

    def transform(self):
        if False:
            for i in range(10):
                print('nop')
        to_tensor_transformer = ToTensorTransformer(self.root)
        to_tensor_transformer.transform()
        attribute_transformer = AttributeJstTransformer(self.root)
        attribute_transformer.transform()
        self.visit(self.root)
        return self.root

    def visit_Assign(self, node):
        if False:
            i = 10
            return i + 15
        if self._update_class_node_dict(node):
            return None
        for child_node in gast.walk(node.value):
            if isinstance(child_node, gast.Call):
                self._visit_Call(child_node)
        return node

    def visit_Expr(self, node):
        if False:
            i = 10
            return i + 15
        value_node = node.value
        for child_node in gast.walk(value_node):
            if isinstance(child_node, gast.Call):
                if utils.is_dygraph_api(child_node):
                    return
                else:
                    self._visit_Call(child_node)
        return node

    def _visit_Call(self, node):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(node, gast.Call)
        func_name = astor.to_source(gast.gast_to_ast(node.func))
        if self._is_dygraph_forward(func_name):
            class_node = self._get_class_node(func_name)
            static_node = utils.to_static_ast(node, class_node)
            return static_node
        else:
            return node

    def _is_dygraph_forward(self, func_id):
        if False:
            return 10
        return func_id in self.class_node_dict

    def _get_class_node(self, func_id):
        if False:
            for i in range(10):
                print('nop')
        return self.class_node_dict[func_id]

    def _update_class_node_dict(self, node):
        if False:
            print('Hello World!')
        assert isinstance(node, gast.Assign)
        node_value = node.value
        if isinstance(node_value, gast.Call):
            if is_to_variable(node_value):
                return False
            if utils.is_dygraph_api(node_value):
                dygraph_api = node_value.func.attr
                if not utils.dygraph_class_to_static_api.get(dygraph_api):
                    return False
                utils.update_args_of_func(node_value, node_value, '__init__')
                target_str = astor.to_source(gast.gast_to_ast(node.targets[0]))
                self.class_node_dict[target_str] = node_value
                return True
        return False

class ToTensorTransformer(BaseTransformer):
    """
    Class to transform paddle.to_tensor and paddle.to_variable to paddle.assign
    """

    def __init__(self, node):
        if False:
            return 10
        assert isinstance(node, gast.AST), 'Input non-gast.AST node for the initialization of ToTensorTransformer.'
        self.root = node

    def transform(self):
        if False:
            i = 10
            return i + 15
        self.visit(self.root)
        return self.root

    def visit_Call(self, node):
        if False:
            return 10
        assert isinstance(node, gast.Call)
        if is_to_variable(node):
            node = to_assign_node(node)
        self.generic_visit(node)
        return node

class NameloadJstTransformer(BaseTransformer):
    """
    change name and attribute load to __jst.Ld(name) pattern.
    for example:
        a.dtype -->  __jst.Ld(__jst.Ld(a).dtype)

    In paddle science and deepxde, we have to support changing tensor into variable
    in arbitrary occasion such as global tensor.

    NOTE: we only deal with ctx=Load() case.
    """

    def __init__(self, root):
        if False:
            for i in range(10):
                print('nop')
        self.root = root

    def transform(self):
        if False:
            print('Hello World!')
        self.visit(self.root)
        return self.root

    def _surround_with_ld(self, node):
        if False:
            i = 10
            return i + 15
        node = gast.parse(f'_jst.Ld({utils.ast_to_source_code(node).strip()})').body[0].value
        return node

    def visit_Call(self, node):
        if False:
            i = 10
            return i + 15
        "\n        Can't convert name of function call, bacause this will affect CallTransformer.\n        "
        node.args = [self.visit(arg) for arg in node.args]
        node.func = self.visit(node.func)
        return node

    def visit_Attribute(self, node):
        if False:
            print('Hello World!')
        assert isinstance(node, gast.Attribute)
        assert isinstance(node.attr, str)
        if utils.ast_to_source_code(node).startswith('_jst.'):
            return node
        self.generic_visit(node)
        if isinstance(node.ctx, gast.Load):
            node = self._surround_with_ld(node)
        return node

    def visit_Name(self, node):
        if False:
            i = 10
            return i + 15
        assert isinstance(node, gast.Name)
        self.generic_visit(node)
        if isinstance(node.ctx, gast.Load):
            node = self._surround_with_ld(node)
        return node

class AttributeJstTransformer(BaseTransformer):
    """
    change some special attribute into __jst.XXX(obj, "attr_name") format.
    for example:
        a.size  -->  __jst.attr(a, "size")

    because `size` have different behavier when in dygraph / static graph mode
    NOTE: we only deal with ctx=Load() case.
    """

    def __init__(self, node):
        if False:
            i = 10
            return i + 15
        assert isinstance(node, gast.AST), 'Input non-gast.AST node for the initialization of ToTensorTransformer.'
        self.interested_name = {'size'}
        self.root = node

    def transform(self):
        if False:
            return 10
        self.visit(self.root)
        return self.root

    def visit_Attribute(self, node):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(node, gast.Attribute)
        assert isinstance(node.attr, str)
        if isinstance(node.ctx, gast.Load) and node.attr in self.interested_name:
            attr = node.attr
            value = node.value
            node = gast.parse(f'_jst.Attr({utils.ast_to_source_code(value).strip()}, "{attr}")').body[0].value
        self.generic_visit(node)
        return node

def is_to_variable(node):
    if False:
        while True:
            i = 10
    assert isinstance(node, gast.Call)
    api_name = utils.ast_to_source_code(node.func).strip()
    if utils.is_dygraph_api(node):
        return api_name.endswith('to_variable')
    return False

def to_assign_node(node):
    if False:
        return 10
    assert isinstance(node, gast.Call)
    assign_api = gast.parse('paddle.assign').body[0].value
    node.func = assign_api
    if node.args:
        node.args = [node.args[0]]
        node.keywords = []
    else:
        for (idx, kw) in enumerate(node.keywords):
            if kw.arg == 'value' or kw.arg == 'data':
                node.keywords[idx].arg = 'x'
                node.keywords = [node.keywords[idx]]
                node.args = []
                break
    return node