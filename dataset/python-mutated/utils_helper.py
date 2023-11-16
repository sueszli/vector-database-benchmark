import inspect
import astor
import numpy as np
import paddle
from paddle import base
from paddle.base import dygraph, layers
from paddle.base.dygraph import to_variable
from paddle.utils import gast
from .ast_utils import ast_to_source_code
from .logging_utils import warn

def index_in_list(array_list, item):
    if False:
        i = 10
        return i + 15
    try:
        return array_list.index(item)
    except ValueError:
        return -1
PADDLE_MODULE_PREFIX = 'paddle.'
DYGRAPH_TO_STATIC_MODULE_PREFIX = 'paddle.jit.dy2static'
DYGRAPH_MODULE_PREFIX = 'paddle.base.dygraph'

def is_dygraph_api(node):
    if False:
        for i in range(10):
            print('nop')
    if is_api_in_module(node, DYGRAPH_TO_STATIC_MODULE_PREFIX):
        return False
    return is_api_in_module(node, DYGRAPH_MODULE_PREFIX)

def is_api_in_module(node, module_prefix):
    if False:
        print('Hello World!')
    assert isinstance(node, gast.Call), 'Input non-Call node for is_dygraph_api'
    func_node = node.func
    while isinstance(func_node, gast.Call):
        func_node = func_node.func
    func_str = astor.to_source(gast.gast_to_ast(func_node)).strip()
    try:
        import paddle.jit.dy2static as _jst
        from paddle import to_tensor
        return eval(f"_is_api_in_module_helper({func_str}, '{module_prefix}')")
    except Exception:
        return False

def _is_api_in_module_helper(obj, module_prefix):
    if False:
        for i in range(10):
            print('nop')
    m = inspect.getmodule(obj)
    return m is not None and m.__name__.startswith(module_prefix)

def is_numpy_api(node):
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(node, gast.Call), 'Input non-Call node for is_numpy_api'
    func_str = astor.to_source(gast.gast_to_ast(node.func))
    try:
        module_result = eval("_is_api_in_module_helper({}, '{}')".format(func_str, 'numpy'))
        return module_result or (func_str.startswith('numpy.') or func_str.startswith('np.'))
    except Exception:
        return False

def is_paddle_api(node):
    if False:
        for i in range(10):
            print('nop')
    return is_api_in_module(node, PADDLE_MODULE_PREFIX)

class NodeVarType:
    """
    Enum class of python variable types. We have to know some variable types
    during compile time to transfer AST. For example, a string variable and a
    tensor variable in if clause may lead to different conversion from dygraph
    to static graph.
    """
    ERROR = -1
    UNKNOWN = 0
    STATEMENT = 1
    CALLABLE = 2
    NONE = 100
    BOOLEAN = 101
    INT = 102
    FLOAT = 103
    STRING = 104
    TENSOR = 105
    NUMPY_NDARRAY = 106
    LIST = 200
    SET = 201
    DICT = 202
    PADDLE_DYGRAPH_API = 300
    PADDLE_CONTROL_IF = 301
    PADDLE_CONTROL_WHILE = 302
    PADDLE_CONTROL_FOR = 303
    PADDLE_RETURN_TYPES = 304
    TENSOR_TYPES = {TENSOR, PADDLE_RETURN_TYPES}
    Annotation_map = {'Tensor': TENSOR, 'paddle.Tensor': TENSOR, 'int': INT, 'float': FLOAT, 'bool': BOOLEAN, 'str': STRING}

    @staticmethod
    def binary_op_output_type(in_type1, in_type2):
        if False:
            while True:
                i = 10
        if in_type1 == in_type2:
            return in_type1
        if in_type1 == NodeVarType.UNKNOWN:
            return in_type2
        if in_type2 == NodeVarType.UNKNOWN:
            return in_type1
        supported_types = [NodeVarType.BOOLEAN, NodeVarType.INT, NodeVarType.FLOAT, NodeVarType.NUMPY_NDARRAY, NodeVarType.TENSOR, NodeVarType.PADDLE_RETURN_TYPES]
        if in_type1 not in supported_types:
            return NodeVarType.UNKNOWN
        if in_type2 not in supported_types:
            return NodeVarType.UNKNOWN
        forbidden_types = [NodeVarType.NUMPY_NDARRAY, NodeVarType.TENSOR]
        if in_type1 in forbidden_types and in_type2 in forbidden_types:
            return NodeVarType.UNKNOWN
        return max(in_type1, in_type2)

    @staticmethod
    def type_from_annotation(annotation):
        if False:
            while True:
                i = 10
        annotation_str = ast_to_source_code(annotation).strip()
        if annotation_str in NodeVarType.Annotation_map:
            return NodeVarType.Annotation_map[annotation_str]
        warn("Currently we don't support annotation: %s" % annotation_str)
        return NodeVarType.UNKNOWN

def set_dynamic_shape(variable, shape_list):
    if False:
        while True:
            i = 10
    if paddle.base.dygraph.base.in_to_static_mode():
        assert isinstance(variable, paddle.base.framework.Variable), 'In to_static mode, variable must be a Variable.'
        variable.desc.set_shape(shape_list)
    else:
        return