import os
from . import logging_utils
from .assert_transformer import AssertTransformer
from .base_transformer import BaseTransformer
from .basic_api_transformer import BasicApiTransformer, NameloadJstTransformer
from .break_continue_transformer import BreakContinueTransformer, BreakTransformOptimizer
from .call_transformer import CallTransformer
from .cast_transformer import CastTransformer
from .create_variable_transformer import CreateVariableTransformer
from .decorator_transformer import DecoratorTransformer
from .early_return_transformer import EarlyReturnTransformer
from .ifelse_transformer import IfElseTransformer
from .logical_transformer import LogicalTransformer
from .loop_transformer import LoopTransformer
from .return_transformer import ReturnTransformer
from .tensor_shape_transformer import TensorShapeTransformer
from .tensorhook_transformer import RegisterHookTransformer
from .typehint_transformer import TypeHintTransformer
from .utils import ast_to_source_code
__all__ = []

def apply_optimization(transformers):
    if False:
        while True:
            i = 10
    "\n    Judge wheter to apply optimized transformation, such as BreakTransformOptimizer.\n    And not all optimized transformations are applied by default. It's controlled by\n    'export FLAGS_optim_transformation=1'\n    "
    flag = str(os.environ.get('FLAGS_optim_transformation')) in ['1', 'True', 'true']
    if flag:
        transformers.insert(3, BreakTransformOptimizer)

class DygraphToStaticAst(BaseTransformer):
    """
    Main class to transform Dygraph to Static Graph
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.translator_logger = logging_utils.TranslatorLogger()

    def get_static_ast(self, root):
        if False:
            return 10
        self.root = root
        self.decorate_func_name = None
        self.transfer_from_node_type(self.root)
        return self.root

    def _apply(self, transformer, node, log_level):
        if False:
            print('Hello World!')
        transformer(node).transform()
        self.translator_logger.log_transformed_code(log_level, self.root, transformer.__name__)

    def transfer_from_node_type(self, node):
        if False:
            while True:
                i = 10
        self.translator_logger.log(1, f'Source code: \n{ast_to_source_code(self.root)}')
        self.visit(node)
        transformers = [RegisterHookTransformer, EarlyReturnTransformer, BasicApiTransformer, TensorShapeTransformer, BreakContinueTransformer, ReturnTransformer, LogicalTransformer, CreateVariableTransformer, LoopTransformer, IfElseTransformer, AssertTransformer, CallTransformer, CastTransformer, DecoratorTransformer, NameloadJstTransformer, TypeHintTransformer]
        apply_optimization(transformers)
        for (index, transformer) in enumerate(transformers):
            self._apply(transformer, node, log_level=index + 1)
        self.translator_logger.log_transformed_code(logging_utils.LOG_AllTransformer, self.root, 'All Transformers')

    def visit_FunctionDef(self, node):
        if False:
            for i in range(10):
                print('nop')
        if self.decorate_func_name is None:
            self.decorate_func_name = node.name
        self.generic_visit(node)
        return node

    def get_module_name(self):
        if False:
            while True:
                i = 10
        '\n        Return the main function name which will be used as module name\n        in ast_to_func.\n        '
        assert self.decorate_func_name, 'decorate_func_name shall not be None.'
        return self.decorate_func_name