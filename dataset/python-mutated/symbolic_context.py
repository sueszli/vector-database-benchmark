from __future__ import annotations
from ..utils import log
from .compile_cache import CompileSIRCache
from .statement_ir import ApiStatement, CallStatement, LayerStatement, MethodStatement, StatementIR, StatementIRFactory, Symbol

class SymbolicTraceContext:
    """
    SymbolicTraceContext is a context manager, which is used to record the symbolic trace.

    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.reset()

    def reset(self):
        if False:
            return 10
        '\n        Reset the context.\n        '
        self.statement_factory = StatementIRFactory()
        self.sir_stack = [self.statement_factory.create()]

    @property
    def TOS(self):
        if False:
            return 10
        '\n        The top SIR of sir_stack.\n\n        Returns:\n            StatementIR: the top of stack.\n        '
        return self.sir_stack[-1]

    def call_SIR(self, sirname, inputs, outputs, stacks):
        if False:
            return 10
        '\n        Call a SIR, which is a subgraph.\n        '
        stmt = CallStatement(sirname, inputs, outputs, stacks)
        self.TOS.add_statement(stmt)

    def call_API(self, api, inputs, outputs, stacks):
        if False:
            print('Hello World!')
        '\n        Call a paddle api.\n        '
        assert callable(api), 'call_API must receive a paddle api.'
        stmt = ApiStatement(api, inputs, outputs, stacks)
        self.TOS.add_statement(stmt)

    def call_METHOD(self, method_name, inputs, outputs, stacks):
        if False:
            while True:
                i = 10
        '\n        Call a method of a api. The API here can be python or Paddle\n        '
        assert isinstance(method_name, str), 'call_METHOD must method api name. string.'
        assert isinstance(inputs[0][0], Symbol), 'call_METHOD must first augument must be Symbol Variable.'
        stmt = MethodStatement(method_name, inputs, outputs, stacks)
        self.TOS.add_statement(stmt)

    def call_LAYER(self, layer, inputs, outputs, stacks):
        if False:
            print('Hello World!')
        '\n        Call a layer of a api.\n        '
        stmt = LayerStatement(layer, inputs, outputs, stacks)
        self.TOS.add_statement(stmt)

    def get_sir(self, name: str):
        if False:
            i = 10
            return i + 15
        '\n        Get a SIR from statement_factory.\n\n        Args:\n            name (str): the name of SIR.\n\n        Returns:\n            StatementIR: the SIR.\n        '
        return self.statement_factory[name]

    def reset_TOS(self):
        if False:
            while True:
                i = 10
        '\n        Reset the TOS.\n        '
        self.sir_stack.pop()
        self.sir_stack.append(self.statement_factory.create())

    def replace_TOS(self, sir):
        if False:
            i = 10
            return i + 15
        '\n        Use deepcopyed sir to replace the TOS.\n        This function will update statment_factory.\n        '
        self.sir_stack.pop()
        self.sir_stack.append(sir)
        self.statement_factory.update(sir)

    def compile_do_nothing(self, ret_vals):
        if False:
            print('Hello World!')
        '\n        Return a dummy function, which will return an empty list.\n\n        Args:\n            ret_vals (list[Symbol]): the return values of the function.\n        '

        def dummy_func(*args, **kwargs):
            if False:
                print('Hello World!')
            return []
        dummy_stmt_ir = StatementIR('dummy_func')
        dummy_stmt_ir.outputs = []
        dummy_stmt_ir.inputs = []
        return (dummy_func, dummy_stmt_ir)

    def compile_fn(self, ret_vals, **kwargs):
        if False:
            print('Hello World!')
        '\n        start compile and return the python function, which must can be to_static without errors.\n        '
        cur_sir: StatementIR = self.TOS
        if len(cur_sir.statements) == 0:
            return self.compile_do_nothing(ret_vals)
        cur_sir.inputs = cur_sir.analyse_inputs()
        cur_sir.outputs = ret_vals
        log(2, 'start subgraph compile and execution.\n')
        log(2, self.TOS, '\n')
        static_func = CompileSIRCache()(self, cur_sir.name, **kwargs)
        return (static_func, cur_sir)