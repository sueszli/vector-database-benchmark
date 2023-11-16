from cinn import ir
from .. import core_api

class IRBuilder:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.ir_builder = core_api.ir.IRBuilder()

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self.ir_builder.EnterWithContext()
        return self

    def __exit__(self, ptype, value, trace) -> None:
        if False:
            print('Hello World!')
        if ptype is None and value is None:
            self.ir_builder.ExitWithContext()

    def get(self):
        if False:
            for i in range(10):
                print('nop')
        return self.ir_builder.get_result()

class IRContext:

    def __init__(self, ir_ctx):
        if False:
            for i in range(10):
                print('nop')
        self.ir_ctx = ir_ctx

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self.ir_ctx.EnterWithContext()

    def __exit__(self, ptype, value, trace) -> None:
        if False:
            i = 10
            return i + 15
        if ptype is None and value is None:
            self.ir_ctx.ExitWithContext()

class ScheduleBlockContext(IRContext):

    def __init__(self, name):
        if False:
            i = 10
            return i + 15
        self.ir_ctx = core_api.ir.IRContext.MakeScheduleBlockContext(name)

class LowerFuncContext(IRContext):

    def __init__(self, name):
        if False:
            print('Hello World!')
        self.ir_ctx = core_api.ir.IRContext.MakeLowerFunctionContext(name)

class ForContext(IRContext):

    def __init__(self, min, extent):
        if False:
            while True:
                i = 10
        self.ir_ctx = ir.Sequential(min, extent)

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        super().__enter__()
        return self.ir_ctx.get_for_loop_var()

class IfContext(IRContext):

    def __init__(self, expr):
        if False:
            for i in range(10):
                print('nop')
        self.ir_ctx = core_api.ir.IRContext.MakeIfContext(expr)

class ThenContext(IRContext):

    def __init__(self):
        if False:
            return 10
        self.ir_ctx = core_api.ir.IRContext.MakeThenContext()

class ElseContext(IRContext):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.ir_ctx = core_api.ir.IRContext.MakeElseContext()